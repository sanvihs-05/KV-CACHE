"""CUDA acceleration for the KV block codec and the cache-assembly path.

    from kvwear import cuda as kvcuda
    kvcuda.available()                    # True if a GPU + nvcc are present and the build works
    ext = kvcuda.ext()                    # the compiled extension (JIT-built once, then cached)

The extension (kvquant_kernels.cu + kvquant_bindings.cpp) is compiled with nvcc on first use
via torch.utils.cpp_extension and cached under ~/.cache/torch_extensions.

What runs on the GPU:
  * GpuCodec     int8/int4 quantization of KV blocks. Same byte format as kvwear.codecs.Codec
                 (byte-identical, enforced by tests), so files written by either are interchangeable.
                 Runs on its own CUDA stream with pinned staging buffers, so the SSD writer
                 thread's encode overlaps with the model's compute on the default stream.
  * BlockLoader  assembles a request's cached prefix into the attention cache: one pinned
                 host->device copy of all (still compressed) blocks, then ONE kernel launch that
                 dequantizes every block and scatters it to its token offset.
"""
from __future__ import annotations

import os
import shutil
import threading
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ext = None
_err: Exception | None = None
_lock = threading.Lock()
METHOD_IDS = {"fp16": 0, "int8": 1, "int4": 2}


def _find_cuda_home() -> str | None:
    for c in (os.environ.get("CUDA_HOME"), os.environ.get("CUDA_PATH"), "/usr/local/cuda", "/usr/local/cuda-12.1"):
        if c and Path(c, "bin", "nvcc").exists():
            return c
    nvcc = shutil.which("nvcc")
    return str(Path(nvcc).resolve().parent.parent) if nvcc else None


def ext(verbose: bool = False):
    """Build (first call) and return the compiled extension module."""
    global _ext, _err
    if _ext is not None:
        return _ext
    with _lock:
        if _ext is not None:
            return _ext
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        home = _find_cuda_home()
        if home is None:
            raise RuntimeError("nvcc not found: install the CUDA toolkit (e.g. apt install cuda-toolkit-12-1)")
        os.environ.setdefault("CUDA_HOME", home)
        major, minor = torch.cuda.get_device_capability()
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{major}.{minor}")
        from torch.utils.cpp_extension import load
        _ext = load(
            name="kvwear_cuda",
            sources=[str(_HERE / "kvquant_bindings.cpp"), str(_HERE / "kvquant_kernels.cu"),
                     str(_HERE / "kvattn_kernels.cu"), str(_HERE / "kvprof_kernels.cu")],
            extra_include_paths=[str(_HERE)],
            extra_cflags=["-O3"],
            # no --use_fast_math: division and rounding must match the CPU codec bit for bit
            extra_cuda_cflags=["-O3", "-lineinfo", "--fmad=false"],
            verbose=verbose,
        )
        return _ext


def available() -> bool:
    global _err
    try:
        ext()
        return True
    except Exception as e:  # pragma: no cover - depends on the machine
        _err = e
        return False


def unavailable_reason() -> str:
    return "" if _err is None else f"{type(_err).__name__}: {_err}"


# --------------------------------------------------------------------------- pinned staging

class PinnedBuffer:
    """Grow-only page-locked host buffer. Pinned memory lets cudaMemcpyAsync run as true DMA
    (no hidden staging copy by the driver) and overlap with kernels on other streams."""

    def __init__(self):
        self.t = None

    def get(self, nbytes: int):
        import torch
        if self.t is None or self.t.numel() < nbytes:
            cap = max(nbytes, 1 << 20)
            cap = 1 << (cap - 1).bit_length()          # round up to a power of two
            self.t = torch.empty(cap, dtype=torch.uint8, pin_memory=True)
        return self.t[:nbytes]


def _as_u8(arr: np.ndarray):
    import torch
    return torch.from_numpy(np.ascontiguousarray(arr).view(np.uint8).reshape(-1))


# --------------------------------------------------------------------------- codec

class GpuCodec:
    """int8 / int4 KV-block quantization on the GPU (payload only; headers are added by Codec)."""

    def __init__(self, method: str, variant: int = 4):
        if method not in ("int8", "int4"):
            raise ValueError("GpuCodec handles int8 and int4 (fp16 needs no kernel)")
        self.method, self.variant = method, variant
        self._tls = threading.local()

    def _state(self):
        import torch
        st = getattr(self._tls, "st", None)
        if st is None:
            st = self._tls.st = {
                "stream": torch.cuda.Stream(),
                "pin_in": PinnedBuffer(), "pin_out": PinnedBuffer(),
            }
        return st

    def quantize_device(self, x, g: int):
        """x: contiguous CUDA float16 tensor -> CUDA uint8 payload (runs on the current stream)."""
        e = ext()
        return (e.quant_int8 if self.method == "int8" else e.quant_int4)(x.reshape(-1), g, self.variant)

    def encode_payload(self, arr: np.ndarray, g: int) -> bytes:
        """Host fp16 block -> payload bytes. H2D, kernel and D2H all run on this thread's own
        stream, so they overlap with whatever the default stream (the model) is doing."""
        import torch
        st = self._state()
        src = _as_u8(arr)
        nb = src.numel()
        with torch.cuda.stream(st["stream"]):
            pin_in = st["pin_in"].get(nb)
            pin_in.copy_(src)                                   # pageable -> pinned (CPU memcpy)
            dev = pin_in.to("cuda", non_blocking=True)          # DMA, async on our stream
            payload = self.quantize_device(dev.view(torch.float16), g)
            pin_out = st["pin_out"].get(payload.numel())
            pin_out.copy_(payload, non_blocking=True)           # DMA back, async
            st["stream"].synchronize()
        return pin_out.numpy().tobytes()

    @staticmethod
    def dequantize_device(payload, n: int, g: int, method: str):
        return ext().dequant(payload, n, g, METHOD_IDS[method])


# --------------------------------------------------------------------------- cache assembly

class BlockLoader:
    """Builds the [R, Ttot, D] attention-cache tensor for one request from a list of blocks.

    Each block is either a host float16 numpy array [L, 2, H, Tb, D] (RAM-tier hit) or an
    EncodedBlock (SSD hit, still quantized). All blocks are packed into one pinned buffer,
    copied with ONE async host->device transfer, and decoded by ONE kernel launch that writes
    each block straight to its token offset. That replaces: per-block CPU decode, a CPU
    concatenate, a pageable H2D copy and a per-layer .contiguous() copy.
    """

    ALIGN = 16

    def __init__(self):
        self.pin = PinnedBuffer()
        self._dma_done = None             # CUDA event: last request's copies out of `pin` finished

    def assemble(self, blocks, device="cuda"):
        import torch
        if self._dma_done is not None:    # never overwrite pinned memory a DMA may still be reading
            self._dma_done.synchronize()
        first = blocks[0]
        shape = tuple(first.shape)                          # [L, 2, H, Tb, D]
        L, two, H, Tb, D = shape
        R = L * two * H
        n = R * Tb * D
        Ttot = Tb * len(blocks)

        parts, offs, methods, g = [], [], [], 0
        total = 0
        for b in blocks:
            if tuple(b.shape) != shape:
                raise ValueError("all blocks of a request must have the same shape")
            if isinstance(b, np.ndarray):
                buf, m = b, 0
            else:                                           # EncodedBlock
                buf, m = b.payload, METHOD_IDS[b.method]
                if m:
                    if g and b.group != g:
                        raise ValueError("mixed group sizes in one request")
                    g = b.group
            mv = memoryview(buf).cast("B") if not isinstance(buf, np.ndarray) else None
            size = buf.nbytes if isinstance(buf, np.ndarray) else len(mv)
            offs.append(total)
            methods.append(m)
            parts.append((buf, mv, size))
            total += (size + self.ALIGN - 1) // self.ALIGN * self.ALIGN
        g = g or D

        # Pipeline: as soon as a block is in the pinned buffer, start its DMA; the copy engine moves
        # block i while the CPU memcpy's block i+1, so the load costs ~max(memcpy, DMA), not the sum.
        pin = self.pin.get(total)
        pin_np = pin.numpy()
        dev = torch.empty(total, dtype=torch.uint8, device=device)
        for (buf, mv, size), off in zip(parts, offs):
            if mv is None:
                pin_np[off:off + size] = np.ascontiguousarray(buf).view(np.uint8).reshape(-1)
            else:
                pin_np[off:off + size] = np.frombuffer(mv, dtype=np.uint8, count=size)
            dev[off:off + size].copy_(pin[off:off + size], non_blocking=True)
        offs_d = torch.tensor(offs, dtype=torch.int64).to(device, non_blocking=True)
        toffs_d = torch.arange(0, Ttot, Tb, dtype=torch.int32).to(device, non_blocking=True)
        meth_d = torch.tensor(methods, dtype=torch.int32).to(device, non_blocking=True)
        dest = torch.empty((R, Ttot, D), dtype=torch.float16, device=device)
        ext().dequant_scatter(dev, offs_d, toffs_d, meth_d, dest, Tb, g)
        self._dma_done = torch.cuda.Event()
        self._dma_done.record()
        return dest.view(L, two, H, Ttot, D)
