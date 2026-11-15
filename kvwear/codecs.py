"""Block codecs: how a KV block is turned into bytes on the SSD.

A KV block is a float16 array of shape [layers, 2 (K/V), kv_heads, tokens, head_dim].
Quantization groups run along head_dim (the last axis).

  fp16        raw bytes, the baseline
  int8        symmetric per-group absmax, 1 byte/value + fp16 scale per group  (~0.52x)
  int4        asymmetric per-group min/max, 2 values/byte + fp16 scale & min   (~0.28x,
              the same scheme as FlexLLMGen's --compress-cache: 4 bits, group 64)
  <x>+zstd    lossless compression of the encoded bytes (falls back to zlib, with a
              warning, if the zstandard package is missing)

The math runs on torch CPU ops when torch is installed (multi-threaded, ~10x faster than
numpy), else numpy. Both produce the same byte format. Every block carries a small
header, so decode() doesn't need to know the shape.
"""
from __future__ import annotations

import struct
import warnings
import zlib

import numpy as np

try:
    import zstandard as _zstd
except Exception:  # pragma: no cover - optional dependency
    _zstd = None

try:
    import torch as _torch
except Exception:  # pragma: no cover
    _torch = None

_MAGIC = b"KVB1"
_METHODS = {"fp16": 0, "int8": 1, "int4": 2}
_METHOD_NAMES = {v: k for k, v in _METHODS.items()}
_HDR = "<BBBH"


def _pick_group(dim: int, group: int) -> int:
    return group if dim % group == 0 else dim


def _header(method: str, shape: tuple, group: int, comp: int) -> bytes:
    return _MAGIC + struct.pack(_HDR, _METHODS[method], comp, len(shape), group) + \
        struct.pack(f"<{len(shape)}I", *shape)


def _parse_header(buf):
    if bytes(buf[:4]) != _MAGIC:
        raise ValueError("not a KV block")
    method, comp, ndim, group = struct.unpack_from(_HDR, buf, 4)
    off = 4 + struct.calcsize(_HDR)
    shape = struct.unpack_from(f"<{ndim}I", buf, off)
    return _METHOD_NAMES[method], comp, tuple(shape), group, off + 4 * ndim


# --------------------------------------------------------------- quantizers (torch / numpy)

def _enc_int8(arr: np.ndarray, g: int, use_torch: bool) -> bytes:
    if use_torch:
        x = _torch.from_numpy(arr).view(-1, g).float()
        s = x.abs().amax(1, keepdim=True) / 127.0
        s[s == 0] = 1.0
        q = _torch.round(x / s).clamp_(-127, 127).to(_torch.int8)
        return s.half().numpy().tobytes() + q.numpy().tobytes()
    x = arr.astype(np.float32).reshape(-1, g)
    s = np.abs(x).max(axis=1, keepdims=True) / 127.0
    s[s == 0] = 1.0
    q = np.clip(np.rint(x / s), -127, 127).astype(np.int8)
    return s.astype(np.float16).tobytes() + q.tobytes()


def _enc_int4(arr: np.ndarray, g: int, use_torch: bool) -> bytes:
    if use_torch:
        x = _torch.from_numpy(arr).view(-1, g).float()
        mn = x.amin(1, keepdim=True)
        s = (x.amax(1, keepdim=True) - mn) / 15.0
        s[s == 0] = 1.0
        q = _torch.round((x - mn) / s).clamp_(0, 15).to(_torch.uint8).view(-1)
        if q.numel() % 2:
            q = _torch.cat([q, q.new_zeros(1)])
        packed = q[0::2] | (q[1::2] << 4)
        return s.half().numpy().tobytes() + mn.half().numpy().tobytes() + packed.numpy().tobytes()
    x = arr.astype(np.float32).reshape(-1, g)
    mn = x.min(axis=1, keepdims=True)
    s = (x.max(axis=1, keepdims=True) - mn) / 15.0
    s[s == 0] = 1.0
    q = np.clip(np.rint((x - mn) / s), 0, 15).astype(np.uint8).reshape(-1)
    if q.size % 2:
        q = np.append(q, np.uint8(0))
    packed = (q[0::2] | (q[1::2] << 4)).astype(np.uint8)
    return s.astype(np.float16).tobytes() + mn.astype(np.float16).tobytes() + packed.tobytes()


def _dec_int8(payload, n: int, g: int, use_torch: bool) -> np.ndarray:
    groups = n // g
    if use_torch:
        buf = payload if isinstance(payload, bytearray) else bytearray(payload)
        s = _torch.frombuffer(buf, dtype=_torch.float16, count=groups).float().view(-1, 1)
        q = _torch.frombuffer(buf, dtype=_torch.int8, count=n, offset=2 * groups).view(groups, g)
        return (q.float() * s).half().numpy()
    s = np.frombuffer(payload, dtype=np.float16, count=groups).astype(np.float32)
    q = np.frombuffer(payload, dtype=np.int8, count=n, offset=2 * groups).astype(np.float32)
    return (q.reshape(groups, g) * s[:, None]).astype(np.float16)


def _dec_int4(payload, n: int, g: int, use_torch: bool) -> np.ndarray:
    groups = n // g
    npk = (n + 1) // 2
    if use_torch:
        buf = payload if isinstance(payload, bytearray) else bytearray(payload)
        s = _torch.frombuffer(buf, dtype=_torch.float16, count=groups).float().view(-1, 1)
        mn = _torch.frombuffer(buf, dtype=_torch.float16, count=groups, offset=2 * groups).float().view(-1, 1)
        packed = _torch.frombuffer(buf, dtype=_torch.uint8, count=npk, offset=4 * groups)
        q = _torch.stack([packed & 0x0F, packed >> 4], dim=1).view(-1)[:n]
        return (q.float().view(groups, g) * s + mn).half().numpy()
    s = np.frombuffer(payload, dtype=np.float16, count=groups).astype(np.float32)
    mn = np.frombuffer(payload, dtype=np.float16, count=groups, offset=2 * groups).astype(np.float32)
    packed = np.frombuffer(payload, dtype=np.uint8, count=npk, offset=4 * groups)
    q = np.empty(npk * 2, dtype=np.uint8)
    q[0::2] = packed & 0x0F
    q[1::2] = packed >> 4
    return (q[:n].astype(np.float32).reshape(groups, g) * s[:, None] + mn[:, None]).astype(np.float16)


# --------------------------------------------------------------- codec

class Codec:
    """Encode/decode float16 KV blocks. `lossy` codecs change values slightly."""

    def __init__(self, method: str = "fp16", group: int = 64, compress: str | None = None,
                 level: int = 3, backend: str = "auto"):
        if method not in _METHODS:
            raise ValueError(f"unknown quantization '{method}'")
        if compress not in (None, "zstd", "zlib"):
            raise ValueError(f"unknown compressor '{compress}'")
        if compress == "zstd" and _zstd is None:
            warnings.warn("zstandard not installed: '+zstd' falls back to zlib (pip install zstandard)")
            compress = "zlib"
        self.method, self.group, self.compress, self.level = method, group, compress, level
        self.use_torch = _torch is not None and backend in ("auto", "torch")

    @property
    def name(self) -> str:
        return self.method + (f"+{self.compress}" if self.compress else "")

    @property
    def lossy(self) -> bool:
        return self.method != "fp16"

    def encode(self, arr: np.ndarray) -> bytes:
        arr = np.ascontiguousarray(arr, dtype=np.float16)
        g = _pick_group(arr.shape[-1], self.group)
        if self.method == "fp16":
            payload = arr.tobytes()
        elif self.method == "int8":
            payload = _enc_int8(arr, g, self.use_torch)
        else:
            payload = _enc_int4(arr, g, self.use_torch)
        comp = 0
        if self.compress == "zstd":
            payload, comp = _zstd.ZstdCompressor(level=self.level).compress(payload), 2
        elif self.compress == "zlib":
            payload, comp = zlib.compress(payload, self.level), 1
        return _header(self.method, arr.shape, g, comp) + payload

    @staticmethod
    def decode(data: bytes | bytearray, backend: str = "auto") -> np.ndarray:
        use_torch = _torch is not None and backend in ("auto", "torch")
        method, comp, shape, g, off = _parse_header(data)
        if comp == 2:
            if _zstd is None:
                raise RuntimeError("block was written with zstd; pip install zstandard")
            payload = bytearray(_zstd.ZstdDecompressor().decompress(bytes(data[off:])))
        elif comp == 1:
            payload = bytearray(zlib.decompress(memoryview(data)[off:]))
        else:
            payload = data[off:]                   # bytearray slice stays writable for torch
        n = int(np.prod(shape))
        if method == "fp16":
            return np.frombuffer(payload, dtype=np.float16, count=n).reshape(shape).copy()
        if method == "int8":
            return _dec_int8(payload, n, g, use_torch).reshape(shape)
        return _dec_int4(payload, n, g, use_torch).reshape(shape)


def make_codec(spec: str) -> Codec:
    """'fp16', 'int8', 'int4', optionally with '+zstd' / '+zlib'."""
    parts = spec.split("+")
    return Codec(parts[0] or "fp16", compress=parts[1] if len(parts) > 1 else None)
