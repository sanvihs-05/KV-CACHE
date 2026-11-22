"""GPU-resident int8 KV cache + int8 decode attention for HF transformers (Qwen2 / Llama-style models).

    from kvwear.cuda.int8_cache import Int8GpuCache, use_int8_attention
    cache = Int8GpuCache.for_model(model, max_tokens=32768)
    with use_int8_attention(model, cache):            # prefill: SDPA; decode: kvwear int8 kernel
        out = model(input_ids=prompt, past_key_values=cache, use_cache=True, logits_to_keep=1)
        ...

Each layer's K/V live on the GPU in kvwear's int8 block format (one block per layer, capacity
`max_tokens`): half the memory of fp16, and the decode-attention kernel reads them directly
(no dequantize pass). New tokens are quantized in place by `append_int8` (kvcache_kernels.cu).
Prefill attention runs PyTorch SDPA on the fresh fp16 K/V; decode steps (one query token) run
`paged_int8_decode` from kvattn_kernels.cu.
"""
from __future__ import annotations

import contextlib
import math
import threading

import torch

from . import _HERE, _find_cuda_home, ext

_cache_ext = None
_lock = threading.Lock()


def cache_ext():
    global _cache_ext
    if _cache_ext is None:
        with _lock:
            if _cache_ext is None:
                import os
                os.environ.setdefault("CUDA_HOME", _find_cuda_home() or "")
                major, minor = torch.cuda.get_device_capability()
                os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{major}.{minor}")
                from torch.utils.cpp_extension import load
                _cache_ext = load(name="kvwear_cuda_cache",
                                  sources=[str(_HERE / "kvcache_bindings.cpp"), str(_HERE / "kvcache_kernels.cu")],
                                  extra_cflags=["-O3"], extra_cuda_cflags=["-O3", "-lineinfo", "--fmad=false"])
    return _cache_ext


try:
    from transformers.cache_utils import Cache as _HFCache
except Exception:  # pragma: no cover
    _HFCache = object


class Int8GpuCache(_HFCache):
    """transformers Cache whose storage is int8 on the GPU (kvwear block format, group = 64)."""

    def __init__(self, layers: int, hkv: int, head_dim: int, max_tokens: int, device="cuda", group: int = 64):
        super().__init__()
        g = group if head_dim % group == 0 else head_dim
        self.layers, self.hkv, self.d, self.tmax, self.g = layers, hkv, head_dim, max_tokens, g
        self.groups = 2 * hkv * max_tokens * (head_dim // g)
        nbytes = 2 * self.groups + 2 * hkv * max_tokens * head_dim
        nbytes = (nbytes + 255) // 256 * 256
        self.buf = torch.empty((layers, nbytes), dtype=torch.uint8, device=device)
        self.offs = torch.zeros(1, dtype=torch.int64, device=device)
        self.lens = [0] * layers
        self._seen_tokens = 0

    @classmethod
    def for_model(cls, model, max_tokens: int, group: int = 64):
        c = model.config
        hd = getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads
        return cls(c.num_hidden_layers, c.num_key_value_heads, hd, max_tokens, model.device, group)

    # --- transformers Cache API
    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        t0 = self.lens[layer_idx]
        k, v = key_states[0].contiguous(), value_states[0].contiguous()          # [Hkv, t, D]
        tnew = k.shape[1]
        cache_ext().append_int8(k, v, self.buf[layer_idx], self.tmax, t0, self.g)
        self.lens[layer_idx] = t0 + tnew
        if layer_idx == 0:
            self._seen_tokens += tnew
        if tnew > 1 and t0 > 0:
            # chunked prefill: attention needs the whole (dequantized) past plus the new rows
            kp, vp = cache_ext().gather_int8(self.buf[layer_idx], self.hkv, t0, self.d, self.tmax, self.g)
            return (torch.cat([kp, k], 1).unsqueeze(0), torch.cat([vp, v], 1).unsqueeze(0))
        return key_states, value_states                                           # decode ignores these

    def get_seq_length(self, layer_idx: int | None = 0) -> int:
        return self.lens[layer_idx or 0]

    def get_max_cache_shape(self):
        return None

    @property
    def nbytes(self) -> int:
        return self.buf.numel()

    def used_bytes(self) -> int:
        """Bytes that hold actual tokens (int8 values + fp16 scales)."""
        per_token = self.layers * 2 * self.hkv * (self.d + 2 * (self.d // self.g))
        return per_token * self.lens[0]

    def decode(self, q, layer_idx: int, scale: float):
        """q: [Hq, D] fp16 -> attention output [Hq, D] over the cached tokens of this layer."""
        T = self.lens[layer_idx]
        hq = q.shape[0]
        from .attention import auto_tps
        variant = 3 if (hq > self.hkv and T >= 8192) else 2
        tps = auto_tps(T, self.hkv if variant == 3 else hq)
        return ext().paged_int8_decode(q, self.buf[layer_idx], self.offs, self.groups, 0, self.hkv, self.tmax, T,
                                       self.g, tps, scale, variant)


_ACTIVE: dict = {"cache": None}


def int8_attention_forward(module, query, key, value, attention_mask, dropout=0.0, scaling=None,
                           sliding_window=None, **kwargs):
    """transformers attention-interface function: int8 kernel for single-token decode, SDPA otherwise."""
    cache = _ACTIVE["cache"]
    B, Hq, qlen, D = query.shape
    if cache is not None and qlen == 1 and B == 1:
        scale = scaling if scaling is not None else 1.0 / math.sqrt(D)
        out = cache.decode(query[0, :, 0, :].contiguous(), module.layer_idx, scale)   # [Hq, D]
        return out.view(1, 1, Hq, D), None
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    return sdpa_attention_forward(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling,
                                  sliding_window=sliding_window, **kwargs)


_REGISTERED = False


@contextlib.contextmanager
def use_int8_attention(model, cache: Int8GpuCache):
    """Route the model's attention through int8_attention_forward while `cache` is active."""
    global _REGISTERED
    from transformers import AttentionInterface
    if not _REGISTERED:
        AttentionInterface.register("kvwear_int8", int8_attention_forward)
        _REGISTERED = True
    old = model.config._attn_implementation
    model.config._attn_implementation = "kvwear_int8"
    _ACTIVE["cache"] = cache
    try:
        yield
    finally:
        _ACTIVE["cache"] = None
        model.config._attn_implementation = old
