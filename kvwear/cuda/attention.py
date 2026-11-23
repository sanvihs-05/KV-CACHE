"""Paged decode attention over int8 KV blocks (kvwear's on-disk block format), no dequantize pass.

    kv = PagedInt8KV.from_payloads(datas)          # list of int8 block files (bytes) -> GPU
    out = kv.decode(q, layer)                      # q: [Hq, D] fp16 -> [Hq, D] fp16

The block table is just the byte offset of each block's payload inside one GPU buffer, so
blocks can be appended as they arrive from the SSD without ever expanding them to fp16.
"""
from __future__ import annotations

import math

import numpy as np

from . import ext
from ..codecs import EncodedBlock


class PagedInt8KV:
    ALIGN = 256

    def __init__(self, base, offs, shape, layers, hkv, tb, d, g):
        self.base, self.offs = base, offs
        self.shape, self.layers, self.hkv, self.tb, self.d, self.g = shape, layers, hkv, tb, d, g
        self.groups = layers * 2 * hkv * tb * (d // g)
        self.tokens = tb * offs.numel()

    @classmethod
    def from_payloads(cls, datas, device="cuda"):
        import torch
        blks = [d if isinstance(d, EncodedBlock) else EncodedBlock(bytearray(d)) for d in datas]
        if not blks or any(b.method != "int8" for b in blks):
            raise ValueError("PagedInt8KV needs int8 blocks")
        shape = tuple(blks[0].shape)
        L, two, H, Tb, D = shape
        g = blks[0].group
        if D % g or g % (D // 32):
            raise ValueError("int8 group size must divide head_dim")
        sizes = [len(memoryview(b.payload).cast("B")) for b in blks]
        offs, tot = [], 0
        for s in sizes:
            offs.append(tot)
            tot += (s + cls.ALIGN - 1) // cls.ALIGN * cls.ALIGN
        host = np.zeros(tot, dtype=np.uint8)
        for b, o, s in zip(blks, offs, sizes):
            host[o:o + s] = np.frombuffer(memoryview(b.payload).cast("B"), dtype=np.uint8, count=s)
        base = torch.from_numpy(host).pin_memory().to(device, non_blocking=True)
        torch.cuda.synchronize()
        return cls(base, torch.tensor(offs, dtype=torch.int64, device=device), shape, L, H, Tb, D, g)

    @property
    def nbytes(self) -> int:
        return self.base.numel()

    def decode(self, q, layer: int, tokens: int | None = None, tps: int | None = None, scale: float | None = None,
               variant: int | None = None):
        """variant 1: one warp per token; 2: sub-warp per token, 8-byte loads; 3: 2 + GQA head packing;
        4: GQA on tensor cores (WMMA, shared-memory tiles); 5: GQA on tensor cores, register-resident
        (mma.sync PTX, FlashAttention-2 style). None picks the fastest eligible kernel (choose_kernel).
        tps (tokens per split) defaults to the measured best for that kernel."""
        T = self.tokens if tokens is None else tokens
        v_auto, tps_auto = choose_kernel(q.shape[0], self.hkv, T, self.d, self.g, self.tb, variant)
        variant = v_auto if variant is None else variant
        tps = tps_auto if tps is None else tps
        scale = 1.0 / math.sqrt(self.d) if scale is None else scale
        return ext().paged_int8_decode(q.contiguous(), self.base, self.offs, self.groups, layer,
                                       self.hkv, self.tb, T, self.g, tps, scale, variant)


_SMS = None


def auto_tps(T: int, heads_in_grid: int, blocks_per_sm: int = 8, cap: int = 1024) -> int:
    """Tokens per split so that heads x splits >= blocks_per_sm x SMs (multiple of 64, 64..cap)."""
    global _SMS
    if _SMS is None:
        import torch
        _SMS = torch.cuda.get_device_properties(0).multi_processor_count
    want_splits = max(1, (blocks_per_sm * _SMS + heads_in_grid - 1) // heads_in_grid)
    tps = -(-T // want_splits)
    return int(min(cap, max(64, -(-tps // 64) * 64)))


def v5_eligible(hq: int, hkv: int, d: int, g: int, tb: int) -> bool:
    r = hq // hkv
    return 1 < r <= 16 and d in (64, 128) and (g == d or (d == 128 and g == 64)) and tb % 16 == 0


def choose_kernel(hq: int, hkv: int, T: int, d: int, g: int, tb: int, variant: int | None = None):
    """(variant, tokens per split). GQA -> v5 (register-resident tensor cores) when eligible; it is
    fastest at every measured length (results/cuda/attention.md). v5 does more work per warp, so it
    wants fewer, longer splits: ~4 CUDA blocks per SM, splits up to 8192 tokens. MHA -> v2."""
    if variant is None:
        if hq > hkv and v5_eligible(hq, hkv, d, g, tb):
            variant = 5
        elif hq > hkv and T >= 8192:
            variant = 3
        else:
            variant = 2
    if variant >= 5 and hq > hkv:
        return variant, auto_tps(T, hkv, blocks_per_sm=4, cap=8192)
    return variant, auto_tps(T, hkv if (variant >= 3 and hq > hkv) else hq)


def dense_fp16_decode(q, K, V, tps: int | None = None, scale: float | None = None):
    """Same split-K kernel on a dense fp16 cache K, V: [Hkv, T, D]."""
    scale = 1.0 / math.sqrt(q.shape[-1]) if scale is None else scale
    tps = auto_tps(K.shape[1], q.shape[0]) if tps is None else tps
    return ext().dense_fp16_decode(q.contiguous(), K.contiguous(), V.contiguous(), tps, scale)


def reference(q, K, V, scale: float | None = None):
    """fp32 reference: softmax(q K^T * scale) V with GQA head mapping. K, V: [Hkv, T, D]."""
    import torch
    Hq, D = q.shape
    Hkv = K.shape[0]
    scale = 1.0 / math.sqrt(D) if scale is None else scale
    Kr = K.float().repeat_interleave(Hq // Hkv, 0)
    Vr = V.float().repeat_interleave(Hq // Hkv, 0)
    s = torch.einsum("hd,htd->ht", q.float(), Kr) * scale
    return torch.einsum("ht,htd->hd", s.softmax(-1), Vr)
