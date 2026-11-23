"""Launch every kvwear kernel once, for Nsight Compute (run through linux/profile_ncu.sh).

    ncu --profile-from-start off ... python -m kvwear.cuda.ncu_targets --out results/cuda/ncu

All inputs are built and every kernel is warmed up first; profiling is switched on with
cudaProfilerStart() only around the measured launches, so ncu sees exactly one launch per
target, in the order written to targets.json (ncu's launch IDs map onto that list).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import ext
from .bench import kv_like


def build_targets():
    e = ext()
    T = []                                            # (name, fn)
    x = torch.from_numpy(kv_like((24, 2, 32, 64, 64))).cuda().reshape(-1)   # one OPT-1.3B block (12 MiB)
    g, n = 64, x.numel()
    for v, nm in enumerate(["v0 naive", "v1 warp", "v2 half2", "v3 register cache", "v4 sub-warp 128-bit"]):
        T.append((f"quantize int8 {nm}", lambda v=v: e.quant_int8(x, g, v)))
    T.append(("quantize int4 v4 sub-warp 128-bit", lambda: e.quant_int4(x, g, 4)))
    p8 = e.quant_int8(x, g, 4)
    T.append(("dequantize int8 v8", lambda: e.dequant(p8, n, g, 1)))
    nb, Tb = 16, 64
    sz = (p8.numel() + 15) // 16 * 16
    base = torch.zeros(nb * sz, dtype=torch.uint8, device="cuda")
    for b in range(nb):
        base[b * sz:b * sz + p8.numel()] = p8
    offs = torch.arange(0, nb * sz, sz, dtype=torch.int64, device="cuda")
    toffs = torch.arange(0, nb * Tb, Tb, dtype=torch.int32, device="cuda")
    meth = torch.ones(nb, dtype=torch.int32, device="cuda")
    dest = torch.empty((24 * 2 * 32, nb * Tb, 64), dtype=torch.float16, device="cuda")
    T.append(("dequant+scatter int8 x16 blocks", lambda: e.dequant_scatter(base, offs, toffs, meth, dest, Tb, g)))

    from ..codecs import Codec
    from .attention import PagedInt8KV, dense_fp16_decode
    codec = Codec("int8", device="cuda")
    keep = []
    for mname, (hq, hkv, d), variants in (("OPT-1.3B", (32, 32, 64), (2,)),
                                          ("Qwen2.5-0.5B", (14, 2, 64), (2, 3, 4, 5)),
                                          ("Llama-3-8B shape", (32, 8, 128), (3, 4, 5))):
        tokens = 32768
        rng = np.random.default_rng(0)
        kv = rng.standard_normal((1, 2, hkv, tokens, d), dtype=np.float32).astype(np.float16)
        paged = PagedInt8KV.from_payloads([codec.encode(np.ascontiguousarray(kv[:, :, :, t:t + 64]))
                                           for t in range(0, tokens, 64)])
        q = torch.randn(hq, d, dtype=torch.float16, device="cuda")
        K, V = torch.from_numpy(kv[0, 0]).cuda(), torch.from_numpy(kv[0, 1]).cuda()
        keep += [paged, q, K, V]
        for v in variants:
            T.append((f"{mname} attention int8 v{v} (T=32K)", lambda p=paged, q=q, v=v: p.decode(q, 0, variant=v)))
        T.append((f"{mname} attention fp16 kernel (T=32K)", lambda q=q, K=K, V=V: dense_fp16_decode(q, K, V)))
    src = torch.randint(0, 255, (256 << 20,), dtype=torch.uint8, device="cuda")
    dst = torch.empty_like(src)
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    T.append(("ceiling: stream_read", lambda: e.stream_bench(0, src, dst, sms * 16)))
    T.append(("ceiling: stream_copy", lambda: e.stream_bench(1, src, dst, sms * 16)))
    T.append(("ceiling: peak_fp32_fma", lambda: e.peak_fp32(sms * 48, 4096)))
    return T, keep


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results/cuda/ncu")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    targets, _keep = build_targets()
    for _ in range(3):
        for _, fn in targets:
            fn()
    torch.cuda.synchronize()
    order = []
    torch.cuda.cudart().cudaProfilerStart()
    for name, fn in targets:
        torch.cuda.nvtx.range_push(name)
        fn()
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
        order.append(name)
    torch.cuda.cudart().cudaProfilerStop()
    (out / "targets.json").write_text(json.dumps(order, indent=1))
    print(f"launched {len(order)} targets")


if __name__ == "__main__":
    main()
