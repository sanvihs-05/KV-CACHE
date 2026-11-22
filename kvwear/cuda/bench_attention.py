"""Decode-step attention benchmark: int8 paged KV (fused) vs fp16.

    python -m kvwear.cuda.bench_attention --out results/cuda

For each context length, one decode step of one layer (a query of one token per head) with:
  * torch SDPA (fp16, cuDNN/flash/mem-efficient backend chosen by PyTorch)
  * kvwear dense fp16 split-K kernel (like-for-like baseline for the int8 kernel)
  * dequantize int8 -> fp16 with the BlockLoader kernel, then SDPA (the unfused way)
  * kvwear paged int8 kernel (reads the on-disk block format directly)
Model shapes: Qwen2.5-0.5B (14 q heads, 2 kv heads, D=64) and OPT-1.3B (32/32, D=64).
Also reports the accuracy of the int8 result vs an fp32 reference on the original fp16 KV.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..codecs import Codec
from .attention import PagedInt8KV, dense_fp16_decode, reference
from .bench import cuda_time

MODELS = {"qwen2.5-0.5b (Hq 14, Hkv 2)": (14, 2, 64), "opt-1.3b (Hq 32, Hkv 32)": (32, 32, 64)}
TB = 64


def make_blocks(hkv, d, tokens, seed=0):
    """One layer of KV as kvwear int8 blocks (+ the original fp16 tensors)."""
    rng = np.random.default_rng(seed)
    kv = (rng.standard_normal((1, 2, hkv, tokens, d), dtype=np.float32)).astype(np.float16)
    codec = Codec("int8")
    datas = [bytearray(codec.encode(np.ascontiguousarray(kv[:, :, :, t:t + TB]))) for t in range(0, tokens, TB)]
    return kv, datas


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results/cuda")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--lengths", default="1024,2048,4096,8192,16384,32768")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for mname, (hq, hkv, d) in MODELS.items():
        for T in map(int, args.lengths.split(",")):
            if hkv * T * d * 2 * 2 > 1.2e9:
                continue
            kv, datas = make_blocks(hkv, d, T)
            K = torch.from_numpy(kv[0, 0]).cuda().contiguous()
            V = torch.from_numpy(kv[0, 1]).cuda().contiguous()
            q = torch.randn(hq, d, dtype=torch.float16, device="cuda")
            paged = PagedInt8KV.from_payloads(datas)
            ref = reference(q, K, V)
            # SDPA with GQA expects [B, H, T, D]
            qs, Ks, Vs = q.view(1, hq, 1, d), K.unsqueeze(0), V.unsqueeze(0)
            if hkv != hq:
                Ks, Vs = Ks.repeat_interleave(hq // hkv, 1), Vs.repeat_interleave(hq // hkv, 1)

            def sdpa():
                return F.scaled_dot_product_attention(qs, Ks, Vs)

            def dense():
                return dense_fp16_decode(q, K, V)

            from . import ext as _ext
            nb = paged.offs.numel()
            toffs = torch.arange(0, T, TB, dtype=torch.int32, device="cuda")
            meth = torch.ones(nb, dtype=torch.int32, device="cuda")
            full = torch.empty((2 * hkv, T, d), dtype=torch.float16, device="cuda")

            def unfused():
                # GPU-resident int8 -> fp16 (dequantize+scatter kernel), then SDPA: the unfused way
                _ext().dequant_scatter(paged.base, paged.offs, toffs, meth, full, TB, paged.g)
                k2, v2 = full[:hkv].unsqueeze(0), full[hkv:].unsqueeze(0)
                if hkv != hq:
                    k2, v2 = k2.repeat_interleave(hq // hkv, 1), v2.repeat_interleave(hq // hkv, 1)
                return F.scaled_dot_product_attention(qs, k2, v2)

            def fused_v1():
                return paged.decode(q, 0, variant=1)

            def fused_v2():
                return paged.decode(q, 0, variant=2)

            def fused():
                return paged.decode(q, 0, variant=3)

            res = {}
            for name, fn in (("torch SDPA fp16", sdpa), ("kvwear dense fp16 kernel", dense),
                             ("dequantize + SDPA (unfused int8)", unfused),
                             ("kvwear paged int8 v1 (warp/token)", fused_v1),
                             ("kvwear paged int8 v2 (sub-warp/token)", fused_v2),
                             ("kvwear paged int8 v3 (+ GQA head packing)", fused)):
                t = cuda_time(fn, args.iters)
                o = fn().reshape(hq, d).float()
                err = ((o - ref).norm() / ref.norm()).item()
                res[name] = t
                kv_bytes = paged.nbytes if "int8" in name else 2 * K.numel() * 2
                rows.append({"model": mname, "tokens": T, "impl": name, "time_us": t * 1e6,
                             "kv_mb_read": kv_bytes / 2**20, "gbps": kv_bytes / t / 1e9, "rel_err_vs_fp32": err})
            print(f"{mname:28s} T={T:6d}  " + "  ".join(f"{k.split(' (')[0][:18]}={v * 1e6:7.1f}us" for k, v in res.items()),
                  flush=True)
            del K, V, paged, Ks, Vs
            torch.cuda.empty_cache()
    (out / "attention.json").write_text(json.dumps(rows, indent=2))
    lines = ["# Decode attention over int8 KV (one layer, one decode step)\n",
             "| model | tokens | impl | time (us) | KV MB read | GB/s | rel. error vs fp32 |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['model']} | {r['tokens']} | {r['impl']} | {r['time_us']:.1f} | {r['kv_mb_read']:.1f} | "
                     f"{r['gbps']:.0f} | {r['rel_err_vs_fp32']:.2e} |")
    (out / "attention.md").write_text("\n".join(lines) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(MODELS), figsize=(12, 4))
    colors = {"torch SDPA fp16": "#9aa5b1", "kvwear dense fp16 kernel": "#c9a227",
              "dequantize + SDPA (unfused int8)": "#b06030", "kvwear paged int8 v1 (warp/token)": "#7fb3e0",
              "kvwear paged int8 v2 (sub-warp/token)": "#3a7fc4", "kvwear paged int8 v3 (+ GQA head packing)": "#0b3d75"}
    for ax, mname in zip(np.atleast_1d(axes), MODELS):
        for impl, c in colors.items():
            rr = [r for r in rows if r["model"] == mname and r["impl"] == impl]
            if rr:
                ax.plot([r["tokens"] for r in rr], [r["time_us"] for r in rr], "o-", color=c, label=impl)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("context tokens")
        ax.set_ylabel("µs per layer per decode step")
        ax.set_title(mname)
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "attention_decode.png", dpi=140)
    print((out / "attention.md").read_text())


if __name__ == "__main__":
    main()
