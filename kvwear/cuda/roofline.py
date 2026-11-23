"""Empirical roofline + static resource/occupancy report for every kvwear CUDA kernel.

    python -m kvwear.cuda.roofline --out results/cuda

1. Ceilings, measured on this GPU by kvprof_kernels.cu (not taken from a spec sheet):
     FP32 FMA throughput (16 independent FMA chains per thread), DRAM read / copy / write bandwidth
     (128-bit grid-stride kernels, 256 MiB buffers, best of several grid sizes).
2. Every kvwear kernel is timed (CUDA events, median) on its production shape and placed on the
   roofline: x = arithmetic intensity (FLOP per compulsory DRAM byte), y = achieved GFLOP/s.
   Bytes are the compulsory traffic (each input byte read once, each output byte written once),
   so "% of bandwidth ceiling" is a lower bound on how well the kernel uses DRAM.
   FLOP counting convention (stated so the x-axis is reproducible):
     quantize int8  3 per value (abs-max, divide, round)      int4  4 (min/max, subtract, divide, round)
     dequant  int8  1 per value (multiply)                    int4  2 (multiply, add)
     attention      4*D per (query head, token) for q.k and p.v (multiply+add each) + 4 for the softmax
3. cudaFuncGetAttributes + cudaOccupancyMaxActiveBlocksPerMultiprocessor give registers, spills,
   shared memory and theoretical occupancy; the limiter (threads, blocks, registers or shared
   memory) is derived with the same rules as NVIDIA's occupancy calculator.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import ext
from .bench import cuda_time, kv_like


# --------------------------------------------------------------------------- ceilings

def measure_ceilings(iters: int = 20) -> dict:
    e = ext()
    p = dict(e.device_props(0))
    sms = p["sms"]
    # FP32: enough blocks for full occupancy on every SM, several waves
    blocks, fma_iters = sms * 6 * 8, 8192
    t = cuda_time(lambda: e.peak_fp32(blocks, fma_iters), iters)
    fp32 = 2 * e.peak_fp32_chains() * fma_iters * blocks * 256 / t / 1e9
    nbytes = 256 << 20
    src = torch.randint(0, 255, (nbytes,), dtype=torch.uint8, device="cuda")
    dst = torch.empty_like(src)
    bw = {}
    for mode, name, traffic in ((0, "read", nbytes), (1, "copy", 2 * nbytes), (2, "write", nbytes)):
        best = 0.0
        for grid in (sms * 4, sms * 8, sms * 16, sms * 32, sms * 64):
            tt = cuda_time(lambda: e.stream_bench(mode, src, dst, grid), iters)
            best = max(best, traffic / tt / 1e9)
        bw[name] = best
    tm = cuda_time(lambda: dst.copy_(src), iters)
    bw["torch_copy"] = 2 * nbytes / tm / 1e9
    del src, dst
    torch.cuda.empty_cache()
    mem = dict(e.device_memory_info(0))
    return {"device": torch.cuda.get_device_name(0), "props": p,
            "fp32_gflops_measured": fp32,
            "fp32_gflops_spec_base_clock": sms * 128 * 2 * p["clock_khz"] / 1e6,   # 128 FP32 lanes/SM on sm_86
            "dram_gbps_spec": mem["peak_gbps"], "dram_gbps_measured": bw}


# --------------------------------------------------------------------------- kernel points

def _pt(name, family, t, flops, nbytes, note=""):
    return {"kernel": name, "family": family, "time_us": t * 1e6, "flops": flops, "bytes": nbytes,
            "ai": flops / nbytes, "gflops": flops / t / 1e9, "gbps": nbytes / t / 1e9, "note": note}


def codec_points(iters: int) -> list[dict]:
    e = ext()
    shape, g = (24, 2, 32, 64, 64), 64                       # one OPT-1.3B block: 64 tokens, 12 MiB
    x = torch.from_numpy(kv_like(shape)).cuda().reshape(-1)
    n, G = x.numel(), x.numel() // g
    pts = []
    names = ["v0 naive", "v1 warp+shuffle", "v2 half2", "v3 register cache", "v4 sub-warp 128-bit"]
    for v, nm in enumerate(names):
        t = cuda_time(lambda: e.quant_int8(x, g, v), iters)
        pts.append(_pt(f"quantize int8 {nm}", "quantize int8", t, 3 * n, 2 * n + n + 2 * G))
        t = cuda_time(lambda: e.quant_int4(x, g, v), iters)
        pts.append(_pt(f"quantize int4 {nm}", "quantize int4", t, 4 * n, 2 * n + n // 2 + 4 * G))
    p8, p4 = e.quant_int8(x, g, 4), e.quant_int4(x, g, 4)
    t = cuda_time(lambda: e.dequant(p8, n, g, 1), iters)
    pts.append(_pt("dequantize int8 (v8)", "dequantize", t, n, n + 2 * G + 2 * n))
    t = cuda_time(lambda: e.dequant(p4, n, g, 2), iters)
    pts.append(_pt("dequantize int4 (v8)", "dequantize", t, 2 * n, n // 2 + 4 * G + 2 * n))
    # batched loader kernel: 16 int8 blocks scattered into one [R, 1024, D] cache
    nb, L, two, H, Tb, D = 16, *shape
    R = L * two * H
    sz = (p8.numel() + 15) // 16 * 16
    base = torch.zeros(nb * sz, dtype=torch.uint8, device="cuda")
    for b in range(nb):
        base[b * sz:b * sz + p8.numel()] = p8
    offs = torch.arange(0, nb * sz, sz, dtype=torch.int64, device="cuda")
    toffs = torch.arange(0, nb * Tb, Tb, dtype=torch.int32, device="cuda")
    meth = torch.ones(nb, dtype=torch.int32, device="cuda")
    dest = torch.empty((R, nb * Tb, D), dtype=torch.float16, device="cuda")
    t = cuda_time(lambda: e.dequant_scatter(base, offs, toffs, meth, dest, Tb, g), iters)
    pts.append(_pt("dequant+scatter int8 x16 blocks", "dequantize", t, nb * n, nb * (n + 2 * G + 2 * n)))
    # reference: PyTorch's own int8 quantize expressed as tensor ops (several kernels, extra traffic)
    xs = x.view(-1, g)

    def torch_q8():
        s = (xs.abs().amax(1, keepdim=True).float() / 127).clamp_min(1e-30)
        return torch.round(xs.float() / s).clamp(-127, 127).to(torch.int8)
    t = cuda_time(torch_q8, iters)
    pts.append(_pt("quantize int8 as PyTorch ops", "PyTorch reference", t, 3 * n, 2 * n + n + 2 * G))
    return pts


def attention_points(iters: int, tokens: int = 32768) -> list[dict]:
    from ..codecs import Codec
    from .attention import PagedInt8KV, dense_fp16_decode
    pts = []
    codec = Codec("int8", device="cuda")
    for mname, (hq, hkv, d) in {"OPT-1.3B": (32, 32, 64), "Qwen2.5-0.5B": (14, 2, 64),
                                "Llama-3-8B shape": (32, 8, 128)}.items():
        rng = np.random.default_rng(0)
        kv = rng.standard_normal((1, 2, hkv, tokens, d), dtype=np.float32).astype(np.float16)
        datas = [codec.encode(np.ascontiguousarray(kv[:, :, :, t:t + 64])) for t in range(0, tokens, 64)]
        paged = PagedInt8KV.from_payloads(datas)
        K = torch.from_numpy(kv[0, 0]).cuda()
        V = torch.from_numpy(kv[0, 1]).cuda()
        q = torch.randn(hq, d, dtype=torch.float16, device="cuda")
        flops = hq * tokens * (4 * d + 4)
        int8_bytes = paged.nbytes
        fp16_bytes = 2 * K.numel() * 2
        variants = [(1, "v1 warp/token"), (2, "v2 sub-warp/token")]
        if hq > hkv:
            variants += [(3, "v3 GQA head packing"), (4, "v4 tensor cores (WMMA)"), (5, "v5 tensor cores (mma.sync)")]
        for v, nm in variants:
            t = min(cuda_time(lambda: paged.decode(q, 0, variant=v), iters) for _ in range(3))
            pts.append(_pt(f"{mname} attn int8 {nm}", "attention int8", t, flops, int8_bytes, f"T={tokens}"))
        t = min(cuda_time(lambda: dense_fp16_decode(q, K, V), iters) for _ in range(3))
        pts.append(_pt(f"{mname} attn fp16 (kvwear kernel)", "attention fp16", t, flops, fp16_bytes, f"T={tokens}"))
        qs, Ks, Vs = q.view(1, hq, 1, d), K.unsqueeze(0), V.unsqueeze(0)
        if hq != hkv:
            Ks, Vs = Ks.repeat_interleave(hq // hkv, 1), Vs.repeat_interleave(hq // hkv, 1)
        t = min(cuda_time(lambda: F.scaled_dot_product_attention(qs, Ks, Vs), iters) for _ in range(3))
        pts.append(_pt(f"{mname} attn torch SDPA fp16", "PyTorch reference", t, flops, 2 * Ks.numel() * 2,
                       f"T={tokens}; reads the GQA-expanded K/V"))
        del paged, K, V, Ks, Vs
        torch.cuda.empty_cache()
    return pts


def gemm_points(iters: int) -> list[dict]:
    """Compute-bound references: cuBLAS FP32 (no TF32) and FP16 (tensor cores) GEMM."""
    pts = []
    n = 4096
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    for dt, nm in ((torch.float32, "cuBLAS SGEMM 4096^3 (FP32 cores)"), (torch.float16, "cuBLAS HGEMM 4096^3 (tensor cores)")):
        a = torch.randn(n, n, dtype=dt, device="cuda")
        b = torch.randn(n, n, dtype=dt, device="cuda")
        t = cuda_time(lambda: a @ b, max(5, iters // 4))
        pts.append(_pt(nm, "compute-bound reference", t, 2 * n ** 3, 3 * n * n * a.element_size()))
    torch.backends.cuda.matmul.allow_tf32 = prev
    return pts


# --------------------------------------------------------------------------- static report

def occupancy_rows(props: dict) -> list[dict]:
    rows = []
    for a in ext().kernel_attributes():
        a = dict(a)
        b = a["block"]
        wpb = (b + 31) // 32
        lim_threads = props["max_threads_per_sm"] // b
        lim_blocks = props["max_blocks_per_sm"]
        per_warp = math.ceil(max(a["regs"], 1) * 32 / 256) * 256            # register allocation unit: 256/warp
        lim_regs = (props["regs_per_sm"] // per_warp) // wpb
        smem = a["static_smem"] + props["reserved_smem_per_block"]
        lim_smem = props["smem_per_sm"] // smem if smem else 10 ** 9
        lims = {"threads": lim_threads, "blocks": lim_blocks, "registers": lim_regs, "shared mem": lim_smem}
        limiter = min(lims, key=lims.get)
        bps = a["blocks_per_sm"]
        a.update({"warps_per_sm": bps * wpb, "occupancy": bps * b / props["max_threads_per_sm"],
                  "limiter": limiter, "calc_blocks_per_sm": min(lims.values())})
        rows.append(a)
    return rows


# --------------------------------------------------------------------------- output

def plot(res: dict, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    c = res["ceilings"]
    bw = c["dram_gbps_measured"]["read"]
    fp = c["fp32_gflops_measured"]
    pts = res["points"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15, 6.2), gridspec_kw={"width_ratios": [1.25, 1]})
    xs = np.logspace(-2, 3.5, 300)
    ax.plot(xs, np.minimum(bw * xs, fp), color="#222", lw=2, label=f"measured roof: {bw:.0f} GB/s, {fp / 1e3:.2f} TFLOP/s FP32")
    ax.plot(xs, np.minimum(c["dram_gbps_spec"] * xs, c["fp32_gflops_spec_base_clock"]), color="#888", lw=1, ls="--",
            label=f"spec: {c['dram_gbps_spec']:.0f} GB/s, {c['fp32_gflops_spec_base_clock'] / 1e3:.2f} TFLOP/s @ base clock")
    import re as _re
    # optimization ladders: same work (same x), each optimization stage one point higher
    ladders = [  # label, regex, colour, (label anchor: 'top' or 'bottom', dx, dy, ha)
        ("quantize int8", r"^quantize int8 v(\d)", "#1f5f99", ("bottom", -8, -4, "right")),
        ("quantize int4", r"^quantize int4 v(\d)", "#5fa2dd", ("bottom", 8, -4, "left")),
        ("OPT-1.3B decode attn (MHA)", r"^OPT-1.3B attn int8 v(\d)", "#e08a1e", ("top", -8, 2, "right")),
        ("Llama-3-8B-shape attn (GQA 4:1)", r"^Llama-3-8B shape attn int8 v(\d)", "#7d3c98", ("top", -8, 4, "right")),
        ("Qwen2.5-0.5B attn (GQA 7:1)", r"^Qwen2.5-0.5B attn int8 v(\d)", "#c0392b", ("top", 8, 6, "left")),
    ]
    in_ladder = set()
    for label, rx, col, (anchor, dx, dy, ha) in ladders:
        lad = sorted(((int(_re.match(rx, q["kernel"]).group(1)), q) for q in pts if _re.match(rx, q["kernel"])),
                     key=lambda x: x[0])
        if not lad:
            continue
        in_ladder.update(q["kernel"] for _, q in lad)
        ax.plot([q["ai"] for _, q in lad], [q["gflops"] for _, q in lad], "-o", color=col, lw=1.3, ms=5,
                zorder=3, mec="white", mew=0.6, label=label)
        for v, q in lad:
            ax.annotate(f"v{v}", (q["ai"], q["gflops"]), fontsize=6.5, color=col,
                        xytext=(5 if ha != "right" else -5, -2), ha="left" if ha != "right" else "right",
                        textcoords="offset points")
        q = lad[-1][1] if anchor == "top" else lad[0][1]
        ax.annotate(label, (q["ai"], q["gflops"]), fontsize=7, color=col, fontweight="bold",
                    xytext=(dx, dy + (6 if anchor == "top" else -10)), textcoords="offset points", ha=ha)
    other = {"dequantize": ("#2e9e6a", "dequantize / scatter (int8, int4)"),
             "attention fp16": ("#c9a227", "kvwear fp16 attention kernel"),
             "PyTorch reference": ("#8e8e8e", "PyTorch reference (SDPA fp16, quantize as tensor ops)"),
             "compute-bound reference": ("#333333", "cuBLAS GEMM 4096^3 (compute-bound reference)")}
    for fam, (col, lab) in other.items():
        pp = [q for q in pts if q["family"] == fam and q["kernel"] not in in_ladder]
        if pp:
            ax.scatter([q["ai"] for q in pp], [q["gflops"] for q in pp], color=col, s=30, zorder=3, label=lab,
                       marker="s" if fam != "compute-bound reference" else "D", edgecolor="white", linewidth=0.5)
    for q in pts:
        if q["family"] == "compute-bound reference":
            ax.annotate(q["kernel"].split(" 4096")[0].replace("cuBLAS ", ""), (q["ai"], q["gflops"]), fontsize=6.5,
                        color="#333", xytext=(-4, 6), textcoords="offset points", ha="right")
    fam_col = {"quantize int8": "#1f5f99", "quantize int4": "#5fa2dd", "dequantize": "#2e9e6a",
               "attention int8": "#c0392b", "attention fp16": "#c9a227", "PyTorch reference": "#8e8e8e",
               "compute-bound reference": "#333333"}
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.1, 3000)
    ax.set_ylim(1, 3e5)
    ax.set_xlabel("arithmetic intensity (FLOP / compulsory DRAM byte)")
    ax.set_ylabel("achieved GFLOP/s")
    ax.set_title(f"Roofline, {c['device']}", fontsize=10)
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=6.5, loc="upper left", ncol=1, framealpha=0.9)

    mem = [p for p in pts if p["family"] != "compute-bound reference"]
    mem.sort(key=lambda p: p["gbps"])
    ax2.barh(range(len(mem)), [100 * p["gbps"] / bw for p in mem],
             color=[fam_col.get(p["family"], "#999") for p in mem])
    ax2.set_yticks(range(len(mem)), [p["kernel"] for p in mem], fontsize=6.5)
    ax2.axvline(100, color="#222", lw=1)
    ax2.set_xlabel(f"% of measured DRAM read ceiling ({bw:.0f} GB/s), compulsory bytes")
    ax2.set_title("Memory-bound kernels: bandwidth efficiency", fontsize=10)
    ax2.grid(alpha=0.25, axis="x")
    fig.tight_layout()
    fig.savefig(out / "roofline.png", dpi=150)

    occ = res["occupancy"]
    fig, ax = plt.subplots(figsize=(9, 0.26 * len(occ) + 1.2))
    lc = {"threads": "#2e9e6a", "blocks": "#7fb3e0", "registers": "#c0392b", "shared mem": "#e08a1e"}
    ax.barh(range(len(occ)), [100 * r["occupancy"] for r in occ], color=[lc[r["limiter"]] for r in occ])
    ax.set_yticks(range(len(occ)), [f"{r['name']}  ({r['regs']} regs)" for r in occ], fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xlabel("theoretical occupancy (% of max warps per SM)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=v, label=f"limited by {k}") for k, v in lc.items()], fontsize=7, loc="lower right")
    ax.grid(alpha=0.25, axis="x")
    fig.tight_layout()
    fig.savefig(out / "occupancy.png", dpi=150)


def write_md(res: dict, out: Path):
    c = res["ceilings"]
    p = c["props"]
    bw = c["dram_gbps_measured"]
    lines = [f"# Roofline and occupancy ({c['device']}, sm_{p['cc_major']}{p['cc_minor']}, {p['sms']} SMs)\n",
             "## Ceilings (measured by kvprof_kernels.cu)\n",
             "| ceiling | measured | spec / theoretical |", "|---|---|---|",
             f"| FP32 FMA throughput | {c['fp32_gflops_measured']:.0f} GFLOP/s | "
             f"{c['fp32_gflops_spec_base_clock']:.0f} GFLOP/s at the {p['clock_khz'] / 1e3:.0f} MHz base clock |",
             f"| DRAM read | {bw['read']:.1f} GB/s | {c['dram_gbps_spec']:.1f} GB/s |",
             f"| DRAM copy (read+write) | {bw['copy']:.1f} GB/s | |",
             f"| DRAM write | {bw['write']:.1f} GB/s | |",
             f"| torch tensor copy | {bw['torch_copy']:.1f} GB/s | |",
             f"\nRidge point (measured): {c['fp32_gflops_measured'] / bw['read']:.1f} FLOP/byte. "
             "Every kvwear kernel sits far left of it: they are all memory-bound, so the right metric "
             "is bandwidth, not FLOP/s.\n",
             "## Kernels on the roofline\n",
             "Bytes are compulsory DRAM traffic; % is of the measured DRAM read ceiling.\n",
             "| kernel | time (us) | FLOP/byte | GFLOP/s | GB/s | % of BW ceiling |", "|---|---|---|---|---|---|"]
    for q in res["points"]:
        lines.append(f"| {q['kernel']} | {q['time_us']:.1f} | {q['ai']:.2f} | {q['gflops']:.0f} | {q['gbps']:.1f} | "
                     f"{100 * q['gbps'] / bw['read']:.0f}% |")
    lines += ["\n## Static resources and theoretical occupancy (cudaFuncGetAttributes, cudaOccupancy API)\n",
              f"Per SM: {p['max_threads_per_sm']} threads, {p['max_blocks_per_sm']} blocks, {p['regs_per_sm']} registers, "
              f"{p['smem_per_sm'] // 1024} KiB shared memory ({p['reserved_smem_per_block']} B reserved per block).\n",
              "| kernel | block | regs/thread | spill+stack B | static smem B | blocks/SM | warps/SM | occupancy | limiter |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in res["occupancy"]:
        lines.append(f"| {r['name']} | {r['block']} | {r['regs']} | {r['local_bytes']} | {r['static_smem']} | "
                     f"{r['blocks_per_sm']} | {r['warps_per_sm']} | {100 * r['occupancy']:.0f}% | {r['limiter']} |")
    (out / "roofline.md").write_text("\n".join(lines) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results/cuda")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--tokens", type=int, default=32768)
    ap.add_argument("--replot", action="store_true", help="redraw the figures from roofline.json")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.replot:
        res = json.loads((out / "roofline.json").read_text())
        write_md(res, out)
        plot(res, out)
        return
    for _ in range(3):                                   # warm-up: JIT build + GPU clocks up
        measure_ceilings(3)
    res = {"ceilings": measure_ceilings(args.iters)}
    print(json.dumps({k: v for k, v in res["ceilings"].items() if k != "props"}, indent=1), flush=True)
    res["points"] = codec_points(args.iters) + attention_points(args.iters, args.tokens) + gemm_points(args.iters)
    res["occupancy"] = occupancy_rows(res["ceilings"]["props"])
    (out / "roofline.json").write_text(json.dumps(res, indent=2))
    write_md(res, out)
    plot(res, out)
    print((out / "roofline.md").read_text())


if __name__ == "__main__":
    main()
