"""Microbenchmarks for the CUDA KV codec.

    python -m kvwear.cuda.bench --out results/cuda

Measures, on real KV-block shapes:
  1. kernel time of each quantize optimization stage (v0 naive .. v3 register-cached) and the
     dequantize kernel, with achieved DRAM bandwidth vs the GPU's theoretical peak and vs a
     device-to-device memcpy (the practical ceiling);
  2. end-to-end encode/decode of one block, host -> host, CPU (numpy, torch) vs CUDA
     (pinned staging, own stream);
  3. PCIe host<->device bandwidth, pageable vs pinned;
  4. whole-request cache assembly (the TTFT-critical load path): CPU decode + concatenate +
     H2D vs the CUDA BlockLoader (one H2D + one batched dequant/scatter launch).
Kernel times use CUDA events; every number is the median of `--iters` runs after warm-up.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from . import BlockLoader, GpuCodec, ext
from ..codecs import Codec, EncodedBlock

SHAPES = {
    "opt-1.3b block (64 tok, 12 MiB)": (24, 2, 32, 64, 64),
    "qwen2.5-0.5b block (64 tok, 768 KiB)": (24, 2, 2, 64, 64),
    "opt-1.3b x16 blocks (1024 tok, 192 MiB)": (24, 2, 32, 1024, 64),
}
VARIANTS = ["v0 naive (thread/group)", "v1 warp/group + shuffle", "v2 + half2 vector I/O", "v3 + register cache (1 pass)",
            "v4 sub-warp groups + 128-bit I/O"]


def cuda_time(fn, iters: int, warmup: int = 3) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b) / 1e3)
    return statistics.median(ts)


def wall_time(fn, iters: int, warmup: int = 6) -> float:
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def kv_like(shape, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(shape, dtype=np.float32) * 1.5).astype(np.float16)


def payload_bytes(method: str, n: int, g: int) -> int:
    groups = n // g
    return 2 * groups + n if method == "int8" else 4 * groups + n // 2


def bench_kernels(iters: int) -> list[dict]:
    e = ext()
    rows = []
    for sname, shape in SHAPES.items():
        x = torch.from_numpy(kv_like(shape)).cuda()
        n = x.numel()
        g = shape[-1]
        # practical ceiling: a device-to-device copy of the same fp16 data (read n*2 + write n*2)
        y = torch.empty_like(x)
        t_copy = cuda_time(lambda: y.copy_(x), iters)
        rows.append({"shape": sname, "op": "memcpy D2D (ceiling)", "method": "-", "time_ms": t_copy * 1e3,
                     "bytes": 4 * n, "gbps": 4 * n / t_copy / 1e9})
        for method in ("int8", "int4"):
            fn = e.quant_int8 if method == "int8" else e.quant_int4
            moved = 2 * n + payload_bytes(method, n, g)            # read fp16 + write payload
            for v, vname in enumerate(VARIANTS):
                t = cuda_time(lambda: fn(x.view(-1), g, v), iters)
                rows.append({"shape": sname, "op": f"quantize {vname}", "method": method, "variant": v,
                             "time_ms": t * 1e3, "bytes": moved, "gbps": moved / t / 1e9})
            p = fn(x.view(-1), g, 4)
            m = 1 if method == "int8" else 2
            t = cuda_time(lambda: e.dequant(p, n, g, m), iters)
            rows.append({"shape": sname, "op": "dequantize", "method": method, "time_ms": t * 1e3,
                         "bytes": moved, "gbps": moved / t / 1e9})
            # torch-on-GPU reference quantizer (what you'd write without a custom kernel)
            if method == "int8":
                def torch_q8():
                    xf = x.view(-1, g).float()
                    s = xf.abs().amax(1, keepdim=True) / 127.0
                    s[s == 0] = 1.0
                    return s.half(), torch.round(xf / s).clamp_(-127, 127).to(torch.int8)
                t = cuda_time(torch_q8, iters)
            else:
                def torch_q4():
                    xf = x.view(-1, g).float()
                    mn = xf.amin(1, keepdim=True)
                    s = (xf.amax(1, keepdim=True) - mn) / 15.0
                    s[s == 0] = 1.0
                    q = torch.round((xf - mn) / s).clamp_(0, 15).to(torch.uint8).view(-1)
                    return s.half(), mn.half(), q[0::2] | (q[1::2] << 4)
                t = cuda_time(torch_q4, iters)
            rows.append({"shape": sname, "op": "quantize torch ops on GPU (no custom kernel)", "method": method,
                         "time_ms": t * 1e3, "bytes": moved, "gbps": moved / t / 1e9})
        del x, y
        torch.cuda.empty_cache()
    return rows


def bench_end_to_end(iters: int) -> list[dict]:
    rows = []
    for sname, shape in list(SHAPES.items())[:2]:
        arr = kv_like(shape)
        for method in ("int8", "int4"):
            cpu_np, cpu_t, gpu = Codec(method, backend="numpy"), Codec(method, backend="torch"), Codec(method, device="cuda")
            assert gpu.encode(arr) == cpu_t.encode(arr) == cpu_np.encode(arr)
            for name, c in (("CPU numpy", cpu_np), ("CPU torch (multi-thread)", cpu_t), ("CUDA (pinned, own stream)", gpu)):
                t = wall_time(lambda: c.encode(arr), iters)
                rows.append({"shape": sname, "op": "encode host->bytes", "impl": name, "method": method,
                             "time_ms": t * 1e3, "mb_per_s": arr.nbytes / t / 1e6})
            data = bytearray(cpu_t.encode(arr))
            for name, be in (("CPU numpy", "numpy"), ("CPU torch (multi-thread)", "torch")):
                t = wall_time(lambda: Codec.decode(data, backend=be), iters)
                rows.append({"shape": sname, "op": "decode bytes->fp16", "impl": name, "method": method,
                             "time_ms": t * 1e3, "mb_per_s": arr.nbytes / t / 1e6})
    return rows


def bench_pcie(iters: int) -> list[dict]:
    rows = []
    nbytes = 64 << 20
    pageable = torch.empty(nbytes, dtype=torch.uint8)
    pinned = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
    dev = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
    for name, host in (("pageable", pageable), ("pinned", pinned)):
        t = cuda_time(lambda: dev.copy_(host, non_blocking=True), iters)
        rows.append({"op": "H2D", "memory": name, "mb": nbytes >> 20, "time_ms": t * 1e3, "gbps": nbytes / t / 1e9})
        t = cuda_time(lambda: host.copy_(dev, non_blocking=True), iters)
        rows.append({"op": "D2H", "memory": name, "mb": nbytes >> 20, "time_ms": t * 1e3, "gbps": nbytes / t / 1e9})
    return rows


def bench_assembly(iters: int) -> list[dict]:
    """The load path of one request: 16 cached OPT-1.3B blocks (1024 tokens) -> attention cache."""
    rows = []
    shape = (24, 2, 32, 64, 64)
    nblk = 16
    arrs = [kv_like(shape, seed=i) for i in range(nblk)]
    loader = BlockLoader()
    for method in ("fp16", "int8", "int4"):
        datas = [bytearray(Codec(method).encode(a)) for a in arrs]

        def cpu_path():
            blocks = [Codec.decode(d) for d in datas]                   # CPU decode
            t = torch.from_numpy(np.concatenate(blocks, axis=3)).to("cuda")
            for l in range(t.shape[0]):                                 # what build_cache did
                t[l, 0].contiguous(), t[l, 1].contiguous()
            torch.cuda.synchronize()

        def gpu_path():
            out = loader.assemble([EncodedBlock(d) for d in datas])
            torch.cuda.synchronize()
            return out

        ref = np.concatenate([Codec.decode(d) for d in datas], axis=3)
        assert np.array_equal(gpu_path().cpu().numpy(), ref)
        for name, fn in (("CPU decode + concat + H2D", cpu_path), ("CUDA BlockLoader (pipelined H2D + 1 launch)", gpu_path)):
            t = wall_time(fn, iters)
            rows.append({"op": f"assemble {nblk} blocks (1024 tok)", "method": method, "impl": name,
                         "time_ms": t * 1e3, "h2d_mb": sum(len(d) for d in datas) / 2**20 if "CUDA" in name
                         else sum(a.nbytes for a in arrs) / 2**20})
    return rows


def _md(rows, cols) -> str:
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(f"{r.get(c, ''):.3f}" if isinstance(r.get(c), float) else str(r.get(c, ""))
                                    for c in cols) + " |")
    return "\n".join(out)


def plot(res, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    peak = res["device"]["peak_gbps"]
    k = [r for r in res["kernels"] if r["shape"].startswith("opt-1.3b x16")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, method in zip(axes, ("int8", "int4")):
        rr = [r for r in k if r["method"] in (method, "-")]
        labels = [r["op"].replace("quantize ", "Q ").replace(" (no custom kernel)", "") for r in rr]
        vals = [r["gbps"] for r in rr]
        colors = ["#9aa5b1" if "memcpy" in r["op"] else "#c9a227" if "torch" in r["op"]
                  else "#2a7ab9" if "Q" in l else "#3a9d5d" for r, l in zip(rr, labels)]
        ax.barh(labels, vals, color=colors)
        ax.axvline(peak, color="#b03030", ls="--", lw=1)
        ax.text(peak - 2, -0.55, f"DRAM peak {peak:.0f} GB/s", color="#b03030", fontsize=8, va="bottom", ha="right")
        ax.set_title(f"{method}: achieved bandwidth, 192 MiB of KV")
        ax.set_xlim(0, peak * 1.12)
        ax.set_xlabel("GB/s (bytes read + written / kernel time)")
        for i, v in enumerate(vals):
            ax.text(v, i, f" {v:.0f}", va="center", fontsize=8)
    axes[0].invert_yaxis()
    fig.tight_layout()
    fig.savefig(out / "kernel_bandwidth.png", dpi=140)
    plt.close(fig)

    a = res["assembly"]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    methods = ["fp16", "int8", "int4"]
    w = 0.38
    xs = np.arange(3)
    cpu = [next(r["time_ms"] for r in a if r["method"] == m and "CPU" in r["impl"]) for m in methods]
    gpu = [next(r["time_ms"] for r in a if r["method"] == m and "CUDA" in r["impl"]) for m in methods]
    ax.bar(xs - w / 2, cpu, w, label="CPU decode + concat + H2D", color="#9aa5b1")
    ax.bar(xs + w / 2, gpu, w, label="CUDA BlockLoader", color="#2a7ab9")
    for x, c, gg in zip(xs, cpu, gpu):
        ax.text(x + w / 2, gg, f"{c / gg:.1f}x", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(xs, methods)
    ax.set_ylabel("ms per request (1024 cached tokens)")
    ax.set_title("Loading 16 cached OPT-1.3B blocks into the attention cache")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "assembly_speedup.png", dpi=140)
    plt.close(fig)


def replot(out: str = "results/cuda"):
    """Redraw the figures from bench.json (no GPU needed)."""
    o = Path(out)
    plot(json.loads((o / "bench.json").read_text()), o)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results/cuda")
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.backends.cudnn.benchmark = False
    dev = dict(ext().device_memory_info(0))
    dev["name"] = torch.cuda.get_device_name(0)
    dev["capability"] = ".".join(map(str, torch.cuda.get_device_capability()))
    dev["sms"] = torch.cuda.get_device_properties(0).multi_processor_count
    print("device:", dev, flush=True)
    res = {"device": dev}
    for name, fn in (("kernels", bench_kernels), ("end_to_end", bench_end_to_end),
                     ("pcie", bench_pcie), ("assembly", bench_assembly)):
        print(f"[{name}] ...", flush=True)
        res[name] = fn(args.iters)
    (out / "bench.json").write_text(json.dumps(res, indent=2))
    md = [f"# CUDA codec benchmarks\n\nDevice: {dev['name']} (sm_{dev['capability'].replace('.', '')}, "
          f"{dev['sms']} SMs), theoretical DRAM bandwidth {dev['peak_gbps']:.0f} GB/s "
          f"({dev['bus_width_bits']}-bit bus, {dev['memory_clock_khz'] / 1e6:.1f} GHz).\n",
          "## Kernels\n", _md(res["kernels"], ["shape", "op", "method", "time_ms", "gbps"]),
          "\n## End-to-end encode / decode of one block (host to host)\n",
          _md(res["end_to_end"], ["shape", "op", "method", "impl", "time_ms", "mb_per_s"]),
          "\n## PCIe\n", _md(res["pcie"], ["op", "memory", "mb", "time_ms", "gbps"]),
          "\n## Request cache assembly (load path)\n", _md(res["assembly"], ["op", "method", "impl", "time_ms", "h2d_mb"])]
    (out / "bench.md").write_text("\n".join(md) + "\n")
    plot(res, out)
    print((out / "bench.md").read_text())


if __name__ == "__main__":
    main()
