"""Compare end-to-end runs that differ only in store.codec_device (cpu vs cuda).

    python -m kvwear.cuda.compare_e2e --runs results/cuda/e2e/runs --out results/cuda/e2e
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

METRICS = ["latency_mean_s", "ttft_mean_s", "load_s_total", "encode_s", "write_thread_s", "ssd_read_s",
           "backpressure_s", "wall_s", "app_bytes_written", "ssd_hits"]


def load(runs: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(runs.glob("*/summary.json")):
        d = json.loads(p.read_text())
        d["run_dir"] = p.parent.name
        rows.append(d)
    df = pd.DataFrame(rows)
    df["config"] = df["config_name"]
    return df


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/cuda/e2e/runs")
    ap.add_argument("--out", default="results/cuda/e2e")
    a = ap.parse_args(argv)
    df = load(Path(a.runs))
    g = df.groupby(["config", "policy", "codec_device"])
    mean, std, n = g[METRICS].mean(), g[METRICS].std(), g.size()
    out_rows = []
    for (cfg, pol), sub in mean.groupby(level=[0, 1]):
        if not {"cpu", "cuda"} <= set(sub.index.get_level_values(2)):
            continue
        c, u = sub.xs("cpu", level=2).iloc[0], sub.xs("cuda", level=2).iloc[0]
        sc, su = std.loc[(cfg, pol, "cpu")], std.loc[(cfg, pol, "cuda")]
        out_rows.append({
            "config": cfg, "policy": pol, "reps": int(min(n[(cfg, pol, "cpu")], n[(cfg, pol, "cuda")])),
            "bytes_identical": bool(c["app_bytes_written"] == u["app_bytes_written"]),
            "ssd_hits": int(c["ssd_hits"]),
            "latency_cpu_s": c["latency_mean_s"], "latency_cuda_s": u["latency_mean_s"],
            "latency_delta_pct": 100 * (u["latency_mean_s"] / c["latency_mean_s"] - 1),
            "latency_noise_pct": 100 * max(sc["latency_mean_s"], su["latency_mean_s"]) / c["latency_mean_s"],
            "ttft_cpu_s": c["ttft_mean_s"], "ttft_cuda_s": u["ttft_mean_s"],
            "ttft_delta_pct": 100 * (u["ttft_mean_s"] / c["ttft_mean_s"] - 1),
            "encode_cpu_s": c["encode_s"], "encode_cuda_s": u["encode_s"],
            "load_plus_read_cpu_s": c["load_s_total"] + c["ssd_read_s"],
            "load_plus_read_cuda_s": u["load_s_total"] + u["ssd_read_s"],
            "wall_cpu_s": c["wall_s"], "wall_cuda_s": u["wall_s"],
        })
    res = pd.DataFrame(out_rows)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "compare.csv", index=False)
    cols = ["config", "policy", "reps", "bytes_identical", "ssd_hits", "encode_cpu_s", "encode_cuda_s",
            "load_plus_read_cpu_s", "load_plus_read_cuda_s", "ttft_delta_pct", "latency_delta_pct",
            "latency_noise_pct", "wall_cpu_s", "wall_cuda_s"]
    try:
        md = res[cols].to_markdown(index=False, floatfmt=".3f")
    except Exception:
        md = res[cols].to_string(index=False)
    (out / "compare.md").write_text("# End-to-end: CPU codec vs CUDA codec\n\n"
                                    "encode = total codec time in the SSD writer thread; load+read = SSD read + "
                                    "decode + host-to-GPU cache assembly, summed over the run (s).\n\n" + md + "\n")
    print(md)


if __name__ == "__main__":
    main()
