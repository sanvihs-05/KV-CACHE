"""Turn raw runs into the project's three answers.

  1. damage   : GB written per 1,000 tokens, and years until the SSD hits its TBW rating
  2./3. trade-off: % SSD writes saved vs % speed lost, relative to a baseline policy

    python -m kvwear.analysis report --experiments results/experiments.csv --tbw-tb 300
    python -m kvwear.analysis report --runs results/runs           # no SMART data: Linux/app bytes
    python -m kvwear.analysis project --gb-per-chat 2 --chats-per-day 50 --tbw-tb 300

Which byte count is "the" write count, best first:
  smart_kv_bytes       SMART delta minus idle writes minus the `no_ssd` control run's delta:
                       SSD writes caused by KV offloading alone. The control runs the same
                       model and workload with zero KV writes, so it cancels the rest of
                       running an LLM (e.g. Windows paging while the WSL VM holds the model).
  smart_llm_bytes      SSD's own host-write counter delta, minus idle background writes
                       (whole-system cost of the run, including that paging)
  linux_bytes_written  sectors Linux wrote to the WSL virtual disk (/proc/diskstats)
  app_bytes_written    bytes the KV store handed to the filesystem
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

BYTE_SOURCES = ["smart_kv_bytes", "smart_llm_bytes", "linux_bytes_written", "app_bytes_written"]
CONTROL_POLICY = "no_ssd"

# brand-neutral palette (light mode); family -> color. Text never wears series colors.
FAMILY_COLORS = {"baseline": "#8a8985", "quantize only": "#1baf7a", "reuse-aware": "#2a78d6", "oracle (bound)": "#eb6834"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def family(policy: str, spec: str | None = None) -> str:
    s = (spec or policy or "")
    if s.startswith("oracle") or policy.startswith("oracle"):
        return "oracle (bound)"
    if s.startswith("reuse") or policy.startswith("reuse"):
        return "reuse-aware"
    if any(c in s for c in ("int8", "int4", "zstd", "zlib")) or "compress" in policy:
        return "quantize only"
    return "baseline"


# ------------------------------------------------------------------ loading


def load_runs(experiments: str | None, runs_dir: str | None) -> pd.DataFrame:
    if experiments and Path(experiments).exists():
        df = pd.read_csv(experiments)
    else:
        recs = []
        for p in sorted(Path(runs_dir or "results/runs").glob("*/summary.json")):
            d = json.loads(p.read_text())
            recs.append({k: v for k, v in d.items() if not isinstance(v, (dict, list))})
        if not recs:
            raise SystemExit("no runs found")
        df = pd.DataFrame(recs)
    for c in df.columns:
        if df[c].dtype == object:
            conv = pd.to_numeric(df[c], errors="coerce")
            if conv.notna().sum() == df[c].notna().sum() and df[c].notna().any():
                df[c] = conv
    return df


def subtract_idle(df: pd.DataFrame) -> tuple[pd.DataFrame, float | None]:
    """SMART delta minus (idle write rate x measurement window)."""
    if "smart_bytes_delta" not in df.columns:
        return df, None
    idle = df[df["policy"] == "idle"]
    rate = None
    if len(idle) and idle["window_s"].sum() > 0:
        rate = float(idle["smart_bytes_delta"].sum() / idle["window_s"].sum())
    df = df[df["policy"] != "idle"].copy()
    if "exit_code" in df.columns:                     # failed runs carry no usable numbers
        df = df[df["exit_code"].fillna(0) == 0]
    df["smart_llm_bytes"] = df["smart_bytes_delta"] - (rate or 0.0) * df["window_s"]
    if "linux_bytes_written" in df.columns:
        df["virt_amplification"] = df["smart_llm_bytes"] / df["linux_bytes_written"].replace(0, np.nan)
    ctl = df[df["policy"] == CONTROL_POLICY]
    if len(ctl):
        # KV-attributable SSD writes: subtract what the same run costs with zero KV writes
        df["smart_kv_bytes"] = df["smart_llm_bytes"] - ctl["smart_llm_bytes"].mean()
        if "linux_bytes_written" in df.columns:
            lin = df["linux_bytes_written"] - ctl["linux_bytes_written"].mean()
            df["virt_amplification"] = (df["smart_kv_bytes"] / lin).where(lin > 0.2e9)
    return df, rate


def pick_source(df: pd.DataFrame) -> str:
    for c in BYTE_SOURCES:
        if c in df.columns and df[c].notna().all():
            return c
    raise SystemExit("runs have no write counters")


# ------------------------------------------------------------------ metrics


def wear_projection(bytes_per_chat: float, chats_per_day: float, tbw_tb: float) -> dict:
    daily = bytes_per_chat * chats_per_day
    yearly = daily * 365
    return {
        "gb_per_day": daily / 1e9,
        "tb_per_year": yearly / 1e12,
        "pct_tbw_per_year": 100 * yearly / (tbw_tb * 1e12) if tbw_tb else None,
        "years_to_tbw": (tbw_tb * 1e12) / yearly if yearly > 0 else float("inf"),
    }


def summarize(df: pd.DataFrame, source: str, baseline: str, tbw_tb: float, chats_per_day: float,
              speed: str = "latency_mean_s") -> pd.DataFrame:
    df = df.copy()
    df["tokens"] = df["prompt_tokens"] + df["gen_tokens"]
    # a control-subtracted count is noisy around zero; clipping each run would bias the mean up
    df["write_bytes"] = df[source] if source == "smart_kv_bytes" else df[source].clip(lower=0)
    df["gb_per_1k_tokens"] = df["write_bytes"] / 1e9 / (df["tokens"] / 1000)
    convs = df["num_conversations"] if "num_conversations" in df.columns else 1
    df["gb_per_chat"] = df["write_bytes"] / 1e9 / convs
    speed_col = speed if speed in df.columns else "wall_s"

    agg = {"write_bytes": ["mean", "std"], "gb_per_1k_tokens": ["mean", "std"], "gb_per_chat": "mean",
           speed_col: ["mean", "std"], "run_id": "count"}
    for c in ("ttft_mean_s", "wall_s", "app_bytes_written", "linux_bytes_written", "smart_llm_bytes",
              "virt_amplification", "ssd_hits", "recomputed_tokens"):
        if c in df.columns and c != speed_col:
            agg[c] = "mean"
    g = df.groupby("policy").agg(agg)
    g.columns = ["_".join(c).rstrip("_") if isinstance(c, tuple) else c for c in g.columns]
    g = g.rename(columns={"run_id_count": "runs", f"{speed_col}_mean": "speed_mean", f"{speed_col}_std": "speed_std"})
    # drop only the trailing aggregation suffix ("ttft_mean_s_mean" -> "ttft_mean_s")
    g.columns = [c[:-len("_mean")] if c != "speed_mean" and c.endswith("_mean") else c for c in g.columns]

    if baseline in g.index:
        b = g.loc[baseline]
        g["writes_saved_pct"] = 100 * (1 - g["write_bytes"] / b["write_bytes"]) if b["write_bytes"] > 0 else np.nan
        g["writes_saved_err"] = 100 * g["write_bytes_std"].fillna(0) / b["write_bytes"] if b["write_bytes"] > 0 else np.nan
        g["slowdown_pct"] = 100 * (g["speed_mean"] / b["speed_mean"] - 1)
        g["slowdown_err"] = 100 * g["speed_std"].fillna(0) / b["speed_mean"]
    proj = g["gb_per_chat"].apply(lambda x: pd.Series(wear_projection(x * 1e9, chats_per_day, tbw_tb)))
    g = g.join(proj)
    specs = df.groupby("policy")["policy_spec"].first() if "policy_spec" in df.columns else None
    g["family"] = [family(p, specs.get(p) if specs is not None else None) for p in g.index]
    g.attrs["speed_metric"] = speed_col
    return g.sort_values("write_bytes", ascending=False)


# ------------------------------------------------------------------ plots


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_tradeoff(g: pd.DataFrame, baseline: str, out: Path, source: str, filename: str = "tradeoff.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if "writes_saved_pct" not in g.columns:
        return None
    fig, ax = plt.subplots(figsize=(7.5, 5), dpi=150)
    _style(ax)
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.axvline(0, color=INK2, linewidth=0.8)
    for fam, color in FAMILY_COLORS.items():
        sub = g[g["family"] == fam]
        if sub.empty:
            continue
        ax.errorbar(sub["writes_saved_pct"], sub["slowdown_pct"], xerr=sub["writes_saved_err"],
                    yerr=sub["slowdown_err"], fmt="o", ms=8, color=color, ecolor=color, elinewidth=1,
                    capsize=0, mec="white", mew=2, label=fam, zorder=3)
    # direct labels; nudge a label down when a neighbour sits close above-left of it
    placed: list[tuple[float, float]] = []
    span_x = max(g["writes_saved_pct"].max() - g["writes_saved_pct"].min(), 1)
    span_y = max(g["slowdown_pct"].max() - g["slowdown_pct"].min(), 1)
    for name, r in g.dropna(subset=["writes_saved_pct", "slowdown_pct"]).sort_values("writes_saved_pct").iterrows():
        x, y = r["writes_saved_pct"], r["slowdown_pct"]
        crowded = any(abs(x - px) < 0.12 * span_x and abs(y - py) < 0.15 * span_y for px, py in placed)
        ax.annotate(name, (x, y), xytext=(7, -12 if crowded else 5), textcoords="offset points",
                    fontsize=8, color=INK)
        placed.append((x, y))
    # Pareto frontier: fewest writes for a given slowdown
    pts = g[["writes_saved_pct", "slowdown_pct"]].dropna().sort_values("slowdown_pct").values
    front, best = [], -np.inf
    for x, y in pts:
        if x > best:
            front.append((x, y))
            best = x
    if len(front) > 1:
        fx, fy = zip(*front)
        ax.plot(fx, fy, color=INK2, linewidth=1, linestyle="--", zorder=2, label="Pareto frontier")
    ax.set_xlabel(f"SSD writes saved vs {baseline} (%)", color=INK, fontsize=10)
    ax.set_ylabel(f"slower vs {baseline} (%, {g.attrs.get('speed_metric', 'latency')})", color=INK, fontsize=10)
    ax.set_title("SSD writes saved vs speed lost", color=INK, fontsize=12, loc="left", pad=20)
    ax.text(0, 1.015, f"bytes from: {source}; bars = 1 std over repetitions; lower-right is better",
            transform=ax.transAxes, fontsize=8, color=INK2, va="bottom")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper left")
    fig.tight_layout()
    p = out / filename
    fig.savefig(p)
    plt.close(fig)
    return p


def plot_writes(g: pd.DataFrame, out: Path, source: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g = g.sort_values("gb_per_1k_tokens")
    fig, ax = plt.subplots(figsize=(7.5, 0.45 * len(g) + 1.4), dpi=150)
    _style(ax)
    ax.grid(axis="y", visible=False)
    colors = [FAMILY_COLORS[f] for f in g["family"]]
    ax.barh(g.index, g["gb_per_1k_tokens"], xerr=g["gb_per_1k_tokens_std"].fillna(0), color=colors,
            height=0.6, edgecolor="white", linewidth=2, error_kw={"elinewidth": 1, "ecolor": INK2})
    for i, v in enumerate(g["gb_per_1k_tokens"]):
        ax.text(v, i, f"  {v:.3f}", va="center", fontsize=8, color=INK)
    ax.set_xlabel("GB written to SSD per 1,000 tokens", color=INK, fontsize=10)
    ax.set_title("SSD writes per policy", color=INK, fontsize=12, loc="left", pad=20)
    ax.text(0, 1.015, f"bytes from: {source}", transform=ax.transAxes, fontsize=8, color=INK2, va="bottom")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for f, c in FAMILY_COLORS.items() if f in set(g["family"])]
    labels = [f for f in FAMILY_COLORS if f in set(g["family"])]
    ax.legend(handles, labels, frameon=False, fontsize=8, labelcolor=INK2, loc="lower right")
    fig.tight_layout()
    p = out / "writes_per_policy.png"
    fig.savefig(p)
    plt.close(fig)
    return p


# ------------------------------------------------------------------ CLI


def report(args):
    df = load_runs(args.experiments, args.runs)
    df, idle_rate = subtract_idle(df)
    if df.empty:
        raise SystemExit("only idle runs found")
    source = args.source or pick_source(df)
    g = summarize(df, source, args.baseline, args.tbw_tb, args.chats_per_day, args.speed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    g.to_csv(out / "summary.csv")
    cols = [c for c in ["family", "runs", "gb_per_1k_tokens", "gb_per_chat", "writes_saved_pct", "slowdown_pct",
                        "ttft_mean_s", "speed_mean", "years_to_tbw", "pct_tbw_per_year", "virt_amplification"]
            if c in g.columns]
    lines = [f"# KV offload SSD wear report", "",
             f"- write counter used: `{source}`",
             f"- baseline: `{args.baseline}`; speed metric: `{g.attrs['speed_metric']}`",
             f"- wear projection: {args.chats_per_day:g} chats/day, SSD rated {args.tbw_tb:g} TBW"]
    if idle_rate is not None:
        lines.append(f"- idle background writes subtracted: {idle_rate * 3600 / 1e9:.3f} GB/hour")
    if source == "smart_kv_bytes":
        ctl = df[df["policy"] == CONTROL_POLICY]["smart_llm_bytes"]
        lines.append(f"- KV-attributable writes = SMART minus the `{CONTROL_POLICY}` control run "
                     f"({ctl.mean() / 1e9:.2f} ± {ctl.std() / 1e9:.2f} GB per run: everything else the LLM run "
                     "writes, mostly Windows paging)")
    try:
        table = g[cols].to_markdown(floatfmt=".3f")
    except ImportError:                              # pandas needs `tabulate` for markdown
        table = "```\n" + g[cols].round(3).to_string() + "\n```"
    lines += ["", table, ""]
    (out / "summary.md").write_text("\n".join(lines))
    p1 = plot_tradeoff(g, args.baseline, out, source)
    alt = "ttft_mean_s" if args.speed != "ttft_mean_s" else "latency_mean_s"
    if alt in df.columns:   # same plot with the other speed metric, for comparison
        plot_tradeoff(summarize(df, source, args.baseline, args.tbw_tb, args.chats_per_day, alt),
                      args.baseline, out, source, filename=f"tradeoff_{alt}.png")
    p2 = plot_writes(g, out, source)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(g[cols].round(3).to_string())
    print(f"\nwrote {out / 'summary.csv'}" + (f", {p1}" if p1 else "") + f", {p2}")


def project(args):
    r = wear_projection(args.gb_per_chat * 1e9, args.chats_per_day, args.tbw_tb)
    print(f"{args.gb_per_chat:g} GB/chat x {args.chats_per_day:g} chats/day = {r['gb_per_day']:.1f} GB/day "
          f"= {r['tb_per_year']:.1f} TB/year")
    print(f"= {r['pct_tbw_per_year']:.1f}% of a {args.tbw_tb:g} TBW rating per year "
          f"-> rated limit reached in {r['years_to_tbw']:.1f} years (LLM writes alone)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("report")
    r.add_argument("--experiments", default="results/experiments.csv")
    r.add_argument("--runs", default="results/runs")
    r.add_argument("--baseline", default="write_through")
    r.add_argument("--source", choices=BYTE_SOURCES)
    r.add_argument("--speed", default="latency_mean_s",
                   help="speed metric for the trade-off: latency_mean_s, ttft_mean_s, latency_p95_s, wall_s")
    r.add_argument("--tbw-tb", type=float, default=300)
    r.add_argument("--chats-per-day", type=float, default=50)
    r.add_argument("--out", default="results/report")
    r.set_defaults(fn=report)
    p = sub.add_parser("project")
    p.add_argument("--gb-per-chat", type=float, required=True)
    p.add_argument("--chats-per-day", type=float, default=50)
    p.add_argument("--tbw-tb", type=float, default=300)
    p.set_defaults(fn=project)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
