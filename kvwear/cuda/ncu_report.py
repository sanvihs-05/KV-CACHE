"""Summarise an Nsight Compute profile of the kvwear kernels (made by linux/profile_ncu.sh).

    python -m kvwear.cuda.ncu_report --dir results/cuda/ncu

Reads raw.csv (`ncu --import kvwear.ncu-rep --csv --page raw`) and targets.json (launch order from
kvwear.cuda.ncu_targets) and writes ncu.md: per kernel the measured DRAM traffic and throughput,
SM / tensor-pipe utilisation, achieved vs theoretical occupancy, L2 hit rate, shared-memory bank
conflicts and the two dominant warp-stall reasons.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
from pathlib import Path

_SCALE = {"": 1, "byte": 1, "kbyte": 1e3, "mbyte": 1e6, "gbyte": 1e9, "nsecond": 1e-9, "usecond": 1e-6,
          "msecond": 1e-3, "second": 1, "%": 1, "inst": 1, "cycle": 1, "warp": 1, "ns": 1e-9, "us": 1e-6,
          "ms": 1e-3, "b": 1, "kb": 1e3, "mb": 1e6, "gb": 1e9}

COLS = {
    "time_us": "gpu__time_duration.sum",
    "dram_pct": "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram_read": "dram__bytes_read.sum",
    "dram_write": "dram__bytes_write.sum",
    "sm_pct": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "occ_achieved": "sm__warps_active.avg.pct_of_peak_sustained_active",
    "occ_theory": "sm__maximum_warps_per_active_cycle_pct",
    "l2_hit": "lts__t_sector_hit_rate.pct",
    "tensor_pct": "sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active",
    "bank_conflicts": "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum",
    "regs": "launch__registers_per_thread",
}
STALL = re.compile(r"^smsp__average_warps_issue_stalled_(\w+?)_per_issue_active\.ratio$")
SKIP_STALL = {"selected", "not_selected"}           # "selected" = issuing, not a stall


def _num(v: str, unit: str) -> float | None:
    v = (v or "").replace(",", "").strip()
    if v in ("", "n/a", "N/A"):
        return None
    try:
        x = float(v)
    except ValueError:
        return None
    return x * _SCALE.get(unit.strip().lower(), 1)


def parse_raw(text: str) -> list[dict]:
    """ncu --page raw --csv: header row, unit row, then one row per profiled kernel launch."""
    rows = list(csv.reader(io.StringIO(text)))
    rows = [r for r in rows if r and not r[0].startswith("==")]   # drop ==PROF== lines
    head, units, data = rows[0], rows[1], rows[2:]
    idx = {h: i for i, h in enumerate(head)}
    out = []
    for r in data:
        rec = {"id": r[idx["ID"]] if "ID" in idx else "", "kernel": r[idx.get("Kernel Name", 0)]}
        for key, col in COLS.items():
            i = idx.get(col)
            rec[key] = _num(r[i], units[i]) if i is not None and i < len(r) else None
        if rec["time_us"] is not None:
            rec["time_us"] *= 1e6
        stalls = []
        for h, i in idx.items():
            m = STALL.match(h)
            if m and m.group(1) not in SKIP_STALL and i < len(r):
                val = _num(r[i], units[i])
                if val:
                    stalls.append((val, m.group(1)))
        stalls.sort(reverse=True)
        rec["stalls"] = stalls[:2]
        out.append(rec)
    return out


def attach_targets(recs: list[dict], targets: list[str]) -> list[dict]:
    """Launch order -> target names. An attention target launches its partial kernel and then
    decode_combine; combine rows are folded into the previous target."""
    out, ti = [], 0
    for r in recs:
        if "decode_combine" in r["kernel"] and out:
            out[-1]["combine_us"] = (out[-1].get("combine_us") or 0) + (r["time_us"] or 0)
            continue
        r["target"] = targets[ti] if ti < len(targets) else r["kernel"]
        ti += 1
        out.append(r)
    return out


def _f(x, fmt="{:.0f}"):
    return "-" if x is None else fmt.format(x)


def write_md(recs: list[dict], path: Path):
    lines = ["# Nsight Compute profile of the kvwear kernels\n",
             "One launch per kernel (kvwear.cuda.ncu_targets), `--set full`. DRAM bytes are measured "
             "by the profiler, not computed. Occupancy: achieved / theoretical. Stalls: the two largest "
             "`smsp__average_warps_issue_stalled_*` (warps waiting per issued instruction).\n",
             "| kernel | time (us) | DRAM MB (r+w) | DRAM % peak | SM % | tensor pipe % | occupancy | L2 hit | "
             "smem bank conflicts | regs | top stalls |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in recs:
        mb = None if r["dram_read"] is None else (r["dram_read"] + (r["dram_write"] or 0)) / 1e6
        occ = f"{_f(r['occ_achieved'])}% / {_f(r['occ_theory'])}%"
        st = ", ".join(f"{n} {v:.1f}" for v, n in r["stalls"]) or "-"
        t = _f(r["time_us"], "{:.1f}")
        if r.get("combine_us"):
            t += f" (+{r['combine_us']:.1f} combine)"
        lines.append(f"| {r['target']} | {t} | {_f(mb, '{:.1f}')} | {_f(r['dram_pct'])}% | {_f(r['sm_pct'])}% | "
                     f"{_f(r['tensor_pct'], '{:.1f}')} | {occ} | {_f(r['l2_hit'])}% | {_f(r['bank_conflicts'])} | "
                     f"{_f(r['regs'])} | {st} |")
    path.write_text("\n".join(lines) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="results/cuda/ncu")
    args = ap.parse_args(argv)
    d = Path(args.dir)
    recs = parse_raw((d / "raw.csv").read_text())
    targets = json.loads((d / "targets.json").read_text()) if (d / "targets.json").exists() else []
    recs = attach_targets(recs, targets)
    (d / "ncu.json").write_text(json.dumps(recs, indent=1))
    write_md(recs, d / "ncu.md")
    print((d / "ncu.md").read_text())


if __name__ == "__main__":
    main()
