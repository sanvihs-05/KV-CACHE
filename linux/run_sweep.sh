#!/usr/bin/env bash
# Linux-only policy sweep (no SMART): every policy x REPS, shuffled order per repetition.
# Write counts come from the KV store and /proc/diskstats. Use it for development and for
# preliminary results; the SMART-bracketed version is windows/run_experiment.ps1.
#
#   bash linux/run_sweep.sh results/prelim/ram256 2 --set store.ram_mb=256
#   POLICIES="spill reuse reuse_int4" bash linux/run_sweep.sh results/x 3
set -uo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source ~/.venvs/kvwear/bin/activate
export PYTHONDONTWRITEBYTECODE=1

OUT=${1:?results dir}; REPS=${2:-2}; shift 2 || true
POLICIES=${POLICIES:-"write_through spill spill_int8 spill_int4 reuse@0.3/fp16 reuse reuse@0.7/fp16 reuse_int8 reuse_int4 oracle no_ssd"}
CONFIG=${CONFIG:-configs/default.yaml}
mkdir -p "$OUT"
for rep in $(seq 1 "$REPS"); do
  for p in $(printf '%s\n' $POLICIES | shuf); do
    id="$(date +%Y%m%d-%H%M%S)-$(echo "$p" | tr '/@+' '___')-r$rep"
    echo "[$(date +%H:%M:%S)] rep $rep policy $p -> $id"
    python -W ignore -m kvwear.runner --config "$CONFIG" --policy "$p" --run-id "$id" \
      --set "results_dir=$OUT" --quiet "$@" | grep '^KVWEAR_RESULT' > "$OUT/.last" \
      || echo "   FAILED: $p rep $rep"
    python - "$OUT/.last" <<'PY'
import json, sys
try:
    d = json.loads(open(sys.argv[1]).read().split(" ", 1)[1])
    print(f"   app {d['app_bytes_written']/1e9:6.2f} GB  linux {(d['linux_bytes_written'] or 0)/1e9:6.2f} GB  "
          f"ssd hits {d['ssd_hits']:4d}  recomputed {d['recomputed_tokens']:6d}  mean latency {d['latency_mean_s']:.3f}s")
except Exception as e:
    print("   (no result)", e)
PY
  done
done
rm -f "$OUT/.last"
python -m kvwear.analysis report --runs "$OUT/runs" --experiments none --out "$OUT/report"
