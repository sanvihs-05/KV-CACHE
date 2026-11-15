#!/usr/bin/env bash
# Week-1 check, Linux half. Proves the KV cache really goes to disk:
#   1) FlexLLMGen with 100% of the attention cache on disk; prints the offload folder size every second
#   2) the multi-turn engine with write_through on the small smoke config
# For each it prints the folder's peak size and what /proc/diskstats says Linux wrote.
# Run on its own, or via windows/week1_check.ps1, which adds SMART readings around it.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source ~/.venvs/kvwear/bin/activate
export PYTHONDONTWRITEBYTECODE=1

report() {   # $1 = KVWEAR_RESULT json line
  python - "$1" <<'PY'
import json, sys
d = json.loads(sys.argv[1].split(" ", 1)[1])
lin, peak = d.get("linux_bytes_written") or 0, d.get("offload_dir_peak_bytes") or 0
print(f"  offload folder peak : {peak/1e9:8.3f} GB")
print(f"  Linux wrote         : {lin/1e9:8.3f} GB  (device {d.get('linux_device')})")
if "app_bytes_written" in d:
    print(f"  KV store wrote      : {d['app_bytes_written']/1e9:8.3f} GB")
if "flexgen_cache_size_gb" in d:
    print(f"  FlexLLMGen cache    : {d['flexgen_cache_size_gb']:8.3f} GB (its own estimate)")
if peak > 0 and lin < 0.5 * peak:
    print("  NOTE: Linux wrote less than the folder held. The files were probably deleted before the")
    print("        kernel flushed them (dirty pages discarded), so part of the 'offload' never hit the disk.")
print(f"CHECK_LINUX_BYTES={int(lin)}")
print(f"CHECK_PEAK_BYTES={int(peak)}")
PY
}

if [ "${1:-}" != "--skip-flexgen" ]; then
  echo "== 1/2 FlexLLMGen, KV cache 100% on disk (--percent 100 0 0 0 100 0)"
  out=$(python -m kvwear.runner --config configs/flexgen.yaml --engine flexgen --run-id week1-flexgen 2>&1 | tee >(cat >&2) | grep '^KVWEAR_RESULT' || true)
  if [ -z "$out" ]; then echo "FlexLLMGen run failed (see output above)"; exit 1; fi
  report "$out"
fi

echo "== 2/2 multi-turn engine, write_through, smoke config"
out=$(python -m kvwear.runner --config configs/smoke.yaml --policy write_through --run-id week1-multiturn 2>&1 | tee >(cat >&2) | grep '^KVWEAR_RESULT' || true)
if [ -z "$out" ]; then echo "multi-turn run failed (see output above)"; exit 1; fi
report "$out"
