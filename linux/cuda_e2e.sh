#!/usr/bin/env bash
# End-to-end: the real model, same trace, CPU codec vs CUDA codec (store.codec_device).
#   bash linux/cuda_e2e.sh [reps]
# Writes results/cuda/e2e/runs/<id>/ and prints a comparison table.
set -uo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source ~/.venvs/kvwear/bin/activate
export PYTHONDONTWRITEBYTECODE=1 CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.1}
REPS=${1:-3}
OUT=results/cuda/e2e
mkdir -p "$OUT"
python -c "from kvwear import cuda; assert cuda.available(), cuda.unavailable_reason(); print('cuda ext ok')" || exit 1
run() {  # config policy device rep extra...
  local cfg=$1 pol=$2 dev=$3 rep=$4; shift 4
  local id; id="$(basename "$cfg" .yaml)-$(echo "$pol" | tr '/@+' '___')-$dev-r$rep"
  echo "[$(date +%H:%M:%S)] $id"
  python -W ignore -m kvwear.runner --config "$cfg" --policy "$pol" --run-id "$id" --quiet \
    --set "results_dir=$OUT" --set "store.codec_device=$dev" "$@" > /dev/null 2>"$OUT/$id.err" \
    || { echo "   FAILED (see $OUT/$id.err)"; return; }
  rm -f "$OUT/$id.err"
}
for rep in $(seq 1 "$REPS"); do
  for pol in spill_int8 reuse_int8 spill_int4 write_through; do
    for dev in $(printf 'cpu\ncuda\n' | shuf); do
      run configs/default.yaml "$pol" "$dev" "$rep" --set store.ram_mb=256
    done
  done
  for pol in reuse_int8 spill_int4; do
    for dev in $(printf 'cpu\ncuda\n' | shuf); do
      run configs/qwen_long.yaml "$pol" "$dev" "$rep"
    done
  done
done
python -m kvwear.cuda.compare_e2e --runs "$OUT/runs" --out "$OUT"
