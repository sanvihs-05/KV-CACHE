#!/usr/bin/env bash
# Profile every kvwear CUDA kernel with Nsight Compute and write a report.
#   bash linux/profile_ncu.sh            # needs ncu (linux/setup_nsight.sh) + GPU counter access
# Output: results/cuda/ncu/kvwear.ncu-rep (open in the Nsight Compute GUI on Windows),
#         results/cuda/ncu/raw.csv, results/cuda/ncu/ncu.md (summary table, made by kvwear.cuda.ncu_report)
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=results/cuda/ncu
mkdir -p "$OUT"
NCU=${NCU:-ncu}
command -v "$NCU" >/dev/null || { echo "ncu not found: run sudo bash linux/setup_nsight.sh" >&2; exit 1; }
# kvwear's own kernels + the ceiling kernels (PyTorch's setup kernels are filtered out)
REGEX='regex:quant_int|dequant|paged_int8|decode_partial|decode_combine|stream_|peak_fp32'
export HOME=${HOME:-/root}
bash linux/cuda_env.sh "$NCU" --profile-from-start off --target-processes all \
    --kernel-name "$REGEX" --set full --import-source yes \
    -o "$OUT/kvwear" -f \
    python -m kvwear.cuda.ncu_targets --out "$OUT"
"$NCU" --import "$OUT/kvwear.ncu-rep" --csv --page raw > "$OUT/raw.csv"
bash linux/cuda_env.sh python -m kvwear.cuda.ncu_report --dir "$OUT"
echo "wrote $OUT/ncu.md"
