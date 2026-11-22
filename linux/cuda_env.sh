#!/usr/bin/env bash
# Run a command in the kvwear venv with the CUDA toolkit on PATH (for the CUDA extension).
#   bash linux/cuda_env.sh python -m kvwear.cuda.bench
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source ~/.venvs/kvwear/bin/activate
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.1}
export PATH="$CUDA_HOME/bin:$PATH" PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
exec "$@"
