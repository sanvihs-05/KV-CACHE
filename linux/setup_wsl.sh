#!/usr/bin/env bash
# One-time setup inside WSL2 Ubuntu. Run from the project folder:  bash linux/setup_wsl.sh
# Puts the venv, FlexLLMGen and model caches on the Linux ext4 disk, never in OneDrive.
set -euo pipefail
cd "$(dirname "$0")/.."

VENV=~/.venvs/kvwear
SRC=~/src

echo "== checking GPU passthrough"
if ! command -v nvidia-smi >/dev/null; then
  echo "nvidia-smi not found in WSL. Install/update the NVIDIA Windows driver (it provides CUDA to WSL);"
  echo "do NOT install a Linux NVIDIA driver inside WSL."; exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

echo "== system packages"
sudo apt-get update -y
sudo apt-get install -y python3-venv python3-pip git

echo "== python venv at $VENV"
python3 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip wheel
# Pinned: FlexLLMGen is archived, so freeze a combination that works. torch 2.5 still
# defaults to torch.load(weights_only=False), which FlexLLMGen's weight converter needs.
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements-wsl.txt

echo "== FlexLLMGen (archived upstream; installed from source, commit recorded below)"
mkdir -p "$SRC"
if [ ! -d "$SRC/FlexLLMGen" ]; then
  git clone https://github.com/FMInference/FlexLLMGen "$SRC/FlexLLMGen"
fi
python -m pip install --no-deps -e "$SRC/FlexLLMGen"
git -C "$SRC/FlexLLMGen" rev-parse HEAD > linux/flexllmgen-commit.txt

echo "== patching FlexLLMGen for laptop RAM (pinned relay buffer 2 GiB/thread -> 256 MiB)"
if git -C "$SRC/FlexLLMGen" apply --check "$PWD/linux/flexllmgen-laptop.patch" 2>/dev/null; then
  git -C "$SRC/FlexLLMGen" apply "$PWD/linux/flexllmgen-laptop.patch"
else
  echo "   (already applied)"
fi

echo "== model weights (only pytorch_model.bin + configs; ~2.9 GB) and FlexLLMGen's numpy copy"
python - <<'PY'
from huggingface_hub import snapshot_download
for m in ["facebook/opt-125m", "facebook/opt-1.3b"]:
    snapshot_download(m, allow_patterns=["*.json", "*.txt", "pytorch_model.bin"])
import os
if not os.path.isdir(os.path.expanduser("~/opt_weights/opt-1.3b-np")):
    from flexllmgen.opt_config import download_opt_weights
    download_opt_weights("opt-1.3b", "~/opt_weights")
PY

echo "== CUDA compiler for kvwear's kernels (nvcc + runtime headers only, ~0.3 GB; optional)"
# kvwear/cuda is JIT-compiled with nvcc on first use. Only nvcc and cudart headers are needed
# (the bindings avoid cuBLAS headers), so skip the multi-GB cuda-toolkit meta-package.
if [ ! -x /usr/local/cuda-12.1/bin/nvcc ]; then
  if ! apt-cache policy cuda-nvcc-12-1 | grep -q Candidate; then
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/cuda-keyring.deb
    sudo dpkg -i /tmp/cuda-keyring.deb && sudo apt-get update -y
  fi
  sudo apt-get install -y --no-install-recommends cuda-nvcc-12-1 cuda-cudart-dev-12-1 ||     echo "   (CUDA compiler not installed: kvwear runs on the CPU codec path; GPU tests will skip)"
fi
python -m pip install ninja

echo "== freezing versions"
python -m pip freeze > linux/pip-freeze.txt
python - <<'PY'
import torch, transformers
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("transformers", transformers.__version__)
assert torch.cuda.is_available(), "CUDA not visible from WSL"
print("gpu", torch.cuda.get_device_name(0))
PY
echo "done. Activate with: source $VENV/bin/activate"
