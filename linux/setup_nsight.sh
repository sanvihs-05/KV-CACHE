#!/usr/bin/env bash
# Make Nsight Compute (ncu) and Nsight Systems (nsys) usable inside WSL without downloading them:
# the Windows installers already ship the Linux command-line tools ("target" directories).
#   sudo bash linux/setup_nsight.sh
# Installs to /opt/nv and links /usr/local/bin/ncu and /usr/local/bin/nsys.
set -euo pipefail
NV="/mnt/c/Program Files/NVIDIA Corporation"
ncu_win=$(ls -d "$NV"/Nsight\ Compute\ * 2>/dev/null | sort -V | tail -1 || true)
nsys_win=$(ls -d "$NV"/Nsight\ Systems\ * 2>/dev/null | sort -V | tail -1 || true)
if [[ -z "$ncu_win" ]]; then
    echo "Nsight Compute not found under $NV (install it on Windows, or: apt install cuda-nsight-compute-12-1)" >&2
    exit 1
fi
echo "ncu  from: $ncu_win"
rm -rf /opt/nv/ncu-root
mkdir -p /opt/nv/ncu-root/target
# ncu looks for its section/rule files at <exe>/../../sections
cp -r "$ncu_win/target/linux-desktop-glibc_2_11_3-x64" /opt/nv/ncu-root/target/
cp -r "$ncu_win/sections" /opt/nv/ncu-root/
[[ -d "$ncu_win/extras" ]] && cp -r "$ncu_win/extras" /opt/nv/ncu-root/
chmod +x /opt/nv/ncu-root/target/linux-desktop-glibc_2_11_3-x64/ncu
ln -sf /opt/nv/ncu-root/target/linux-desktop-glibc_2_11_3-x64/ncu /usr/local/bin/ncu
if [[ -n "$nsys_win" && -d "$nsys_win/target-linux-x64" ]]; then
    echo "nsys from: $nsys_win"
    rm -rf /opt/nv/nsys-root
    mkdir -p /opt/nv/nsys-root
    cp -r "$nsys_win/target-linux-x64" /opt/nv/nsys-root/
    chmod +x /opt/nv/nsys-root/target-linux-x64/nsys
    ln -sf /opt/nv/nsys-root/target-linux-x64/nsys /usr/local/bin/nsys
fi
ncu --version | tail -1
cat <<'EOF'

GPU performance counters are off for non-admin users on Windows by default (ncu then reports
ERR_NVGPUCTRPERM, also inside WSL). Enable them once, on Windows:
  NVIDIA Control Panel -> Desktop menu -> Enable Developer Settings
  -> Developer -> Manage GPU Performance Counters -> "Allow access to the GPU performance
     counters to all users" -> Apply
Then: bash linux/profile_ncu.sh
EOF
