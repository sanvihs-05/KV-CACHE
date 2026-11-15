#!/usr/bin/env bash
# Linux-counter preliminary sweeps for the Qwen2.5-0.5B configs.
cd "$(dirname "$0")/.."
CONFIG=configs/qwen.yaml POLICIES='write_through spill spill_int8 reuse reuse_int8 oracle no_ssd' bash linux/run_sweep.sh results/prelim/qwen 2 > ~/qwen_sweep.log 2>&1
CONFIG=configs/qwen_long.yaml POLICIES='write_through spill reuse reuse_int8 oracle no_ssd' bash linux/run_sweep.sh results/prelim/qwen_long 2 > ~/qwen_long_sweep.log 2>&1
echo done > ~/qwen_prelim.done
