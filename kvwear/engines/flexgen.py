"""Drive FlexLLMGen as the off-the-shelf offloading baseline (step 1: measure the damage).

FlexLLMGen's --percent takes six numbers:
    weights_gpu weights_cpu  cache_gpu cache_cpu  activations_gpu activations_cpu
Whatever isn't on GPU or CPU goes to disk (--offload-dir). The default here, 100 0 0 0 100 0,
keeps weights on the GPU and sends *all* of the attention (KV) cache to the SSD.
--compress-cache turns on FlexLLMGen's own 4-bit group-wise cache compression.

Why the reuse-aware policy can't live inside FlexLLMGen: it does one batched generation
at a time, and every decode step reads the whole KV cache back (attention needs all of
it). Inside one generation no written block goes unread, so there is nothing for trick #1
to skip. Skipping pays off only *across* requests (multi-turn reuse), which FlexLLMGen
doesn't do. That part runs in the multi-turn engine (runner.py --engine hf).
"""
from __future__ import annotations

import re
import subprocess
import sys
import time

from ..diskstats import DirSizeMonitor


def build_command(cfg, offload_dir: str, log_file: str) -> list[str]:
    e = cfg.engine
    cmd = [sys.executable, "-m", "flexllmgen.flex_opt",
           "--model", e.model,
           "--percent", *[str(int(p)) for p in e.flexgen_percent],
           "--offload-dir", offload_dir,
           "--prompt-len", str(e.flexgen_prompt_len),
           "--gen-len", str(e.flexgen_gen_len),
           "--gpu-batch-size", str(e.flexgen_gpu_batch_size),
           "--num-gpu-batches", str(e.flexgen_num_gpu_batches),
           "--log-file", log_file]
    if e.flexgen_compress_cache:
        cmd.append("--compress-cache")
    cmd += [str(a) for a in e.flexgen_extra_args]
    return cmd


_METRIC = re.compile(r"(prefill|decode|total)\s+(latency|throughput):\s*([\d.]+)")
_SIZE = re.compile(r"(model size|cache size|hidden size \(p\)|peak gpu mem):\s*([\d.]+)\s*GB")


def parse_output(text: str) -> dict:
    out = {}
    for stage, kind, val in _METRIC.findall(text):
        out[f"flexgen_{stage}_{kind}"] = float(val)
    for name, val in _SIZE.findall(text):
        out["flexgen_" + re.sub(r"\W+", "_", name).strip("_") + "_gb"] = float(val)
    return out


def run_flexgen(cfg, offload_dir: str, log_file: str, verbose: bool = True) -> dict:
    cmd = build_command(cfg, offload_dir, log_file)
    print("[flexgen] " + " ".join(cmd), flush=True)
    mon = DirSizeMonitor(offload_dir, interval=1.0, verbose=verbose)
    mon.start()
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.perf_counter() - t0
    mon.stop()
    text = proc.stdout + "\n" + proc.stderr
    if verbose:
        print(text[-4000:], flush=True)
    if proc.returncode != 0:
        raise RuntimeError(f"FlexLLMGen exited with {proc.returncode}; see output above")
    e = cfg.engine
    seqs = e.flexgen_gpu_batch_size * e.flexgen_num_gpu_batches
    res = {
        "wall_s": wall,
        "prompt_tokens": seqs * e.flexgen_prompt_len,
        "gen_tokens": seqs * e.flexgen_gen_len,
        "offload_dir_peak_bytes": mon.peak,
        "flexgen_cmd": " ".join(cmd),
        **parse_output(text),
    }
    return res
