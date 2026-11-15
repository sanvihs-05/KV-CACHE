"""Inference engines for the multi-turn experiment.

  hf   real model (HF transformers), real KV tensors, real GPU timings
  sim  no model: KV-shaped random tensors + modeled compute time. For developing and
       sweeping policies quickly on any machine. SSD writes are still real.

FlexLLMGen (single-shot offloading baseline) is driven separately: engines/flexgen.py.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# (layers, kv_heads, head_dim) for common small models, so `sim` needs no download
KNOWN_SHAPES = {
    "facebook/opt-125m": (12, 12, 64),
    "facebook/opt-350m": (24, 16, 64),
    "facebook/opt-1.3b": (24, 32, 64),
    "facebook/opt-2.7b": (32, 32, 80),
    "facebook/opt-6.7b": (32, 32, 128),
    "Qwen/Qwen2.5-0.5B-Instruct": (24, 2, 64),
    "Qwen/Qwen2.5-1.5B-Instruct": (28, 2, 128),
    "meta-llama/Llama-3.2-1B-Instruct": (16, 8, 64),
}


@dataclass
class TurnResult:
    load_s: float          # moving cached KV blocks onto the GPU
    prefill_s: float       # computing KV for the uncached part of the prompt
    ttft_s: float          # load + prefill + first token
    decode_s: float        # remaining generated tokens
    extract_s: float       # copying the new KV blocks back to host memory
    new_kv: np.ndarray     # [layers, 2, kv_heads, T, head_dim] for the requested token range


def make_engine(cfg):
    if cfg.kind == "hf":
        from .hf import HFEngine
        return HFEngine(cfg)
    if cfg.kind == "sim":
        from .sim import SimEngine
        return SimEngine(cfg)
    raise ValueError(f"engine '{cfg.kind}' is not a multi-turn engine (flexgen runs via engines/flexgen.py)")
