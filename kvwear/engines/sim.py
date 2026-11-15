"""Simulated engine: KV-shaped tensors and modeled compute time, no GPU needed.

Use it to develop policies, run threshold sweeps and test the pipeline. Its SSD writes are
real, but its *speed* numbers are a model, so never report them as measurements.
"""
from __future__ import annotations

import time

import numpy as np

from ..workload import FakeTokenizer
from . import KNOWN_SHAPES, TurnResult


class SimEngine:
    def __init__(self, cfg):
        self.cfg = cfg
        if cfg.model in KNOWN_SHAPES:
            self.layers, self.kv_heads, self.head_dim = KNOWN_SHAPES[cfg.model]
        else:
            from transformers import AutoConfig
            c = AutoConfig.from_pretrained(cfg.model)
            self.layers = c.num_hidden_layers
            self.kv_heads = getattr(c, "num_key_value_heads", None) or c.num_attention_heads
            self.head_dim = getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads
        self.tokenizer = FakeTokenizer()
        self.max_context = 2048
        self.rng = np.random.default_rng(0)
        self.description = f"sim:{cfg.model}"

    @property
    def bytes_per_token(self) -> int:
        return self.layers * 2 * self.kv_heads * self.head_dim * 2

    def run_turn(self, prompt_ids, blocks, n_cached, gen_tokens, kv_range) -> TurnResult:
        t0 = time.perf_counter()
        if blocks:   # touch the data like a host->device copy would
            _ = sum(float(b.reshape(-1)[0]) for b in blocks)
        t1 = time.perf_counter()
        time.sleep(self.cfg.sim_prefill_s_per_token * (len(prompt_ids) - n_cached))
        t2 = time.perf_counter()
        time.sleep(self.cfg.sim_decode_s_per_token)
        ttft = time.perf_counter() - t0
        t3 = time.perf_counter()
        time.sleep(self.cfg.sim_decode_s_per_token * max(gen_tokens - 1, 0))
        t4 = time.perf_counter()
        s, e = kv_range
        shape = (self.layers, 2, self.kv_heads, max(e - s, 0), self.head_dim)
        kv = self.rng.standard_normal(shape, dtype=np.float32).astype(np.float16)
        t5 = time.perf_counter()
        return TurnResult(t1 - t0, t2 - t1, ttft, t4 - t3, t5 - t4, kv)
