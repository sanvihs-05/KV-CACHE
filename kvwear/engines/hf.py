"""Real-model engine on HF transformers.

Per turn: rebuild a DynamicCache from the reused KV blocks, prefill only the uncached
tokens (in chunks, to fit a 4 GB GPU), greedily decode `gen_tokens`, then copy the KV of
the new full blocks back to host memory for the store.
"""
from __future__ import annotations

import time

import numpy as np

from ..workload import HFTokenizer
from . import TurnResult


def _cache_layers(cache):
    """(key, value) per layer, for any transformers cache flavour."""
    if isinstance(cache, (tuple, list)):
        return list(cache)
    if hasattr(cache, "layers") and cache.layers and hasattr(cache.layers[0], "keys"):
        return [(l.keys, l.values) for l in cache.layers]          # transformers >= 4.54
    if hasattr(cache, "key_cache"):
        return list(zip(cache.key_cache, cache.value_cache))
    return list(cache.to_legacy_cache())


class HFEngine:
    def __init__(self, cfg):
        import torch
        from transformers import AutoModelForCausalLM

        self.torch = torch
        self.cfg = cfg
        self.device = cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu"
        dtype = getattr(torch, cfg.dtype)
        self.model = AutoModelForCausalLM.from_pretrained(cfg.model, torch_dtype=dtype).to(self.device).eval()
        c = self.model.config
        self.layers = c.num_hidden_layers
        self.kv_heads = getattr(c, "num_key_value_heads", None) or c.num_attention_heads
        self.head_dim = getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads
        self.max_context = getattr(c, "max_position_embeddings", 2048)
        self.tokenizer = HFTokenizer(cfg.model)
        gpu = torch.cuda.get_device_name(0) if self.device.startswith("cuda") else "cpu"
        self.description = f"hf:{cfg.model} on {gpu}"
        self.loader = None          # set by enable_gpu_loader(): CUDA cache assembly

    # SSD hits may arrive still quantized (EncodedBlock) once the GPU loader is on
    accepts_encoded = True

    def enable_gpu_loader(self) -> None:
        from ..cuda import BlockLoader
        if not self.device.startswith("cuda"):
            raise RuntimeError("the GPU loader needs a CUDA device")
        self.loader = BlockLoader()
        self.description += " +cuda-codec"

    @property
    def bytes_per_token(self) -> int:
        return self.layers * 2 * self.kv_heads * self.head_dim * 2

    def _sync(self):
        if self.device.startswith("cuda"):
            self.torch.cuda.synchronize()

    def build_cache(self, blocks: list):
        from transformers import DynamicCache
        cache = DynamicCache()
        if not blocks:
            return cache
        if self.loader is not None:
            # CUDA path: one pinned H2D copy + one dequantize/scatter kernel for all blocks;
            # per-layer K/V are views of one contiguous tensor (no extra copies)
            t = self.loader.assemble(blocks, self.device)          # [L, 2, H, T, D]
            for l in range(t.shape[0]):
                cache.update(t[l, 0].unsqueeze(0), t[l, 1].unsqueeze(0), l)
            return cache
        blocks = [b if isinstance(b, np.ndarray) else b.to_numpy() for b in blocks]
        arr = np.concatenate(blocks, axis=3)                      # [L, 2, H, T, D]
        t = self.torch.from_numpy(arr).to(self.device, non_blocking=True)
        for l in range(t.shape[0]):
            cache.update(t[l, 0].unsqueeze(0).contiguous(), t[l, 1].unsqueeze(0).contiguous(), l)
        return cache

    def forward_prompt(self, ids: list[int], cache, start: int):
        """Run tokens ids[start:] through the model on top of `cache`. Returns (logits, cache)."""
        torch = self.torch
        chunk = max(self.cfg.prefill_chunk, 1)
        logits = None
        pos = start
        while pos < len(ids):
            end = min(pos + chunk, len(ids))
            inp = torch.tensor([ids[pos:end]], device=self.device)
            mask = torch.ones((1, end), dtype=torch.long, device=self.device)
            out = self.model(input_ids=inp, past_key_values=cache, attention_mask=mask, use_cache=True)
            cache, logits = out.past_key_values, out.logits
            pos = end
        return logits, cache

    def run_turn(self, prompt_ids, blocks, n_cached, gen_tokens, kv_range) -> TurnResult:
        torch = self.torch
        with torch.inference_mode():
            self._sync()
            t0 = time.perf_counter()
            cache = self.build_cache(blocks)
            self._sync()
            t1 = time.perf_counter()
            logits, cache = self.forward_prompt(prompt_ids, cache, n_cached)
            self._sync()
            t2 = time.perf_counter()
            nxt = logits[:, -1:].argmax(-1)
            _ = nxt.item()                                        # first token is ready
            ttft = time.perf_counter() - t0
            t3 = time.perf_counter()
            total = len(prompt_ids)
            for _step in range(max(gen_tokens - 1, 0)):
                total += 1
                mask = torch.ones((1, total), dtype=torch.long, device=self.device)
                out = self.model(input_ids=nxt, past_key_values=cache, attention_mask=mask, use_cache=True)
                cache = out.past_key_values
                nxt = out.logits[:, -1:].argmax(-1)
            self._sync()
            t4 = time.perf_counter()
            s, e = kv_range
            if e > s:
                layers = _cache_layers(cache)
                k = torch.stack([kk[0, :, s:e] for kk, _ in layers])      # [L, H, T, D]
                v = torch.stack([vv[0, :, s:e] for _, vv in layers])
                kv = torch.stack([k, v], dim=1).to(torch.float16).cpu().numpy()
            else:
                kv = np.zeros((self.layers, 2, self.kv_heads, 0, self.head_dim), np.float16)
            t5 = time.perf_counter()
            del cache
        return TurnResult(t1 - t0, t2 - t1, ttft, t4 - t3, t5 - t4, kv)
