"""Experiment configuration (YAML file + CLI overrides)."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class WorkloadConfig:
    seed: int = 0
    num_conversations: int = 20
    concurrency: int = 4              # conversations "open" at the same time (interleaved turns)
    mix: dict = field(default_factory=lambda: {"doc_qa": 0.35, "debug": 0.30, "general": 0.35})
    num_shared_docs: int = 3          # several users paste the same document -> shared prefix
    system_prompt_tokens: int = 200
    doc_tokens: tuple = (700, 1000)
    log_tokens: tuple = (200, 400)
    user_tokens: tuple = (15, 60)
    reply_tokens: tuple = (60, 140)
    turns: dict = field(default_factory=lambda: {"doc_qa": (3, 7), "debug": (2, 5), "general": (1, 6)})
    gen_tokens: int = 32              # tokens decoded per turn (timed, then discarded)
    max_context_tokens: int = 2000    # prompt budget; oldest turns are dropped beyond this


@dataclass
class StoreConfig:
    block_tokens: int = 64
    ram_mb: float = 1024
    ssd_gb: float = 20
    offload_dir: str = "~/kvwear_offload"
    async_writes: bool = True
    max_pending_mb: float = 512       # writer back-pressure
    fsync: bool = False               # fsync every block file (slower, but writes hit the device at once)
    bypass_page_cache: bool = False   # fsync + fadvise(DONTNEED) so SSD reads really come from the SSD


@dataclass
class EngineConfig:
    kind: str = "hf"                  # hf | sim | flexgen
    model: str = "facebook/opt-1.3b"
    dtype: str = "float16"
    device: str = "cuda"
    prefill_chunk: int = 512          # chunked prefill keeps peak VRAM low on a 4 GB GPU
    # sim engine: synthetic compute cost (seconds) -- only for developing policies without a GPU
    sim_prefill_s_per_token: float = 0.0004
    sim_decode_s_per_token: float = 0.02
    # flexgen engine
    flexgen_percent: tuple = (100, 0, 0, 0, 100, 0)
    flexgen_prompt_len: int = 512
    flexgen_gen_len: int = 32
    flexgen_gpu_batch_size: int = 4
    flexgen_num_gpu_batches: int = 1
    flexgen_compress_cache: bool = False
    flexgen_extra_args: tuple = ()


@dataclass
class ExperimentConfig:
    name: str = "default"
    policy: str = "write_through"
    results_dir: str = "results"
    workload: WorkloadConfig = field(default_factory=WorkloadConfig)
    store: StoreConfig = field(default_factory=StoreConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _merge(dc, values: dict):
    known = {f.name: f for f in fields(dc)}
    for k, v in (values or {}).items():
        if k not in known:
            raise KeyError(f"unknown config key '{k}' for {type(dc).__name__}")
        cur = getattr(dc, k)
        if dataclasses.is_dataclass(cur):
            _merge(cur, v)
        else:
            if isinstance(cur, tuple) and isinstance(v, list):
                v = tuple(v)
            if isinstance(cur, dict) and isinstance(v, dict):
                v = {kk: tuple(vv) if isinstance(vv, list) else vv for kk, vv in v.items()}
            setattr(dc, k, v)
    return dc


def _parse_scalar(text: str) -> Any:
    return yaml.safe_load(text)


def apply_overrides(cfg, overrides: list[str] | None) -> None:
    """Dotted overrides, e.g. store.ram_mb=512, applied in place."""
    for ov in overrides or []:
        key, _, val = ov.partition("=")
        parts = key.strip().split(".")
        nested: dict = {}
        cur = nested
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = _parse_scalar(val)
        _merge(cfg, nested)


def load_config(path: str | None, overrides: list[str] | None = None) -> ExperimentConfig:
    cfg = ExperimentConfig()
    if path:
        data = yaml.safe_load(Path(path).read_text()) or {}
        _merge(cfg, data)
    apply_overrides(cfg, overrides)
    return cfg
