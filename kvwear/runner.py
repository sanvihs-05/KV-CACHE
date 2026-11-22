"""Experiment runner: one policy, one repetition, one results folder.

    python -m kvwear.runner --config configs/default.yaml --policy reuse_int4 --run-id r1
    python -m kvwear.runner --config configs/default.yaml --engine sim --policy spill
    python -m kvwear.runner --config configs/flexgen.yaml --engine flexgen --run-id fg1

Outputs results/runs/<run-id>/{summary.json,requests.csv} and prints one line
`KVWEAR_RESULT {...}` that windows/run_experiment.ps1 parses.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

from .blocks import block_hashes
from .config import ExperimentConfig, load_config
from .diskstats import DirSizeMonitor, DiskCounter
from .policy import ContinuationModel, RequestContext, make_policy
from .store import TieredKVStore, prepare_offload_dir
from .workload import generate_workload, turn_gap_stats


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _env_info() -> dict:
    info = {"python": platform.python_version(), "platform": platform.platform()}
    for mod in ("torch", "transformers", "numpy", "flexllmgen"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            pass
    try:
        import torch
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["cuda"] = torch.version.cuda
    except Exception:
        pass
    return info


def _pct(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    return xs[min(int(q * len(xs)), len(xs) - 1)]


def future_uses(requests, block_tokens) -> dict[str, list[int]]:
    """hash -> request indices that look the block up (for the oracle policy)."""
    uses: dict[str, list[int]] = {}
    for r in requests:
        for h in block_hashes(r.prompt_ids, block_tokens):
            uses.setdefault(h, []).append(r.index)
    return uses


def warn_offload_location(path: str):
    p = os.path.realpath(os.path.expanduser(path))
    if p.startswith("/mnt/"):
        print(f"WARNING: offload_dir {p} is on a Windows drive mounted into WSL (DrvFs). "
              "Writes bypass the ext4 virtual disk, are slow, and if it is under OneDrive they "
              "get uploaded. Use a folder in the Linux home, e.g. ~/kvwear_offload.", file=sys.stderr)
    if "onedrive" in p.lower():
        print("WARNING: offload_dir is inside OneDrive. OneDrive would upload every KV block.", file=sys.stderr)


def run_multiturn(cfg: ExperimentConfig, run_dir: Path, verbose: bool, train_set: list | None = None) -> dict:
    from .engines import make_engine

    engine = make_engine(cfg.engine)
    wcfg = cfg.workload
    wcfg.max_context_tokens = min(wcfg.max_context_tokens, engine.max_context - wcfg.gen_tokens)
    wl = generate_workload(wcfg, engine.tokenizer)
    B = cfg.store.block_tokens

    # reuse predictor is trained on a *different* trace than the one evaluated
    import copy
    train_cfg = copy.deepcopy(wcfg)
    train_cfg.seed = wcfg.seed + 10_007
    train_cfg.num_conversations = max(wcfg.num_conversations, 60)
    if train_set:       # distribution-shift test: train the predictor on a differently-shaped workload
        from .config import ExperimentConfig as _EC, apply_overrides
        holder = _EC(); holder.workload = train_cfg
        apply_overrides(holder, train_set)
        train_cfg = holder.workload
    counts, gaps = turn_gap_stats(generate_workload(train_cfg, engine.tokenizer))
    policy = make_policy(cfg.policy, continuation=ContinuationModel(counts, gaps),
                         future=future_uses(wl.requests, B))

    warn_offload_location(cfg.store.offload_dir)
    s = cfg.store
    gpu_decode = False
    if s.codec_device == "cuda":
        policy.codec.to_device("cuda")                       # GPU quantization in the writer thread
        if getattr(engine, "accepts_encoded", False):
            engine.enable_gpu_loader()                       # GPU dequantize + scatter on load
            gpu_decode = True
    elif s.codec_device != "cpu":
        raise ValueError(f"store.codec_device must be 'cpu' or 'cuda', not '{s.codec_device}'")
    store = TieredKVStore(policy, s.offload_dir, ram_bytes=int(s.ram_mb * 2**20), ssd_bytes=int(s.ssd_gb * 2**30),
                          async_writes=s.async_writes, max_pending_bytes=int(s.max_pending_mb * 2**20),
                          fsync=s.fsync, bypass_page_cache=s.bypass_page_cache, read_encoded=gpu_decode)
    disk = DiskCounter(store.root)
    mon = DirSizeMonitor(store.root, interval=2.0)
    mon.start()

    rows = []
    started = _now()
    t_run = time.perf_counter()
    for r in wl.requests:
        t0 = time.perf_counter()
        hashes = block_hashes(r.prompt_ids, B)
        ctx = RequestContext(r.index, r.conv_id, r.turn, r.has_doc)
        store.begin_request(ctx, hashes)
        ram0, ssd0 = store.stats.ram_hits, store.stats.ssd_hits
        tl0 = time.perf_counter()
        blocks = store.lookup_prefix(hashes)
        lookup_s = time.perf_counter() - tl0
        ram_hit, ssd_hit = store.stats.ram_hits - ram0, store.stats.ssd_hits - ssd0
        if len(blocks) * B >= len(r.prompt_ids):     # need >= 1 uncached token for logits
            blocks = blocks[:-1]
        n_cached = len(blocks) * B
        kv_range = (n_cached, len(hashes) * B)
        res = engine.run_turn(r.prompt_ids, blocks, n_cached, r.gen_tokens, kv_range)
        tp0 = time.perf_counter()
        for j, h in enumerate(hashes[len(blocks):]):
            store.put(h, res.new_kv[:, :, :, j * B:(j + 1) * B].copy())
        put_s = time.perf_counter() - tp0
        latency = time.perf_counter() - t0
        rows.append({
            "index": r.index, "conv_id": r.conv_id, "turn": r.turn, "kind": r.kind,
            "is_last_turn": int(r.is_last_turn), "prompt_tokens": len(r.prompt_ids),
            "cached_tokens": n_cached, "recomputed_tokens": len(r.prompt_ids) - n_cached,
            "ram_hit_blocks": ram_hit, "ssd_hit_blocks": ssd_hit,
            "lookup_s": lookup_s, "load_s": res.load_s, "prefill_s": res.prefill_s,
            "ttft_s": lookup_s + res.ttft_s, "decode_s": res.decode_s, "extract_s": res.extract_s,
            "put_s": put_s, "latency_s": latency, "gen_tokens": r.gen_tokens,
        })
        if verbose:
            print(f"[{r.index + 1:4d}/{len(wl.requests)}] conv {r.conv_id:3d} turn {r.turn} "
                  f"prompt {len(r.prompt_ids):5d} cached {n_cached:5d} (ram {ram_hit}, ssd {ssd_hit} blk) "
                  f"ttft {rows[-1]['ttft_s']:.3f}s total {latency:.3f}s", flush=True)
    serve_s = time.perf_counter() - t_run
    drain_s = store.close()
    wall_s = time.perf_counter() - t_run
    disk_delta = disk.delta(do_sync=True)
    mon.stop()

    lat = [x["latency_s"] for x in rows]
    ttft = [x["ttft_s"] for x in rows]
    dec_s = sum(x["decode_s"] for x in rows)
    dec_tok = sum(max(x["gen_tokens"] - 1, 0) for x in rows)
    summary = {
        "started": started, "finished": _now(), "engine": engine.description,
        "wall_s": wall_s, "serve_s": serve_s, "drain_s": drain_s,
        "num_requests": len(rows), "num_conversations": len(wl.conversations),
        "prompt_tokens": wl.prompt_tokens, "gen_tokens": wl.gen_tokens,
        "cached_tokens": sum(x["cached_tokens"] for x in rows),
        "recomputed_tokens": sum(x["recomputed_tokens"] for x in rows),
        "kv_bytes_per_token_fp16": engine.bytes_per_token,
        "latency_mean_s": statistics.fmean(lat), "latency_p50_s": _pct(lat, 0.5), "latency_p95_s": _pct(lat, 0.95),
        "ttft_mean_s": statistics.fmean(ttft), "ttft_p95_s": _pct(ttft, 0.95),
        "decode_tok_per_s": dec_tok / dec_s if dec_s > 0 else None,
        "offload_dir_peak_bytes": mon.peak, "codec_device": cfg.store.codec_device,
        "load_s_total": sum(x["load_s"] for x in rows), "extract_s_total": sum(x["extract_s"] for x in rows),
        **store.summary(), **disk_delta,
    }
    with open(run_dir / "requests.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return summary


def run_flexgen_engine(cfg: ExperimentConfig, run_dir: Path, verbose: bool) -> dict:
    from .engines.flexgen import run_flexgen
    warn_offload_location(cfg.store.offload_dir)
    root = prepare_offload_dir(cfg.store.offload_dir)
    disk = DiskCounter(root)
    started = _now()
    res = run_flexgen(cfg, str(root), str(run_dir / "flexgen.log"), verbose=verbose)
    return {"started": started, "finished": _now(), "engine": f"flexgen:{cfg.engine.model}",
            "policy": "flexgen" + ("_compress" if cfg.engine.flexgen_compress_cache else ""),
            "flexgen_percent": list(cfg.engine.flexgen_percent), **res, **disk.delta(do_sync=True)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--policy", help="preset name or spec, e.g. reuse_int4 or 'reuse@0.3/int8+zstd'")
    ap.add_argument("--engine", choices=["hf", "sim", "flexgen"])
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override config, e.g. --set store.ram_mb=512 --set workload.seed=3")
    ap.add_argument("--train-set", action="append", default=[], metavar="KEY=VALUE",
                    help="override the workload the reuse predictor is TRAINED on (shift test), "
                         "e.g. --train-set workload.mix.doc_qa=0.1 --train-set workload.turns.general=[4,9]")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, args.set)
    if args.policy:
        cfg.policy = args.policy
    if args.engine:
        cfg.engine.kind = args.engine
    run_id = args.run_id or f"{dt.datetime.now():%Y%m%d-%H%M%S}-{cfg.engine.kind}-{cfg.policy}".replace("/", "_")
    run_dir = Path(cfg.results_dir) / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2, default=list))

    if cfg.engine.kind == "flexgen":
        summary = run_flexgen_engine(cfg, run_dir, not args.quiet)
    else:
        summary = run_multiturn(cfg, run_dir, not args.quiet, args.train_set)
    summary = {"run_id": run_id, "config_name": cfg.name, **summary, "env": _env_info()}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    flat = {k: v for k, v in summary.items() if not isinstance(v, (dict, list))}
    flat["summary_path"] = str((run_dir / "summary.json").resolve())
    print("KVWEAR_RESULT " + json.dumps(flat, default=str), flush=True)


if __name__ == "__main__":
    main()
