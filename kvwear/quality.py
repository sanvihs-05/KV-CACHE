"""How much does a lossy codec hurt the model? (The third axis of the trade-off.)

For sample prompts from the workload: compute the prefix KV in fp16, pass it through the
codec (encode -> decode), then score the next `--cont` tokens of the prompt with
(a) the exact cache and (b) the lossy cache. Reports per codec:

  top1_agree   share of positions where the most likely next token is unchanged
  kl           mean KL(exact || lossy) of the next-token distributions (nats)
  ppl_exact / ppl_lossy   perplexity of the actual continuation tokens

    python -m kvwear.quality --model facebook/opt-1.3b --codecs int8,int4 --samples 20
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .codecs import Codec, make_codec
from .config import EngineConfig, WorkloadConfig
from .workload import generate_workload


def _score(engine, kv: np.ndarray, ids: list[int], start: int):
    import torch
    cache = engine.build_cache([kv])
    logits, _ = engine.forward_prompt(ids, cache, start)
    return torch.log_softmax(logits[0].float(), dim=-1)           # [T, vocab]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="facebook/opt-1.3b")
    ap.add_argument("--codecs", default="int8,int4")
    ap.add_argument("--samples", type=int, default=20)
    ap.add_argument("--cont", type=int, default=64, help="continuation tokens scored after the prefix")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/quality.json")
    args = ap.parse_args(argv)

    import torch
    from .engines.hf import HFEngine, _cache_layers

    engine = HFEngine(EngineConfig(kind="hf", model=args.model, device=args.device))
    wl = generate_workload(WorkloadConfig(max_context_tokens=min(2000, engine.max_context - 8)), engine.tokenizer)
    reqs = sorted(wl.requests, key=lambda r: -len(r.prompt_ids))
    step = max(len(reqs) // args.samples, 1)
    samples = [r for r in reqs[::step] if len(r.prompt_ids) > args.cont + 64][:args.samples]

    codecs = [make_codec(c) for c in args.codecs.split(",")]
    acc = {c.name: {"top1": [], "kl": [], "nll_exact": [], "nll_lossy": []} for c in codecs}
    with torch.inference_mode():
        for i, r in enumerate(samples):
            ids = r.prompt_ids
            p = len(ids) - args.cont
            _, cache = engine.forward_prompt(ids[:p], engine.build_cache([]), 0)
            layers = _cache_layers(cache)
            kv = torch.stack([torch.stack([k[0], v[0]]) for k, v in layers]).half().cpu().numpy()
            del cache
            ref = _score(engine, kv, ids, p)
            target = torch.tensor(ids[p + 1:], device=ref.device)
            for c in codecs:
                lossy_kv = Codec.decode(bytearray(c.encode(kv)))
                out = _score(engine, lossy_kv, ids, p)
                a = acc[c.name]
                a["top1"].append((ref.argmax(-1) == out.argmax(-1)).float().mean().item())
                a["kl"].append((ref.exp() * (ref - out)).sum(-1).mean().item())
                a["nll_exact"].append(-ref[:-1].gather(1, target[:, None]).mean().item())
                a["nll_lossy"].append(-out[:-1].gather(1, target[:, None]).mean().item())
            print(f"[{i + 1}/{len(samples)}] prefix {p} tokens done", flush=True)

    res = {}
    for name, a in acc.items():
        res[name] = {
            "top1_agree": float(np.mean(a["top1"])), "kl": float(np.mean(a["kl"])),
            "ppl_exact": float(np.exp(np.mean(a["nll_exact"]))), "ppl_lossy": float(np.exp(np.mean(a["nll_lossy"]))),
            "samples": len(a["top1"]),
        }
        print(f"{name:10s} top-1 agreement {res[name]['top1_agree']:.4f}  KL {res[name]['kl']:.5f}  "
              f"ppl {res[name]['ppl_exact']:.3f} -> {res[name]['ppl_lossy']:.3f}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"model": args.model, "results": res}, indent=2))


if __name__ == "__main__":
    main()
