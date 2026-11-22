"""End-to-end generation with the KV cache held on the GPU in int8 (kvwear format) vs fp16.

    python -m kvwear.cuda.bench_generate --out results/cuda

Model: Qwen2.5-0.5B-Instruct (fp16 weights). Prompt: real text (this repository's docs and source
code) truncated to each context length. For every length:
  fp16   transformers DynamicCache + PyTorch SDPA (the stock path)
  int8   Int8GpuCache (int8 K/V on the GPU) + kvwear paged int8 decode kernel
Both prefill the same prompt, then run `--new` decode steps. The int8 run is teacher-forced with
the fp16 run's tokens, so `agreement` = fraction of steps where both predict the same next token.
Reported: decode tokens/s, prefill time, KV-cache bytes, peak GPU memory, agreement.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .int8_cache import Int8GpuCache, use_int8_attention

ROOT = Path(__file__).resolve().parents[2]


def corpus_ids(tok, max_tokens: int) -> list[int]:
    files = [ROOT / "README.md", ROOT / "docs" / "CUDA.md"] + sorted((ROOT / "kvwear").rglob("*.py")) \
        + sorted((ROOT / "kvwear" / "cuda").glob("*.cu")) + sorted((ROOT / "windows").glob("*.ps*1"))
    text = "\n\n".join(p.read_text(encoding="utf-8", errors="ignore") for p in files)
    ids = tok(text, add_special_tokens=False)["input_ids"]
    while len(ids) < max_tokens:
        ids = ids + ids
    return ids[:max_tokens]


@torch.inference_mode()
def run_fp16(model, ids, new):
    from transformers import DynamicCache
    cache = DynamicCache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    out = model(input_ids=ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
    nxt = out.logits[:, -1].argmax(-1, keepdim=True)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    toks = [nxt.item()]
    for _ in range(new - 1):
        out = model(input_ids=nxt, past_key_values=cache, use_cache=True)
        nxt = out.logits[:, -1].argmax(-1, keepdim=True)
        toks.append(nxt.item())                          # .item() syncs: honest per-token latency
    t2 = time.perf_counter()
    kv = sum(k.numel() * k.element_size() + v.numel() * v.element_size()
             for k, v in zip(cache.key_cache, cache.value_cache))
    return {"prefill_s": t1 - t0, "decode_s": t2 - t1, "tokens": toks, "kv_bytes": kv,
            "peak_mem": torch.cuda.max_memory_allocated()}


@torch.inference_mode()
def run_int8(model, ids, new, ref_tokens):
    cache = Int8GpuCache.for_model(model, max_tokens=ids.shape[1] + new + 8)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    # prefill under the stock SDPA path (is_causal, no dense mask); K/V still go into the int8 cache
    out = model(input_ids=ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
    pred = [out.logits[:, -1].argmax(-1).item()]
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    with use_int8_attention(model, cache):
        for i in range(new - 1):
            x = torch.tensor([[ref_tokens[i]]], device=model.device)       # teacher forcing
            out = model(input_ids=x, past_key_values=cache, use_cache=True)
            pred.append(out.logits[:, -1].argmax(-1).item())
    t2 = time.perf_counter()
    agree = sum(int(a == b) for a, b in zip(pred, ref_tokens)) / len(ref_tokens)
    return {"prefill_s": t1 - t0, "decode_s": t2 - t1, "tokens": pred, "kv_bytes": cache.used_bytes(),
            "kv_alloc": cache.nbytes, "peak_mem": torch.cuda.max_memory_allocated(), "agreement": agree}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--lengths", default="1024,4096,8192,16384,24576,31744")
    ap.add_argument("--new", type=int, default=64)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--out", default="results/cuda")
    a = ap.parse_args(argv)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float16, attn_implementation="sdpa").cuda().eval()
    lengths = [int(x) for x in a.lengths.split(",")]
    all_ids = corpus_ids(tok, max(lengths))
    # warm-up (kernels, extension build, cuBLAS handles)
    w = torch.tensor([all_ids[:512]], device="cuda")
    r = run_fp16(model, w, 8)
    run_int8(model, w, 8, r["tokens"])
    rows = []
    for L in lengths:
        ids = torch.tensor([all_ids[:L]], device="cuda")
        best = {}
        for rep in range(a.reps):
            f = run_fp16(model, ids, a.new)
            q = run_int8(model, ids, a.new, f["tokens"])
            for k, res in (("fp16", f), ("int8", q)):
                if k not in best or res["decode_s"] < best[k]["decode_s"]:
                    best[k] = res
        f, q = best["fp16"], best["int8"]
        row = {"context": L, "new_tokens": a.new,
               "fp16_decode_tok_s": (a.new - 1) / f["decode_s"], "int8_decode_tok_s": (a.new - 1) / q["decode_s"],
               "speedup": f["decode_s"] / q["decode_s"],
               "fp16_prefill_s": f["prefill_s"], "int8_prefill_s": q["prefill_s"],
               "fp16_kv_mb": f["kv_bytes"] / 2**20, "int8_kv_mb": q["kv_bytes"] / 2**20,
               "fp16_peak_mb": f["peak_mem"] / 2**20, "int8_peak_mb": q["peak_mem"] / 2**20,
               "next_token_agreement": q["agreement"]}
        rows.append(row)
        print({k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()}, flush=True)
        del ids
        torch.cuda.empty_cache()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "generate.json").write_text(json.dumps({"model": a.model, "rows": rows}, indent=2))
    md = [f"# Generation with an int8 GPU KV cache ({a.model}, {a.new} decode steps, best of {a.reps})\n",
          "| context | fp16 tok/s | int8 tok/s | speedup | fp16 KV MB | int8 KV MB | peak GPU MB fp16 / int8 | next-token agreement |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['context']} | {r['fp16_decode_tok_s']:.1f} | {r['int8_decode_tok_s']:.1f} | {r['speedup']:.2f}x | "
                  f"{r['fp16_kv_mb']:.0f} | {r['int8_kv_mb']:.0f} | {r['fp16_peak_mb']:.0f} / {r['int8_peak_mb']:.0f} | "
                  f"{100 * r['next_token_agreement']:.1f}% |")
    (out / "generate.md").write_text("\n".join(md) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    xs = [r["context"] for r in rows]
    ax[0].plot(xs, [r["fp16_decode_tok_s"] for r in rows], "o-", color="#9aa5b1", label="fp16 cache + SDPA")
    ax[0].plot(xs, [r["int8_decode_tok_s"] for r in rows], "o-", color="#1f5f9e", label="int8 GPU cache + kvwear kernel")
    ax[0].set_xlabel("context tokens"); ax[0].set_ylabel("decode tokens / s"); ax[0].legend(fontsize=8)
    ax[0].set_title("Decode throughput, Qwen2.5-0.5B, RTX 3050 Laptop"); ax[0].grid(alpha=.3)
    ax[1].plot(xs, [r["fp16_kv_mb"] for r in rows], "o-", color="#9aa5b1", label="fp16 KV")
    ax[1].plot(xs, [r["int8_kv_mb"] for r in rows], "o-", color="#1f5f9e", label="int8 KV (+scales)")
    ax[1].set_xlabel("context tokens"); ax[1].set_ylabel("KV cache MB"); ax[1].legend(fontsize=8)
    ax[1].set_title("KV-cache memory"); ax[1].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(out / "generate.png", dpi=140)
    print((out / "generate.md").read_text())


if __name__ == "__main__":
    main()
