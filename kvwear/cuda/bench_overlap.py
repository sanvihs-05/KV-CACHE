"""CUDA streams + pinned memory: does KV encoding overlap with model compute?

    python -m kvwear.cuda.bench_overlap --out results/cuda

A stand-in "model" (fp16 matmuls, ~25 ms per step, the order of an OPT-1.3B prefill chunk) runs on the default
stream while a writer thread quantizes evicted KV blocks for the SSD. Three schedules:

  serial      wait for the model (synchronize), then encode on the GPU: the naive blocking version
  cpu-thread  encode on the CPU in a background thread (kvwear's CPU path)
  gpu-stream  encode on the GPU in a background thread on its own stream with pinned buffers
              (kvwear's CUDA path): copies and kernels interleave with the model's kernels

Every GPU op is bracketed by CUDA events, so the timeline plot shows real device time on two
lanes. The headline is the model's own throughput: how much the writer slows it down.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path

import numpy as np
import torch

from ..codecs import Codec


MODEL_REPS = 60


class Timeline:
    def __init__(self):
        self.base = torch.cuda.Event(enable_timing=True)
        self.base.record()
        self.spans = []                                   # (lane, label, start_event, end_event)
        self.lock = threading.Lock()

    def span(self, lane, label, stream=None):
        tl = self

        class _S:
            def __enter__(s):
                s.a = torch.cuda.Event(enable_timing=True)
                s.a.record(stream or torch.cuda.current_stream())
                return s

            def __exit__(s, *exc):
                b = torch.cuda.Event(enable_timing=True)
                b.record(stream or torch.cuda.current_stream())
                with tl.lock:
                    tl.spans.append((lane, label, s.a, b))
        return _S()

    def resolve(self):
        torch.cuda.synchronize()
        return [(lane, lab, self.base.elapsed_time(a), self.base.elapsed_time(b)) for lane, lab, a, b in self.spans]


def model_step(W, x, reps):
    for _ in range(reps):
        x = torch.relu(x @ W)
    return x


def run(schedule: str, steps: int, blocks_per_step: int, block: np.ndarray, tl: Timeline | None):
    torch.manual_seed(0)
    W = torch.randn(2048, 2048, dtype=torch.float16, device="cuda") / 45
    x = torch.randn(512, 2048, dtype=torch.float16, device="cuda")
    gpu_codec = Codec("int8", device="cuda")
    cpu_codec = Codec("int8")
    work: list = []
    cv = threading.Condition()
    done = {"stop": False, "encoded": 0}

    def writer():
        codec = gpu_codec if schedule == "gpu-stream" else cpu_codec
        while True:
            with cv:
                while not work and not done["stop"]:
                    cv.wait()
                if not work and done["stop"]:
                    return
                arr = work.pop(0)
            if schedule == "gpu-stream" and tl is not None:
                st = gpu_codec._gpu._state()["stream"]
                with tl.span("writer stream", "encode", st):
                    codec.encode(arr)
            else:
                codec.encode(arr)
            done["encoded"] += 1

    th = None
    if schedule in ("cpu-thread", "gpu-stream"):
        th = threading.Thread(target=writer, daemon=True)
        th.start()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for s in range(steps):
        if tl is not None:
            with tl.span("model (default stream)", "prefill"):
                x = model_step(W, x, MODEL_REPS)
        else:
            x = model_step(W, x, MODEL_REPS)
        if schedule == "serial":
            # the naive blocking version: wait for the model, then encode (nothing overlaps)
            torch.cuda.synchronize()
            for _ in range(blocks_per_step):
                if tl is not None:
                    with tl.span("model (default stream)", "encode"):
                        gpu_codec.encode(block)
                else:
                    gpu_codec.encode(block)
        else:
            with cv:
                work.extend([block] * blocks_per_step)
                cv.notify()
    torch.cuda.synchronize()
    model_s = time.perf_counter() - t0
    if th is not None:
        with cv:
            done["stop"] = True
            cv.notify()
        th.join()
    total_s = time.perf_counter() - t0
    return model_s, total_s


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results/cuda")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--blocks-per-step", type=int, default=2)
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    block = (np.random.default_rng(0).standard_normal((24, 2, 32, 64, 64), dtype=np.float32)).astype(np.float16)
    run("serial", 3, 1, block, None)                                     # warm-up + JIT
    run("gpu-stream", 3, 1, block, None)
    for _ in range(3):                                                   # let the GPU reach steady clocks
        run("none", args.steps, 0, block, None)
    base_s = min(run("none", args.steps, 0, block, None)[0] for _ in range(3))
    rows = [{"schedule": "model only (no KV writes)", "model_s": base_s, "total_s": base_s, "model_slowdown_pct": 0.0,
             "total_overhead_pct": 0.0}]
    timelines = {}
    for sch in ("serial", "cpu-thread", "gpu-stream"):
        m, t = min((run(sch, args.steps, args.blocks_per_step, block, None) for _ in range(3)), key=lambda r: r[1])
        rows.append({"schedule": sch, "model_s": m, "total_s": t, "model_slowdown_pct": 100 * (m / base_s - 1),
                     "total_overhead_pct": 100 * (t / base_s - 1)})
        tl = Timeline()
        run(sch, 6, args.blocks_per_step, block, tl)
        timelines[sch] = tl.resolve()
        print(rows[-1], flush=True)
    (out / "overlap.json").write_text(json.dumps({"rows": rows, "timelines": timelines}, indent=2))
    md = ["# Writer/model overlap (CUDA streams + pinned memory)\n",
          f"{args.steps} model steps; {args.blocks_per_step} OPT-1.3B blocks (12 MiB each) encoded to int8 per step.\n",
          "model time = until the last model step finishes; total = until every block is also encoded.\n",
          "| schedule | model time (s) | total incl. drain (s) | model slowdown | total vs model-only |",
          "|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['schedule']} | {r['model_s']:.3f} | {r['total_s']:.3f} | {r['model_slowdown_pct']:+.1f}% | "
                  f"{r['total_overhead_pct']:+.1f}% |")
    (out / "overlap.md").write_text("\n".join(md) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(timelines), 1, figsize=(11, 1.6 * len(timelines) + 0.6), sharex=True)
    col = {"prefill": "#2a7ab9", "encode": "#e08a1e"}
    for ax, (sch, spans) in zip(np.atleast_1d(axes), timelines.items()):
        lanes = sorted({s[0] for s in spans}, key=lambda x: (not x.startswith("model"), x))
        for lane, lab, a, b in spans:
            ax.barh(lanes.index(lane), b - a, left=a, color=col.get(lab, "#999"), height=0.6,
                    edgecolor="white", linewidth=0.8)
        ax.set_yticks(range(len(lanes)), lanes, fontsize=8)
        note = {"serial": "  (blocking: model waits for every encode)",
                "cpu-thread": "  (encode runs on the CPU: invisible to the GPU, but finishes long after the model)",
                "gpu-stream": "  (encode on its own CUDA stream overlaps the model's kernels)"}.get(sch, "")
        ax.set_title(sch + note, fontsize=9, loc="left")
        ax.set_xlabel("ms (GPU time, CUDA events)", fontsize=8)
    from matplotlib.patches import Patch
    np.atleast_1d(axes)[0].legend(handles=[Patch(color=c, label=k) for k, c in col.items()], fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "overlap_timeline.png", dpi=140)
    print((out / "overlap.md").read_text())


if __name__ == "__main__":
    main()
