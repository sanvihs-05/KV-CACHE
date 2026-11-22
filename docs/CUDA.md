# kvwear CUDA: GPU kernels for KV-cache offloading

This part of kvwear moves the KV-cache codec and the cache-loading path onto the GPU, and adds a
decode-attention kernel that reads quantized KV blocks directly. Everything is hand-written CUDA C
(`kvwear/cuda/*.cu`), compiled with `nvcc` through `torch.utils.cpp_extension`, and benchmarked
on an RTX 3050 Laptop GPU (Ampere, sm_86, 16 SMs, 4 GB, 128-bit GDDR6, **188 GB/s theoretical**,
177 GB/s measured device-to-device copy).

| Component | File | What it does | Headline |
|---|---|---|---|
| Quantize kernels (5 optimization stages) | `kvquant_kernels.cu` | fp16 → int8 / int4 KV blocks, bit-identical to the CPU codec | **177 GB/s** (= memcpy ceiling, 94% of peak), **23×** over naive, **16×** over PyTorch ops |
| Batched dequantize + scatter | `kvquant_kernels.cu` | all cached blocks of a request → attention cache in **one launch**, DMA pipelined with host copies | load path **8–25×** faster than CPU decode + concat + copy |
| GPU codec on its own stream | `__init__.py` (`GpuCodec`) | SSD writer thread encodes on a private CUDA stream with pinned buffers | **4×** faster than multi-threaded CPU encode; encoding fully hidden behind model compute (+2% vs +49% blocking) |
| Paged int8 decode attention (3 stages) | `kvattn_kernels.cu` | flash-decoding over int8 blocks via a block table, no dequantize pass | **1.95×** vs fp16 kernel, **4.0×** vs PyTorch SDPA (OPT, 32K tokens); **2.8×** vs SDPA on Qwen (GQA) |
| int8 GPU KV cache for generation | `kvcache_kernels.cu`, `int8_cache.py` | HF transformers `Cache` + attention hook: KV stays int8 on the GPU | Qwen2.5-0.5B decode **2.18× faster** at 31.7K context, KV memory halved, 95–100% next-token agreement |
| End to end (real model, same trace) | `store.codec_device=cuda` | GPU codec + loader inside kvwear's experiments | TTFT **−47% to −60%**, latency **−13% to −19%** for quantized policies on OPT-1.3B; identical SSD bytes |

All kernels are verified by `tests/test_cuda.py` (83 tests): byte-for-byte equality with the CPU
codec for every optimization stage and shape, exact equality of the GPU load path with the CPU
path, identical SSD files from the store, and attention within 2e-3 of an fp32 reference.

---

## 1. Where the GPU fits in kvwear

```
            prefill/decode on GPU (HF transformers)
                     │ new KV blocks                        ▲ attention cache
                     ▼                                      │
   host RAM tier (fp16 blocks, LRU) ──evict──► admission ──┐│
                                                           ▼│
   SSD writer thread ─ GpuCodec: pinned H2D → quantize kernel → D2H (own CUDA stream)
                     │                                      ▲
                     ▼                                      │ BlockLoader: pinned H2D of still-
   SSD (block files, int8/int4, unchanged format)  ─────────┘ quantized blocks → 1 dequant/scatter launch
                     │
                     └──► PagedInt8KV: blocks uploaded as-is, decode attention reads int8 directly
```

Turned on with one switch, `--set store.codec_device=cuda`. The on-disk format, every write count
and every hit/miss decision are identical to the CPU path (the RAM tier accounts EncodedBlocks at
their logical fp16 size), so all SSD-wear results remain valid; only time changes.

---

## 2. Quantization kernels: an optimization ladder

The block format (shared with the CPU codec): groups of `g` = 64 consecutive values along
head_dim; int8 stores one fp16 scale (absmax/127) per group, int4 one fp16 scale and min per group
and two values per byte. Quantization is a **memory-bound** operation (≈ 0.5 flop/byte), so the
goal is to approach the DRAM bandwidth ceiling.

| Stage | Idea | int8 GB/s | int4 GB/s |
|---|---|---|---|
| v0 naive | 1 thread per group, serial loop over 64 values | 7.7 | 9.6 |
| v1 warp per group | 32 lanes share a group; `__shfl_xor_sync` butterfly reduction for absmax / min / max; adjacent lanes read adjacent values (**coalescing**) | 88 | 62 |
| v2 vectorized | `half2` 4-byte loads, `char2` stores: half the memory instructions | 115 | 73 |
| v3 register cache | the group stays in registers between the reduction and the quantize pass: global memory read **once** | 128 | 80 |
| **v4 sub-warp** | `g/8` lanes per group (a warp handles 4 groups), **128-bit loads** (8 halves per lane), 8-byte int8 / 4-byte int4 stores, shuffles confined to the sub-warp | **177** | **170–175** |
| dequantize (8 values/thread) | 8-byte int8 / 4-byte int4 load, one 16-byte store of 8 halves | 170 | 168 |
| reference: `cudaMemcpy` D2D | read + write of the same bytes | 177 | |
| reference: the same math as PyTorch GPU ops | 6 separate elementwise/reduction kernels, fp32 temporaries | 10.8 | 7.5 |

(192 MiB of OPT-1.3B KV, CUDA-event timing, median of 20.)

![kernel bandwidth](../results/cuda/kernel_bandwidth.png)

Why each step helps:

* **v0 → v1 (11×)**: in v0, lane *i* reads address `i × 128 B`, so each warp-wide load touches 32
  different cache lines. In v1 a warp reads 64 contiguous bytes. Coalescing is the single biggest win.
* **v1 → v2**: same bytes, half the load/store instructions (`half2`).
* **v2 → v3**: the second pass re-read the group from L1/L2; keeping it in registers removes that.
* **v3 → v4**: v3's lanes still issue 4-byte loads and 1–2-byte stores (int4: 32 lanes × 1 byte =
  a 32-byte warp store, poor DRAM efficiency). v4 issues 16-byte loads and 4–8-byte stores, so a
  warp moves 512 B per instruction. This closes int4's gap (80 → 175 GB/s).

**Bit-exact with the CPU.** The GPU must produce exactly the CPU bytes, or SSD write counts and
quality numbers would silently change. The kernels therefore use IEEE division (`__fdiv_rn`, no
`--use_fast_math`), round-half-to-even (`rintf`, as `torch.round`/`np.rint`), quantize with the
fp32 scale before storing it as fp16 (as the CPU does), and are compiled with `--fmad=false`
(plus `__fmul_rn`/`__fadd_rn`) because a fused multiply-add rounds differently from the CPU's
separate multiply and add. Tests compare bytes for all five stages, odd shapes (g = 16, 32, 128, 256,
groups not multiple of 4) and fallbacks (v4 drops to v3 when alignment does not allow 16-byte access).

---

## 3. The load path: one copy, one launch

When a request reuses cached prefix blocks (from RAM or SSD), they must become contiguous
`[layers, 2, heads, tokens, head_dim]` K/V tensors on the GPU before prefill.

**CPU path (before):** decode each SSD block on the CPU (int8/int4 → fp16), `np.concatenate` the
blocks, copy the result from pageable memory to the GPU, then a `.contiguous()` copy per layer.

**CUDA path (`BlockLoader`):**
1. pack every block's bytes (fp16 for RAM hits, *still-quantized* payloads for SSD hits) into one
   **pinned** buffer (16-byte aligned offsets);
2. async host-to-device copies from pinned memory (DMA at 12.7 GB/s vs 7.5–9 GB/s from pageable
   memory), **pipelined**: each block's DMA is issued as soon as its bytes are in the pinned buffer, so
   the copy engine moves block *i* while the CPU packs block *i+1* (cost ≈ max(memcpy, DMA), not the sum);
3. **one** kernel launch, `dequant_scatter`, with `blockIdx.y` = KV block: each CUDA block reads
   its method (fp16 copy / int8 / int4) from a small table, so the branch is uniform per CUDA
   block (no warp divergence); each thread decodes 8 values (one 8/4/16-byte load) and writes them
   with one 16-byte store **directly at their token offset** in the final cache tensor; per-layer
   K/V are views of that tensor (no further copies). A CUDA event guards the reused pinned buffer.

Quantized blocks cross PCIe compressed (int8: 0.52×, int4: 0.28× of fp16), so the GPU path moves
fewer bytes *and* skips the CPU decode.

| 16 cached OPT-1.3B blocks (1024 tokens, 192 MiB fp16) | CPU path | CUDA path | speedup | bytes over PCIe |
|---|---|---|---|---|
| fp16 | 273 ms | 33 ms | **8.4×** | 192 MiB |
| int8 | 264 ms | 19 ms | **14×** | 99 MiB |
| int4 | 319 ms | 13 ms | **25×** | 54 MiB |

(First version, one bulk copy after packing and a 2-values-per-thread kernel: 56 / 26 / 18 ms;
pipelining the DMA and widening the kernel's memory accesses gave a further 1.4–1.8×.)

This is on the time-to-first-token path of every request with a cache hit.

![load path](../results/cuda/assembly_speedup.png)

---

## 4. Encode on the writer thread: streams and pinned memory

The SSD writer is a background thread. With `codec_device=cuda`, `GpuCodec` gives that thread
its **own CUDA stream** and grow-only **pinned** staging buffers: pageable→pinned memcpy, async
H2D, quantize kernel, async D2H, then one `stream.synchronize()`. Because the work is on a
non-default stream, its copies and kernels interleave with the model's kernels on the default
stream instead of serializing behind them.

`kvwear/cuda/bench_overlap.py`: 30 model steps of ~20 ms (fp16 matmuls) while 2 OPT-1.3B blocks
(12 MiB each) per step must be encoded to int8:

| schedule | time until model done | until all blocks encoded | vs model alone |
|---|---|---|---|
| model alone | 0.61 s | — | — |
| serial (synchronize, then encode on GPU) | 0.91 s | 0.91 s | **+49%** |
| CPU writer thread (kvwear's CPU path) | 0.62 s | 1.43 s | model +0.6%, but encoding finishes **+133%** later (the writer cannot keep up → backpressure) |
| **GPU writer on its own stream** | 0.62 s | 0.63 s | **+2%**: encoding is hidden |

![stream timeline](../results/cuda/overlap_timeline.png)

Per 12 MiB OPT-1.3B block (int8), measured step by step: pinned memcpy 1.1 ms, H2D 1.4 ms, kernel
0.4 ms, D2H 0.8 ms, copy-out 0.75 ms → **4.6 ms vs 19.2 ms** for the multi-threaded CPU encoder
(66 ms numpy). For a 768 KiB Qwen2.5-0.5B block the fixed cost of a GPU round trip (~0.5 ms)
exceeds the CPU encode time (0.43 ms), so `Codec` dispatches by size: blocks under 2 MiB are
encoded on the CPU even with `codec_device=cuda` (identical bytes either way).

| | pageable | pinned |
|---|---|---|
| host → device | 9.1 GB/s | **12.7 GB/s** |
| device → host | 7.6 GB/s | **12.7 GB/s** |

---

## 5. Paged int8 decode attention

During decoding each new token attends over the whole cached context: for every head,
`softmax(q·Kᵀ/√d)·V`. With one query token this is a memory-bound matrix-vector product, so the
time is set by how many KV bytes are read. kvwear's int8 block format already stores **one scale
per (layer, K/V, head, token) row**, which is exactly the granularity attention needs. So the
kernel reads KV blocks *in their SSD format*, through a block table (byte offset of each block in a
GPU buffer, as in vLLM's PagedAttention), and never materializes fp16 K/V:

* half the bytes per decode step versus fp16;
* twice the context in the same GPU memory;
* blocks arriving from the SSD can be appended to the table without any conversion.

**Algorithm: flash-decoding (split-K).** Kernel 1 launches (heads × splits) CUDA blocks; each
handles one contiguous chunk of tokens with an **online softmax** (running max *m*, normalizer *l*,
weighted V accumulator, rescaled by `exp(m_old − m_new)` as new tokens arrive), so scores are never
written to memory. Partial states of the warps are merged in **shared memory**; kernel 2 merges the
splits of each head with the same log-sum-exp rescaling. Tokens per split is chosen at run time so
the grid has ≥ 8 CUDA blocks per SM.

| Stage | Idea |
|---|---|
| v1 | one warp per token: each lane holds 2 of 64 dims, 5-step shuffle reduction per token |
| v2 | one **sub-warp of 8 lanes per token** (8 int8 values = one 8-byte load per lane): 4 tokens in flight per warp, 3-step reduction, per-sub-warp softmax states; uniform loop trip count so full-mask shuffles stay legal |
| v3 | v2 + **GQA head packing**: one CUDA block per *KV* head scores each loaded K/V row against all R = Hq/Hkv query heads held in registers (R is a template parameter so the per-head state stays in registers) |

One layer, one decode step (µs; lower is better):

| Context | Model | PyTorch SDPA fp16 | kvwear fp16 kernel | dequantize + SDPA | **kvwear int8 (best)** | int8 DRAM GB/s |
|---|---|---|---|---|---|---|
| 8K | OPT-1.3B (32 heads) | 795 | 411 | 1794 | **243** (v2) | 142 |
| 16K | OPT-1.3B | 1570 | 750 | 3331 | **429** (v2) | 161 |
| 32K | OPT-1.3B | 3245 | 1563 | 6681 | **803** (v2) | **172 (92% of peak)** |
| 16K | Qwen2.5-0.5B (14 q / 2 kv heads) | 459 | 290 | 1390 | **174** (v3) | — |
| 32K | Qwen2.5-0.5B | 731 | 344 | 2427 | **257** (v3) | — |

* At long contexts int8 is **1.95× faster than the same algorithm in fp16** (it reads half the bytes)
  and **4× faster than PyTorch SDPA** on OPT; on Qwen, GQA packing (v3) adds 1.2× over v2.
* "Dequantize + SDPA" (expand int8 to fp16 with the scatter kernel, then attend) is the slowest:
  fusing the dequantization into attention is what makes int8 KV pay off.
* v1 → v2 is the same lesson as the quantizer: v1 was latency-bound (2-byte loads, long dependent
  shuffle chain, 84 GB/s); v2's wider loads and more tokens in flight reach 172 GB/s.
* Accuracy: relative error vs an fp32 reference on the original fp16 KV is 8.5e-3 (the int8
  quantization itself); against the dequantized KV the kernel is within 2e-3 (tests).
* At 1–2K tokens all variants are dominated by launch overhead (~60–90 µs for two kernels), and on
  Qwen v3's grid (2 KV heads × splits) is too small below 8K tokens, so the default picks v3 only for
  GQA models at ≥ 8K tokens and v2 otherwise.

![decode attention](../results/cuda/attention_decode.png)

This kernel is benchmarked and tested standalone; wiring it into HF transformers' attention for
end-to-end generation is future work (§8).

---

## 6. End-to-end on the real model

Same trace, same policies, only `store.codec_device` differs (cpu vs cuda); 3 repetitions each,
order shuffled; OPT-1.3B with a 256 MB RAM tier and Qwen2.5-0.5B with 8–12K-token documents
(`linux/cuda_e2e.sh`, raw runs in `results/cuda/e2e/`). **SSD bytes written are identical in every
configuration**, so all wear results hold unchanged.

| Config | Policy | codec time in writer (s) cpu → cuda | load+read on TTFT path (s) cpu → cuda | **TTFT** | **mean latency** (noise) |
|---|---|---|---|---|---|
| OPT-1.3B | `reuse_int8` | 5.6 → 1.3 | 13.7 → 4.5 | **−47%** | **−12.6%** (±1.6) |
| OPT-1.3B | `spill_int8` | 7.6 → 1.9 | 16.0 → 4.8 | **−52%** | **−16.2%** (±1.8) |
| OPT-1.3B | `spill_int4` | 9.6 → 1.7 | 17.6 → 3.6 | **−60%** | **−18.8%** (±2.5) |
| OPT-1.3B | `write_through` (fp16) | 0.5 → 0.5 | 12.5 → 12.6 | +2% | 0% (±3) |
| Qwen2.5-0.5B, long docs | `reuse_int8` | 0.9 → 1.0 | 3.6 → 2.2 | **−14%** | 0% (±3) |
| Qwen2.5-0.5B, long docs | `spill_int4` | 1.7 → 0.9 | 5.0 → 1.7 | **−33%** | **−6.3%** (±2.8) |

* The win is on the load path (SSD hit → attention cache), which is on the time-to-first-token
  path: quantized blocks are copied to the GPU still compressed and expanded by one kernel.
* fp16 `write_through` has nothing to decode, so it is unchanged, as expected.
* Qwen2.5-0.5B blocks (768 KiB) are below the 2 MiB GPU-encode threshold, so encoding stays on
  the CPU there; its gains come from the load path only.
* Quantized policies were already the ones that cut SSD wear the most (64–80%, main README); on the
  GPU they are now also **faster** than the CPU baseline instead of costing decode time.

### Generation with the KV cache held on the GPU in int8

`kvwear/cuda/int8_cache.py` plugs the int8 path into HF transformers: `Int8GpuCache` (a
`transformers` `Cache`) keeps every layer's K/V on the GPU in kvwear's int8 format (new tokens are
quantized in place by `append_int8`, `kvcache_kernels.cu`), and an attention function registered
through `AttentionInterface` runs the paged int8 kernel for every decode step (prefill keeps
PyTorch SDPA). Qwen2.5-0.5B, prompt = this repository's own docs and code, 64 decode steps, the
int8 run teacher-forced with the fp16 run's tokens (`bench_generate.py`):

| context | fp16 cache + SDPA | **int8 GPU cache + kvwear kernel** | speedup | KV MB fp16 → int8 | next-token agreement |
|---|---|---|---|---|---|
| 1K | 23.2 tok/s | 24.6 tok/s | 1.06× | 13 → 7 | 95.3% |
| 8K | 23.3 | 26.5 | 1.13× | 97 → 50 | 96.9% |
| 16K | 19.3 | 26.0 | 1.35× | 193 → 99 | 95.3% |
| 24K | 15.3 | 25.9 | 1.70× | 289 → 149 | 100% |
| 31.7K | 12.4 | **27.0** | **2.18×** | 373 → 192 | 98.4% |

The fp16 path slows down as the context grows (decode attention reads more KV each step); the int8
path stays flat because its attention reads half the bytes with a kernel running near the DRAM
limit, so per-token time is dominated by the rest of the model. KV memory is halved (int8 + one fp16
scale per 64 values), and greedy predictions agree with fp16 on 95–100% of steps.

![generation](../results/cuda/generate.png)

---

## CUDA concepts demonstrated (where to look)

| Concept | Where | What it shows |
|---|---|---|
| Thread hierarchy, grid-stride loops | all kernels; `dequant_scatter`, `gather_int8_rows` | work decomposition independent of grid size |
| Memory coalescing | quantize v0 → v1 (11× faster) | adjacent lanes must touch adjacent addresses |
| Vectorized memory access (32/64/128-bit) | quantize v2, v4; dequantize v8; attention v2 | fewer, wider transactions: 80 → 175 GB/s for int4 |
| Warp-level primitives (`__shfl_xor_sync`) | all reductions (absmax, min/max, dot products) | reductions without shared memory or `__syncthreads` |
| Sub-warp cooperative groups | quantize v4 (g/8 lanes/group), attention v2 (8 lanes/token) | matching the parallel unit to the data unit |
| Registers as a cache / register pressure | quantize v3; attention v3 (`R` as a template parameter) | single-pass algorithms; spills when arrays are runtime-sized |
| Shared memory + `__syncthreads` | attention: merging per-warp softmax states | intra-block communication |
| Avoiding warp divergence | `dequant_scatter`: method chosen per CUDA block (`blockIdx.y`), uniform loop counts in attention | branch uniformity |
| Templates / compile-time specialization | quantize v4 (`LPG`), attention (`D`, `R`) | unrolling, register allocation |
| Split-K parallelism + online softmax (flash-decoding) | `kvattn_kernels.cu` | parallelism along the sequence for 1-token queries; numerically stable streaming softmax |
| Memory-bound analysis / roofline | `bench.py` (bytes moved ÷ time vs 188 GB/s peak and memcpy) | knowing when a kernel is done optimizing |
| Pinned (page-locked) memory, DMA | `PinnedBuffer`, PCIe benchmark | 12.7 vs 7.5–9 GB/s, prerequisite for async copies |
| CUDA streams, events, concurrency | `GpuCodec` (writer stream), `bench_overlap.py` timeline | +2% instead of +49% |
| Copy/compute pipelining | `BlockLoader` (per-block async DMA while the CPU packs the next) | 1.4–1.8× on the load path |
| Numerical reproducibility | `--fmad=false`, `__fdiv_rn`, `rintf`, `__fmul_rn`/`__fadd_rn` | GPU output byte-identical to the CPU reference |
| Interop with a framework | `torch.utils.cpp_extension` JIT build, current-stream-aware launchers, `Cache` subclass + `AttentionInterface` hook | custom kernels inside PyTorch / transformers generation |
| Testing GPU code | `tests/test_cuda.py` | bit-exact and tolerance-based checks, fallback paths, edge shapes |

---

## 7. Build, run, test

```bash
# once, in WSL: nvcc + CUDA runtime headers (no cuBLAS etc. needed)
sudo apt-get install -y --no-install-recommends cuda-nvcc-12-1 cuda-cudart-dev-12-1
pip install ninja

bash linux/cuda_env.sh python -m pytest -q tests/test_cuda.py     # 83 tests; first call builds (~40 s)
bash linux/cuda_env.sh python -m kvwear.cuda.bench --out results/cuda            # kernels, PCIe, load path
bash linux/cuda_env.sh python -m kvwear.cuda.bench_attention --out results/cuda  # int8 decode attention
bash linux/cuda_env.sh python -m kvwear.cuda.bench_overlap --out results/cuda    # stream overlap timeline
bash linux/cuda_env.sh bash linux/cuda_e2e.sh 3                                   # real model, cpu vs cuda codec
bash linux/cuda_env.sh python -m kvwear.cuda.bench_generate --out results/cuda   # generation, int8 GPU KV cache

# use it in any experiment
python -m kvwear.runner --config configs/default.yaml --policy reuse_int8 --set store.codec_device=cuda
```

`linux/cuda_env.sh` puts `/usr/local/cuda-12.1/bin` on PATH and activates the venv. The extension is
JIT-compiled for the local GPU's architecture (`TORCH_CUDA_ARCH_LIST` from the device), cached in
`~/.cache/torch_extensions`, and rebuilt automatically when a source file changes.

Files:

```
kvwear/cuda/
  kvquant_kernels.cu     quantize v0..v4, dequantize, batched dequantize+scatter, DRAM-bandwidth query
  kvattn_kernels.cu      paged int8 decode attention v1..v3, dense fp16 baseline, split combine
  kvquant_bindings.cpp   PyTorch bindings (current-stream aware, shape/alignment checks)
  kvcache_kernels.cu     append new K/V rows to a GPU int8 cache; gather (dequantize) for chunked prefill
  kvcache_bindings.cpp   bindings for the cache extension
  int8_cache.py          Int8GpuCache (transformers Cache) + int8 decode-attention hook
  __init__.py            JIT build, PinnedBuffer, GpuCodec (own stream), BlockLoader
  attention.py           PagedInt8KV (block table), dense baseline, fp32 reference
  bench.py, bench_attention.py, bench_overlap.py, bench_generate.py, compare_e2e.py
tests/test_cuda.py       83 tests (skipped automatically without a GPU / nvcc)
```

---

## 8. Limitations and next steps

* The int8 GPU cache is wired into Qwen2-style models (transformers `AttentionInterface`) for
  batch-1 decoding; batched decoding, OPT's attention class, and feeding SSD blocks straight into
  the GPU cache's block table (no fp16 round trip on SSD hits) are next.
* Encoding still copies pageable → pinned on the CPU (1.1 ms per 12 MiB block); reading SSD blocks
  straight into a pinned slab pool would remove that copy for the load path too.
* Kernel timings are on one laptop GPU whose clocks vary with temperature; all numbers are
  medians of repeated CUDA-event measurements after warm-up.
* WSL2 does not expose Nsight Compute counters; achieved bandwidth is computed from bytes moved and
  CUDA-event time, compared against the theoretical peak from `cudaDevAttrMemoryClockRate` and
  `cudaDevAttrGlobalMemoryBusWidth` and against a measured device-to-device copy.
