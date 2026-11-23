# Roofline and occupancy (NVIDIA GeForce RTX 3050 Laptop GPU, sm_86, 16 SMs)

## Ceilings (measured by kvprof_kernels.cu)

| ceiling | measured | spec / theoretical |
|---|---|---|
| FP32 FMA throughput | 4852 GFLOP/s | 6144 GFLOP/s at the 1500 MHz base clock |
| DRAM read | 170.0 GB/s | 187.9 GB/s |
| DRAM copy (read+write) | 158.2 GB/s | |
| DRAM write | 169.7 GB/s | |
| torch tensor copy | 161.2 GB/s | |

Ridge point (measured): 28.5 FLOP/byte. Every kvwear kernel sits far left of it: they are all memory-bound, so the right metric is bandwidth, not FLOP/s.

## Kernels on the roofline

Bytes are compulsory DRAM traffic; % is of the measured DRAM read ceiling.

| kernel | time (us) | FLOP/byte | GFLOP/s | GB/s | % of BW ceiling |
|---|---|---|---|---|---|
| quantize int8 v0 naive | 2574.8 | 0.99 | 7 | 7.4 | 4% |
| quantize int4 v0 naive | 1762.2 | 1.56 | 14 | 9.1 | 5% |
| quantize int8 v1 warp+shuffle | 372.7 | 0.99 | 51 | 51.2 | 30% |
| quantize int4 v1 warp+shuffle | 421.9 | 1.56 | 60 | 38.2 | 22% |
| quantize int8 v2 half2 | 252.9 | 0.99 | 75 | 75.4 | 44% |
| quantize int4 v2 half2 | 305.2 | 1.56 | 82 | 52.8 | 31% |
| quantize int8 v3 register cache | 225.2 | 0.99 | 84 | 84.7 | 50% |
| quantize int4 v3 register cache | 313.9 | 1.56 | 80 | 51.4 | 30% |
| quantize int8 v4 sub-warp 128-bit | 178.2 | 0.99 | 106 | 107.0 | 63% |
| quantize int4 v4 sub-warp 128-bit | 211.9 | 1.56 | 119 | 76.1 | 45% |
| dequantize int8 (v8) | 184.7 | 0.33 | 34 | 103.3 | 61% |
| dequantize int4 (v8) | 141.3 | 0.78 | 89 | 114.1 | 67% |
| dequant+scatter int8 x16 blocks | 1997.3 | 0.33 | 50 | 152.8 | 90% |
| quantize int8 as PyTorch ops | 1972.7 | 0.99 | 10 | 9.7 | 6% |
| OPT-1.3B attn int8 v1 warp/token | 2725.9 | 1.97 | 100 | 50.8 | 30% |
| OPT-1.3B attn int8 v2 sub-warp/token | 1001.9 | 1.97 | 272 | 138.1 | 81% |
| OPT-1.3B attn fp16 (kvwear kernel) | 1856.5 | 1.02 | 147 | 144.6 | 85% |
| OPT-1.3B attn torch SDPA fp16 | 4094.5 | 1.02 | 67 | 65.6 | 39% |
| Qwen2.5-0.5B attn int8 v1 warp/token | 769.5 | 13.79 | 155 | 11.2 | 7% |
| Qwen2.5-0.5B attn int8 v2 sub-warp/token | 394.2 | 13.79 | 303 | 21.9 | 13% |
| Qwen2.5-0.5B attn int8 v3 GQA head packing | 290.2 | 13.79 | 411 | 29.8 | 18% |
| Qwen2.5-0.5B attn int8 v4 tensor cores (WMMA) | 128.5 | 13.79 | 928 | 67.3 | 40% |
| Qwen2.5-0.5B attn int8 v5 tensor cores (mma.sync) | 90.1 | 13.79 | 1324 | 96.0 | 56% |
| Qwen2.5-0.5B attn fp16 (kvwear kernel) | 428.5 | 7.11 | 278 | 39.1 | 23% |
| Qwen2.5-0.5B attn torch SDPA fp16 | 1061.3 | 1.02 | 112 | 110.7 | 65% |
| Llama-3-8B shape attn int8 v1 warp/token | 3319.3 | 7.82 | 163 | 20.8 | 12% |
| Llama-3-8B shape attn int8 v2 sub-warp/token | 1952.3 | 7.82 | 277 | 35.4 | 21% |
| Llama-3-8B shape attn int8 v3 GQA head packing | 1284.1 | 7.82 | 421 | 53.9 | 32% |
| Llama-3-8B shape attn int8 v4 tensor cores (WMMA) | 1199.1 | 7.82 | 451 | 57.7 | 34% |
| Llama-3-8B shape attn int8 v5 tensor cores (mma.sync) | 494.6 | 7.82 | 1094 | 139.9 | 82% |
| Llama-3-8B shape attn fp16 (kvwear kernel) | 2153.0 | 4.03 | 251 | 62.3 | 37% |
| Llama-3-8B shape attn torch SDPA fp16 | 8037.4 | 1.01 | 67 | 66.8 | 39% |
| cuBLAS SGEMM 4096^3 (FP32 cores) | 70646.8 | 682.67 | 1945 | 2.8 | 2% |
| cuBLAS HGEMM 4096^3 (tensor cores) | 14900.2 | 1365.33 | 9224 | 6.8 | 4% |

## Static resources and theoretical occupancy (cudaFuncGetAttributes, cudaOccupancy API)

Per SM: 1536 threads, 16 blocks, 65536 registers, 100 KiB shared memory (1024 B reserved per block).

| kernel | block | regs/thread | spill+stack B | static smem B | blocks/SM | warps/SM | occupancy | limiter |
|---|---|---|---|---|---|---|---|---|
| quant_int8 v0 naive | 256 | 28 | 0 | 0 | 6 | 48 | 100% | threads |
| quant_int8 v1 warp | 256 | 31 | 0 | 0 | 6 | 48 | 100% | threads |
| quant_int8 v2 warp_vec | 256 | 31 | 0 | 0 | 6 | 48 | 100% | threads |
| quant_int8 v3 warp_reg | 256 | 34 | 0 | 0 | 6 | 48 | 100% | threads |
| quant_int8 v4 subwarp<8> (g=64) | 256 | 26 | 0 | 0 | 6 | 48 | 100% | threads |
| quant_int8 v4 subwarp<16> (g=128) | 256 | 26 | 0 | 0 | 6 | 48 | 100% | threads |
| quant_int4 v0 naive | 256 | 28 | 0 | 0 | 6 | 48 | 100% | threads |
| quant_int4 v1 warp | 256 | 32 | 0 | 0 | 6 | 48 | 100% | threads |
| quant_int4 v2 warp_vec | 256 | 33 | 0 | 0 | 6 | 48 | 100% | threads |
| quant_int4 v3 warp_reg | 256 | 30 | 0 | 0 | 6 | 48 | 100% | threads |
| quant_int4 v4 subwarp<8> (g=64) | 256 | 32 | 0 | 0 | 6 | 48 | 100% | threads |
| dequant_int8 (half2) | 256 | 18 | 0 | 0 | 6 | 48 | 100% | threads |
| dequant_int4 (half2) | 256 | 20 | 0 | 0 | 6 | 48 | 100% | threads |
| dequant_v8 (8 values/thread) | 256 | 24 | 0 | 0 | 6 | 48 | 100% | threads |
| dequant_scatter (half2) | 256 | 34 | 0 | 0 | 6 | 48 | 100% | threads |
| dequant_scatter_v8 | 256 | 28 | 0 | 0 | 6 | 48 | 100% | threads |
| attn paged_int8 v1 <D=64> | 128 | 48 | 0 | 1056 | 10 | 40 | 83% | registers |
| attn paged_int8 v1 <D=128> | 128 | 40 | 0 | 2080 | 12 | 48 | 100% | threads |
| attn dense_fp16 <D=64> | 128 | 32 | 0 | 1056 | 12 | 48 | 100% | threads |
| attn dense_fp16 <D=128> | 128 | 36 | 0 | 2080 | 12 | 48 | 100% | threads |
| attn paged_int8 v2 <D=64> | 128 | 62 | 0 | 4224 | 8 | 32 | 67% | registers |
| attn paged_int8 v2 <D=128> | 128 | 56 | 0 | 4160 | 9 | 36 | 75% | registers |
| attn paged_int8 v3 <D=64,R=2> | 128 | 96 | 16 | 4224 | 5 | 20 | 42% | registers |
| attn paged_int8 v3 <D=64,R=4> | 128 | 127 | 0 | 4224 | 4 | 16 | 33% | registers |
| attn paged_int8 v3 <D=64,R=7> (Qwen2.5-0.5B) | 128 | 168 | 0 | 4224 | 3 | 12 | 25% | registers |
| attn paged_int8 v3 <D=64,R=8> | 128 | 202 | 0 | 4224 | 2 | 8 | 17% | registers |
| attn paged_int8 v3 <D=128,R=4> | 128 | 117 | 0 | 4160 | 4 | 16 | 33% | registers |
| attn paged_int8 v3 <D=128,R=8> | 128 | 200 | 0 | 4160 | 2 | 8 | 17% | registers |
| attn paged_int8 v4 WMMA <D=64,TILE=64,NW=8> | 256 | 66 | 0 | 31744 | 3 | 24 | 50% | registers |
| attn paged_int8 v4 WMMA <D=128,TILE=32,NW=8> | 256 | 116 | 0 | 33792 | 2 | 16 | 33% | registers |
| attn paged_int8 v5 mma.sync <D=64,g=64> | 128 | 96 | 0 | 16896 | 5 | 20 | 42% | registers |
| attn paged_int8 v5 mma.sync <D=128,g=128> | 128 | 168 | 0 | 33280 | 2 | 8 | 17% | shared mem |
| attn paged_int8 v5 mma.sync <D=128,g=64> | 128 | 168 | 0 | 33280 | 2 | 8 | 17% | shared mem |
| attn decode_combine <D=64> | 64 | 40 | 0 | 0 | 16 | 32 | 67% | blocks |
| peak_fp32_fma<16> (ceiling) | 256 | 37 | 0 | 0 | 6 | 48 | 100% | threads |
| stream_read (ceiling) | 256 | 44 | 0 | 0 | 5 | 40 | 83% | registers |
| stream_copy (ceiling) | 256 | 40 | 0 | 0 | 6 | 48 | 100% | threads |
| stream_write (ceiling) | 256 | 26 | 0 | 0 | 6 | 48 | 100% | threads |
