# CUDA codec benchmarks

Device: NVIDIA GeForce RTX 3050 Laptop GPU (sm_86, 16 SMs), theoretical DRAM bandwidth 188 GB/s (128-bit bus, 5.9 GHz).

## Kernels

| shape | op | method | time_ms | gbps |
|---|---|---|---|---|
| opt-1.3b block (64 tok, 12 MiB) | memcpy D2D (ceiling) | - | 0.155 | 162.369 |
| opt-1.3b block (64 tok, 12 MiB) | quantize v0 naive (thread/group) | int8 | 2.401 | 7.942 |
| opt-1.3b block (64 tok, 12 MiB) | quantize v1 warp/group + shuffle | int8 | 0.254 | 75.097 |
| opt-1.3b block (64 tok, 12 MiB) | quantize v2 + half2 vector I/O | int8 | 0.198 | 96.248 |
| opt-1.3b block (64 tok, 12 MiB) | quantize v3 + register cache (1 pass) | int8 | 0.170 | 111.866 |
| opt-1.3b block (64 tok, 12 MiB) | quantize v4 sub-warp groups + 128-bit I/O | int8 | 0.126 | 150.802 |
| opt-1.3b block (64 tok, 12 MiB) | dequantize | int8 | 0.122 | 155.849 |
| opt-1.3b block (64 tok, 12 MiB) | quantize torch ops on GPU (no custom kernel) | int8 | 1.907 | 10.002 |
| opt-1.3b block (64 tok, 12 MiB) | quantize v0 naive (thread/group) | int4 | 1.613 | 9.996 |
| opt-1.3b block (64 tok, 12 MiB) | quantize v1 warp/group + shuffle | int4 | 0.305 | 52.832 |
| opt-1.3b block (64 tok, 12 MiB) | quantize v2 + half2 vector I/O | int4 | 0.291 | 55.339 |
| opt-1.3b block (64 tok, 12 MiB) | quantize v3 + register cache (1 pass) | int4 | 0.273 | 59.077 |
| opt-1.3b block (64 tok, 12 MiB) | quantize v4 sub-warp groups + 128-bit I/O | int4 | 0.111 | 145.778 |
| opt-1.3b block (64 tok, 12 MiB) | dequantize | int4 | 0.118 | 136.904 |
| opt-1.3b block (64 tok, 12 MiB) | quantize torch ops on GPU (no custom kernel) | int4 | 2.392 | 6.740 |
| qwen2.5-0.5b block (64 tok, 768 KiB) | memcpy D2D (ceiling) | - | 0.083 | 18.865 |
| qwen2.5-0.5b block (64 tok, 768 KiB) | quantize v0 naive (thread/group) | int8 | 0.111 | 10.773 |
| qwen2.5-0.5b block (64 tok, 768 KiB) | quantize v1 warp/group + shuffle | int8 | 0.058 | 20.551 |
| qwen2.5-0.5b block (64 tok, 768 KiB) | quantize v2 + half2 vector I/O | int8 | 0.102 | 11.713 |
| qwen2.5-0.5b block (64 tok, 768 KiB) | quantize v3 + register cache (1 pass) | int8 | 0.036 | 32.789 |
| qwen2.5-0.5b block (64 tok, 768 KiB) | quantize v4 sub-warp groups + 128-bit I/O | int8 | 0.041 | 29.077 |
| qwen2.5-0.5b block (64 tok, 768 KiB) | dequantize | int8 | 0.039 | 30.234 |
| qwen2.5-0.5b block (64 tok, 768 KiB) | quantize torch ops on GPU (no custom kernel) | int8 | 0.276 | 4.312 |
| qwen2.5-0.5b block (64 tok, 768 KiB) | quantize v0 naive (thread/group) | int4 | 0.054 | 18.566 |
| qwen2.5-0.5b block (64 tok, 768 KiB) | quantize v1 warp/group + shuffle | int4 | 0.058 | 17.382 |
| qwen2.5-0.5b block (64 tok, 768 KiB) | quantize v2 + half2 vector I/O | int4 | 0.056 | 18.102 |
| qwen2.5-0.5b block (64 tok, 768 KiB) | quantize v3 + register cache (1 pass) | int4 | 0.052 | 19.235 |
| qwen2.5-0.5b block (64 tok, 768 KiB) | quantize v4 sub-warp groups + 128-bit I/O | int4 | 0.051 | 19.723 |
| qwen2.5-0.5b block (64 tok, 768 KiB) | dequantize | int4 | 0.043 | 23.238 |
| qwen2.5-0.5b block (64 tok, 768 KiB) | quantize torch ops on GPU (no custom kernel) | int4 | 0.417 | 2.418 |
| opt-1.3b x16 blocks (1024 tok, 192 MiB) | memcpy D2D (ceiling) | - | 2.278 | 176.766 |
| opt-1.3b x16 blocks (1024 tok, 192 MiB) | quantize v0 naive (thread/group) | int8 | 39.743 | 7.678 |
| opt-1.3b x16 blocks (1024 tok, 192 MiB) | quantize v1 warp/group + shuffle | int8 | 3.478 | 87.746 |
| opt-1.3b x16 blocks (1024 tok, 192 MiB) | quantize v2 + half2 vector I/O | int8 | 2.771 | 110.108 |
| opt-1.3b x16 blocks (1024 tok, 192 MiB) | quantize v3 + register cache (1 pass) | int8 | 2.373 | 128.583 |
| opt-1.3b x16 blocks (1024 tok, 192 MiB) | quantize v4 sub-warp groups + 128-bit I/O | int8 | 1.752 | 174.118 |
| opt-1.3b x16 blocks (1024 tok, 192 MiB) | dequantize | int8 | 1.798 | 169.743 |
| opt-1.3b x16 blocks (1024 tok, 192 MiB) | quantize torch ops on GPU (no custom kernel) | int8 | 28.377 | 10.753 |
| opt-1.3b x16 blocks (1024 tok, 192 MiB) | quantize v0 naive (thread/group) | int4 | 26.835 | 9.612 |
| opt-1.3b x16 blocks (1024 tok, 192 MiB) | quantize v1 warp/group + shuffle | int4 | 4.218 | 61.157 |
| opt-1.3b x16 blocks (1024 tok, 192 MiB) | quantize v2 + half2 vector I/O | int4 | 3.629 | 71.079 |
| opt-1.3b x16 blocks (1024 tok, 192 MiB) | quantize v3 + register cache (1 pass) | int4 | 3.305 | 78.037 |
| opt-1.3b x16 blocks (1024 tok, 192 MiB) | quantize v4 sub-warp groups + 128-bit I/O | int4 | 1.510 | 170.784 |
| opt-1.3b x16 blocks (1024 tok, 192 MiB) | dequantize | int4 | 1.533 | 168.234 |
| opt-1.3b x16 blocks (1024 tok, 192 MiB) | quantize torch ops on GPU (no custom kernel) | int4 | 34.520 | 7.472 |

## End-to-end encode / decode of one block (host to host)

| shape | op | method | impl | time_ms | mb_per_s |
|---|---|---|---|---|---|
| opt-1.3b block (64 tok, 12 MiB) | encode host->bytes | int8 | CPU numpy | 57.693 | 218.102 |
| opt-1.3b block (64 tok, 12 MiB) | encode host->bytes | int8 | CPU torch (multi-thread) | 19.181 | 656.012 |
| opt-1.3b block (64 tok, 12 MiB) | encode host->bytes | int8 | CUDA (pinned, own stream) | 4.616 | 2726.072 |
| opt-1.3b block (64 tok, 12 MiB) | decode bytes->fp16 | int8 | CPU numpy | 31.194 | 403.370 |
| opt-1.3b block (64 tok, 12 MiB) | decode bytes->fp16 | int8 | CPU torch (multi-thread) | 6.878 | 1829.553 |
| opt-1.3b block (64 tok, 12 MiB) | encode host->bytes | int4 | CPU numpy | 78.689 | 159.906 |
| opt-1.3b block (64 tok, 12 MiB) | encode host->bytes | int4 | CPU torch (multi-thread) | 23.678 | 531.423 |
| opt-1.3b block (64 tok, 12 MiB) | encode host->bytes | int4 | CUDA (pinned, own stream) | 15.729 | 799.995 |
| opt-1.3b block (64 tok, 12 MiB) | decode bytes->fp16 | int4 | CPU numpy | 45.480 | 276.672 |
| opt-1.3b block (64 tok, 12 MiB) | decode bytes->fp16 | int4 | CPU torch (multi-thread) | 10.275 | 1224.584 |
| qwen2.5-0.5b block (64 tok, 768 KiB) | encode host->bytes | int8 | CPU numpy | 2.223 | 353.806 |
| qwen2.5-0.5b block (64 tok, 768 KiB) | encode host->bytes | int8 | CPU torch (multi-thread) | 0.381 | 2065.918 |
| qwen2.5-0.5b block (64 tok, 768 KiB) | encode host->bytes | int8 | CUDA (pinned, own stream) | 0.635 | 1239.377 |
| qwen2.5-0.5b block (64 tok, 768 KiB) | decode bytes->fp16 | int8 | CPU numpy | 1.602 | 490.903 |
| qwen2.5-0.5b block (64 tok, 768 KiB) | decode bytes->fp16 | int8 | CPU torch (multi-thread) | 0.249 | 3154.498 |
| qwen2.5-0.5b block (64 tok, 768 KiB) | encode host->bytes | int4 | CPU numpy | 3.527 | 223.006 |
| qwen2.5-0.5b block (64 tok, 768 KiB) | encode host->bytes | int4 | CPU torch (multi-thread) | 0.821 | 958.068 |
| qwen2.5-0.5b block (64 tok, 768 KiB) | encode host->bytes | int4 | CUDA (pinned, own stream) | 0.451 | 1742.361 |
| qwen2.5-0.5b block (64 tok, 768 KiB) | decode bytes->fp16 | int4 | CPU numpy | 2.281 | 344.740 |
| qwen2.5-0.5b block (64 tok, 768 KiB) | decode bytes->fp16 | int4 | CPU torch (multi-thread) | 0.455 | 1726.830 |

## PCIe

| op | memory | mb | time_ms | gbps |
|---|---|---|---|---|
| H2D | pageable | 64 | 13.437 | 4.994 |
| D2H | pageable | 64 | 9.854 | 6.810 |
| H2D | pinned | 64 | 5.232 | 12.826 |
| D2H | pinned | 64 | 5.230 | 12.831 |

## Request cache assembly (load path)

| op | method | impl | time_ms | h2d_mb |
|---|---|---|---|---|
| assemble 16 blocks (1024 tok) | fp16 | CPU decode + concat + H2D | 273.349 | 192.000 |
| assemble 16 blocks (1024 tok) | fp16 | CUDA BlockLoader (pipelined H2D + 1 launch) | 32.736 | 192.000 |
| assemble 16 blocks (1024 tok) | int8 | CPU decode + concat + H2D | 264.420 | 192.000 |
| assemble 16 blocks (1024 tok) | int8 | CUDA BlockLoader (pipelined H2D + 1 launch) | 18.937 | 99.000 |
| assemble 16 blocks (1024 tok) | int4 | CPU decode + concat + H2D | 318.949 | 192.000 |
| assemble 16 blocks (1024 tok) | int4 | CUDA BlockLoader (pipelined H2D + 1 launch) | 12.952 | 54.000 |
