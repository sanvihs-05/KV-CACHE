# Decode attention over int8 KV (one layer, one decode step)

| model | tokens | impl | time (us) | KV MB read | GB/s | rel. error vs fp32 |
|---|---|---|---|---|---|---|
| qwen2.5-0.5b (Hq 14, Hkv 2) | 1024 | torch SDPA fp16 | 103.8 | 0.5 | 5 | 2.78e-04 |
| qwen2.5-0.5b (Hq 14, Hkv 2) | 1024 | kvwear dense fp16 kernel | 27.5 | 0.5 | 19 | 2.04e-04 |
| qwen2.5-0.5b (Hq 14, Hkv 2) | 1024 | dequantize + SDPA (unfused int8) | 171.5 | 0.3 | 2 | 8.67e-03 |
| qwen2.5-0.5b (Hq 14, Hkv 2) | 1024 | kvwear paged int8 v1 (warp/token) | 74.2 | 0.3 | 4 | 8.66e-03 |
| qwen2.5-0.5b (Hq 14, Hkv 2) | 1024 | kvwear paged int8 v2 (sub-warp/token) | 56.2 | 0.3 | 5 | 8.66e-03 |
| qwen2.5-0.5b (Hq 14, Hkv 2) | 1024 | kvwear paged int8 v3 (+ GQA head packing) | 57.1 | 0.3 | 5 | 8.66e-03 |
| qwen2.5-0.5b (Hq 14, Hkv 2) | 1024 | kvwear paged int8 v4 (tensor cores, WMMA) | 57.4 | 0.3 | 5 | 8.68e-03 |
| qwen2.5-0.5b (Hq 14, Hkv 2) | 1024 | kvwear paged int8 v5 (tensor cores, mma.sync) | 51.8 | 0.3 | 5 | 8.66e-03 |
| qwen2.5-0.5b (Hq 14, Hkv 2) | 2048 | torch SDPA fp16 | 123.2 | 1.0 | 9 | 2.67e-04 |
| qwen2.5-0.5b (Hq 14, Hkv 2) | 2048 | kvwear dense fp16 kernel | 46.7 | 1.0 | 22 | 2.11e-04 |
| qwen2.5-0.5b (Hq 14, Hkv 2) | 2048 | dequantize + SDPA (unfused int8) | 204.2 | 0.5 | 3 | 7.92e-03 |
| qwen2.5-0.5b (Hq 14, Hkv 2) | 2048 | kvwear paged int8 v1 (warp/token) | 94.2 | 0.5 | 6 | 7.91e-03 |
| qwen2.5-0.5b (Hq 14, Hkv 2) | 2048 | kvwear paged int8 v2 (sub-warp/token) | 64.0 | 0.5 | 8 | 7.91e-03 |
| qwen2.5-0.5b (Hq 14, Hkv 2) | 2048 | kvwear paged int8 v3 (+ GQA head packing) | 76.3 | 0.5 | 7 | 7.91e-03 |
| qwen2.5-0.5b (Hq 14, Hkv 2) | 2048 | kvwear paged int8 v4 (tensor cores, WMMA) | 52.9 | 0.5 | 10 | 7.93e-03 |
| qwen2.5-0.5b (Hq 14, Hkv 2) | 2048 | kvwear paged int8 v5 (tensor cores, mma.sync) | 66.6 | 0.5 | 8 | 7.90e-03 |
| qwen2.5-0.5b (Hq 14, Hkv 2) | 4096 | torch SDPA fp16 | 147.5 | 2.0 | 14 | 2.81e-04 |
| qwen2.5-0.5b (Hq 14, Hkv 2) | 4096 | kvwear dense fp16 kernel | 145.4 | 2.0 | 14 | 2.06e-04 |
| qwen2.5-0.5b (Hq 14, Hkv 2) | 4096 | dequantize + SDPA (unfused int8) | 350.2 | 1.0 | 3 | 8.42e-03 |
| qwen2.5-0.5b (Hq 14, Hkv 2) | 4096 | kvwear paged int8 v1 (warp/token) | 173.6 | 1.0 | 6 | 8.43e-03 |
| qwen2.5-0.5b (Hq 14, Hkv 2) | 4096 | kvwear paged int8 v2 (sub-warp/token) | 71.7 | 1.0 | 15 | 8.43e-03 |
| qwen2.5-0.5b (Hq 14, Hkv 2) | 4096 | kvwear paged int8 v3 (+ GQA head packing) | 110.1 | 1.0 | 10 | 8.43e-03 |
| qwen2.5-0.5b (Hq 14, Hkv 2) | 4096 | kvwear paged int8 v4 (tensor cores, WMMA) | 71.6 | 1.0 | 15 | 8.45e-03 |
| qwen2.5-0.5b (Hq 14, Hkv 2) | 4096 | kvwear paged int8 v5 (tensor cores, mma.sync) | 92.5 | 1.0 | 12 | 8.42e-03 |
| qwen2.5-0.5b (Hq 14, Hkv 2) | 8192 | torch SDPA fp16 | 268.3 | 4.0 | 16 | 2.69e-04 |
| qwen2.5-0.5b (Hq 14, Hkv 2) | 8192 | kvwear dense fp16 kernel | 134.1 | 4.0 | 31 | 1.96e-04 |
| qwen2.5-0.5b (Hq 14, Hkv 2) | 8192 | dequantize + SDPA (unfused int8) | 868.4 | 2.1 | 2 | 8.18e-03 |
| qwen2.5-0.5b (Hq 14, Hkv 2) | 8192 | kvwear paged int8 v1 (warp/token) | 294.9 | 2.1 | 7 | 8.17e-03 |
| qwen2.5-0.5b (Hq 14, Hkv 2) | 8192 | kvwear paged int8 v2 (sub-warp/token) | 130.0 | 2.1 | 17 | 8.17e-03 |
| qwen2.5-0.5b (Hq 14, Hkv 2) | 8192 | kvwear paged int8 v3 (+ GQA head packing) | 124.3 | 2.1 | 17 | 8.17e-03 |
| qwen2.5-0.5b (Hq 14, Hkv 2) | 8192 | kvwear paged int8 v4 (tensor cores, WMMA) | 114.2 | 2.1 | 19 | 8.17e-03 |
| qwen2.5-0.5b (Hq 14, Hkv 2) | 8192 | kvwear paged int8 v5 (tensor cores, mma.sync) | 95.2 | 2.1 | 23 | 8.17e-03 |
| qwen2.5-0.5b (Hq 14, Hkv 2) | 16384 | torch SDPA fp16 | 450.4 | 8.0 | 19 | 2.74e-04 |
| qwen2.5-0.5b (Hq 14, Hkv 2) | 16384 | kvwear dense fp16 kernel | 281.6 | 8.0 | 30 | 2.06e-04 |
| qwen2.5-0.5b (Hq 14, Hkv 2) | 16384 | dequantize + SDPA (unfused int8) | 1676.8 | 4.1 | 3 | 7.99e-03 |
| qwen2.5-0.5b (Hq 14, Hkv 2) | 16384 | kvwear paged int8 v1 (warp/token) | 524.3 | 4.1 | 8 | 7.99e-03 |
| qwen2.5-0.5b (Hq 14, Hkv 2) | 16384 | kvwear paged int8 v2 (sub-warp/token) | 201.2 | 4.1 | 21 | 7.99e-03 |
| qwen2.5-0.5b (Hq 14, Hkv 2) | 16384 | kvwear paged int8 v3 (+ GQA head packing) | 179.6 | 4.1 | 24 | 7.99e-03 |
| qwen2.5-0.5b (Hq 14, Hkv 2) | 16384 | kvwear paged int8 v4 (tensor cores, WMMA) | 113.6 | 4.1 | 38 | 7.99e-03 |
| qwen2.5-0.5b (Hq 14, Hkv 2) | 16384 | kvwear paged int8 v5 (tensor cores, mma.sync) | 90.0 | 4.1 | 48 | 8.00e-03 |
| qwen2.5-0.5b (Hq 14, Hkv 2) | 32768 | torch SDPA fp16 | 1061.9 | 16.0 | 16 | 2.87e-04 |
| qwen2.5-0.5b (Hq 14, Hkv 2) | 32768 | kvwear dense fp16 kernel | 494.6 | 16.0 | 34 | 2.06e-04 |
| qwen2.5-0.5b (Hq 14, Hkv 2) | 32768 | dequantize + SDPA (unfused int8) | 3482.0 | 8.2 | 2 | 8.64e-03 |
| qwen2.5-0.5b (Hq 14, Hkv 2) | 32768 | kvwear paged int8 v1 (warp/token) | 1005.6 | 8.2 | 9 | 8.62e-03 |
| qwen2.5-0.5b (Hq 14, Hkv 2) | 32768 | kvwear paged int8 v2 (sub-warp/token) | 386.0 | 8.2 | 22 | 8.62e-03 |
| qwen2.5-0.5b (Hq 14, Hkv 2) | 32768 | kvwear paged int8 v3 (+ GQA head packing) | 276.4 | 8.2 | 31 | 8.62e-03 |
| qwen2.5-0.5b (Hq 14, Hkv 2) | 32768 | kvwear paged int8 v4 (tensor cores, WMMA) | 130.0 | 8.2 | 67 | 8.63e-03 |
| qwen2.5-0.5b (Hq 14, Hkv 2) | 32768 | kvwear paged int8 v5 (tensor cores, mma.sync) | 115.7 | 8.2 | 75 | 8.61e-03 |
| opt-1.3b (Hq 32, Hkv 32) | 1024 | torch SDPA fp16 | 103.4 | 8.0 | 81 | 2.79e-04 |
| opt-1.3b (Hq 32, Hkv 32) | 1024 | kvwear dense fp16 kernel | 70.6 | 8.0 | 119 | 2.04e-04 |
| opt-1.3b (Hq 32, Hkv 32) | 1024 | dequantize + SDPA (unfused int8) | 203.8 | 4.1 | 21 | 8.43e-03 |
| opt-1.3b (Hq 32, Hkv 32) | 1024 | kvwear paged int8 v1 (warp/token) | 82.3 | 4.1 | 53 | 8.42e-03 |
| opt-1.3b (Hq 32, Hkv 32) | 1024 | kvwear paged int8 v2 (sub-warp/token) | 82.9 | 4.1 | 52 | 8.42e-03 |
| opt-1.3b (Hq 32, Hkv 32) | 2048 | torch SDPA fp16 | 218.5 | 16.0 | 77 | 2.85e-04 |
| opt-1.3b (Hq 32, Hkv 32) | 2048 | kvwear dense fp16 kernel | 121.3 | 16.0 | 138 | 2.07e-04 |
| opt-1.3b (Hq 32, Hkv 32) | 2048 | dequantize + SDPA (unfused int8) | 418.1 | 8.2 | 21 | 8.34e-03 |
| opt-1.3b (Hq 32, Hkv 32) | 2048 | kvwear paged int8 v1 (warp/token) | 215.0 | 8.2 | 40 | 8.32e-03 |
| opt-1.3b (Hq 32, Hkv 32) | 2048 | kvwear paged int8 v2 (sub-warp/token) | 71.2 | 8.2 | 122 | 8.32e-03 |
| opt-1.3b (Hq 32, Hkv 32) | 4096 | torch SDPA fp16 | 339.5 | 32.0 | 99 | 2.81e-04 |
| opt-1.3b (Hq 32, Hkv 32) | 4096 | kvwear dense fp16 kernel | 216.6 | 32.0 | 155 | 2.05e-04 |
| opt-1.3b (Hq 32, Hkv 32) | 4096 | dequantize + SDPA (unfused int8) | 827.3 | 16.5 | 21 | 8.70e-03 |
| opt-1.3b (Hq 32, Hkv 32) | 4096 | kvwear paged int8 v1 (warp/token) | 354.8 | 16.5 | 49 | 8.69e-03 |
| opt-1.3b (Hq 32, Hkv 32) | 4096 | kvwear paged int8 v2 (sub-warp/token) | 165.9 | 16.5 | 104 | 8.69e-03 |
| opt-1.3b (Hq 32, Hkv 32) | 8192 | torch SDPA fp16 | 785.4 | 64.0 | 85 | 2.92e-04 |
| opt-1.3b (Hq 32, Hkv 32) | 8192 | kvwear dense fp16 kernel | 438.8 | 64.0 | 153 | 2.12e-04 |
| opt-1.3b (Hq 32, Hkv 32) | 8192 | dequantize + SDPA (unfused int8) | 1665.0 | 33.0 | 21 | 8.54e-03 |
| opt-1.3b (Hq 32, Hkv 32) | 8192 | kvwear paged int8 v1 (warp/token) | 687.1 | 33.0 | 50 | 8.53e-03 |
| opt-1.3b (Hq 32, Hkv 32) | 8192 | kvwear paged int8 v2 (sub-warp/token) | 321.5 | 33.0 | 108 | 8.53e-03 |
| opt-1.3b (Hq 32, Hkv 32) | 16384 | torch SDPA fp16 | 1777.7 | 128.0 | 76 | 2.92e-04 |
| opt-1.3b (Hq 32, Hkv 32) | 16384 | kvwear dense fp16 kernel | 815.1 | 128.0 | 165 | 2.08e-04 |
| opt-1.3b (Hq 32, Hkv 32) | 16384 | dequantize + SDPA (unfused int8) | 3497.0 | 66.0 | 20 | 8.57e-03 |
| opt-1.3b (Hq 32, Hkv 32) | 16384 | kvwear paged int8 v1 (warp/token) | 1397.2 | 66.0 | 50 | 8.55e-03 |
| opt-1.3b (Hq 32, Hkv 32) | 16384 | kvwear paged int8 v2 (sub-warp/token) | 573.4 | 66.0 | 121 | 8.55e-03 |
| opt-1.3b (Hq 32, Hkv 32) | 32768 | torch SDPA fp16 | 3578.4 | 256.0 | 75 | 2.90e-04 |
| opt-1.3b (Hq 32, Hkv 32) | 32768 | kvwear dense fp16 kernel | 1830.4 | 256.0 | 147 | 2.03e-04 |
| opt-1.3b (Hq 32, Hkv 32) | 32768 | dequantize + SDPA (unfused int8) | 7100.9 | 132.0 | 19 | 8.64e-03 |
| opt-1.3b (Hq 32, Hkv 32) | 32768 | kvwear paged int8 v1 (warp/token) | 3002.3 | 132.0 | 46 | 8.62e-03 |
| opt-1.3b (Hq 32, Hkv 32) | 32768 | kvwear paged int8 v2 (sub-warp/token) | 1310.7 | 132.0 | 106 | 8.62e-03 |
| llama-3-8b shape (Hq 32, Hkv 8, D 128) | 1024 | torch SDPA fp16 | 253.4 | 4.0 | 17 | 2.76e-04 |
| llama-3-8b shape (Hq 32, Hkv 8, D 128) | 1024 | kvwear dense fp16 kernel | 54.3 | 4.0 | 77 | 2.08e-04 |
| llama-3-8b shape (Hq 32, Hkv 8, D 128) | 1024 | dequantize + SDPA (unfused int8) | 456.0 | 2.1 | 5 | 8.30e-03 |
| llama-3-8b shape (Hq 32, Hkv 8, D 128) | 1024 | kvwear paged int8 v1 (warp/token) | 155.1 | 2.1 | 14 | 8.31e-03 |
| llama-3-8b shape (Hq 32, Hkv 8, D 128) | 1024 | kvwear paged int8 v2 (sub-warp/token) | 135.2 | 2.1 | 16 | 8.31e-03 |
| llama-3-8b shape (Hq 32, Hkv 8, D 128) | 1024 | kvwear paged int8 v3 (+ GQA head packing) | 74.1 | 2.1 | 29 | 8.31e-03 |
| llama-3-8b shape (Hq 32, Hkv 8, D 128) | 1024 | kvwear paged int8 v4 (tensor cores, WMMA) | 98.3 | 2.1 | 22 | 8.31e-03 |
| llama-3-8b shape (Hq 32, Hkv 8, D 128) | 1024 | kvwear paged int8 v5 (tensor cores, mma.sync) | 131.6 | 2.1 | 16 | 8.31e-03 |
| llama-3-8b shape (Hq 32, Hkv 8, D 128) | 2048 | torch SDPA fp16 | 340.5 | 8.0 | 25 | 2.81e-04 |
| llama-3-8b shape (Hq 32, Hkv 8, D 128) | 2048 | kvwear dense fp16 kernel | 96.7 | 8.0 | 87 | 2.08e-04 |
| llama-3-8b shape (Hq 32, Hkv 8, D 128) | 2048 | dequantize + SDPA (unfused int8) | 1108.5 | 4.1 | 4 | 8.36e-03 |
| llama-3-8b shape (Hq 32, Hkv 8, D 128) | 2048 | kvwear paged int8 v1 (warp/token) | 243.7 | 4.1 | 18 | 8.36e-03 |
| llama-3-8b shape (Hq 32, Hkv 8, D 128) | 2048 | kvwear paged int8 v2 (sub-warp/token) | 138.6 | 4.1 | 31 | 8.36e-03 |
| llama-3-8b shape (Hq 32, Hkv 8, D 128) | 2048 | kvwear paged int8 v3 (+ GQA head packing) | 112.1 | 4.1 | 39 | 8.36e-03 |
| llama-3-8b shape (Hq 32, Hkv 8, D 128) | 2048 | kvwear paged int8 v4 (tensor cores, WMMA) | 135.2 | 4.1 | 32 | 8.36e-03 |
| llama-3-8b shape (Hq 32, Hkv 8, D 128) | 2048 | kvwear paged int8 v5 (tensor cores, mma.sync) | 150.5 | 4.1 | 29 | 8.37e-03 |
| llama-3-8b shape (Hq 32, Hkv 8, D 128) | 4096 | torch SDPA fp16 | 782.8 | 16.0 | 21 | 2.78e-04 |
| llama-3-8b shape (Hq 32, Hkv 8, D 128) | 4096 | kvwear dense fp16 kernel | 210.3 | 16.0 | 80 | 2.01e-04 |
| llama-3-8b shape (Hq 32, Hkv 8, D 128) | 4096 | dequantize + SDPA (unfused int8) | 2466.8 | 8.2 | 4 | 8.26e-03 |
| llama-3-8b shape (Hq 32, Hkv 8, D 128) | 4096 | kvwear paged int8 v1 (warp/token) | 449.4 | 8.2 | 19 | 8.26e-03 |
| llama-3-8b shape (Hq 32, Hkv 8, D 128) | 4096 | kvwear paged int8 v2 (sub-warp/token) | 274.4 | 8.2 | 32 | 8.26e-03 |
| llama-3-8b shape (Hq 32, Hkv 8, D 128) | 4096 | kvwear paged int8 v3 (+ GQA head packing) | 239.0 | 8.2 | 36 | 8.26e-03 |
| llama-3-8b shape (Hq 32, Hkv 8, D 128) | 4096 | kvwear paged int8 v4 (tensor cores, WMMA) | 184.3 | 8.2 | 47 | 8.26e-03 |
| llama-3-8b shape (Hq 32, Hkv 8, D 128) | 4096 | kvwear paged int8 v5 (tensor cores, mma.sync) | 115.1 | 8.2 | 75 | 8.26e-03 |
| llama-3-8b shape (Hq 32, Hkv 8, D 128) | 8192 | torch SDPA fp16 | 1962.5 | 32.0 | 17 | 2.84e-04 |
| llama-3-8b shape (Hq 32, Hkv 8, D 128) | 8192 | kvwear dense fp16 kernel | 377.9 | 32.0 | 89 | 2.06e-04 |
| llama-3-8b shape (Hq 32, Hkv 8, D 128) | 8192 | dequantize + SDPA (unfused int8) | 5018.5 | 16.5 | 3 | 8.33e-03 |
| llama-3-8b shape (Hq 32, Hkv 8, D 128) | 8192 | kvwear paged int8 v1 (warp/token) | 845.8 | 16.5 | 20 | 8.31e-03 |
| llama-3-8b shape (Hq 32, Hkv 8, D 128) | 8192 | kvwear paged int8 v2 (sub-warp/token) | 479.7 | 16.5 | 36 | 8.31e-03 |
| llama-3-8b shape (Hq 32, Hkv 8, D 128) | 8192 | kvwear paged int8 v3 (+ GQA head packing) | 336.8 | 16.5 | 51 | 8.31e-03 |
| llama-3-8b shape (Hq 32, Hkv 8, D 128) | 8192 | kvwear paged int8 v4 (tensor cores, WMMA) | 286.1 | 16.5 | 60 | 8.32e-03 |
| llama-3-8b shape (Hq 32, Hkv 8, D 128) | 8192 | kvwear paged int8 v5 (tensor cores, mma.sync) | 183.3 | 16.5 | 94 | 8.30e-03 |
| llama-3-8b shape (Hq 32, Hkv 8, D 128) | 16384 | torch SDPA fp16 | 4109.8 | 64.0 | 16 | 2.88e-04 |
| llama-3-8b shape (Hq 32, Hkv 8, D 128) | 16384 | kvwear dense fp16 kernel | 881.7 | 64.0 | 76 | 2.08e-04 |
| llama-3-8b shape (Hq 32, Hkv 8, D 128) | 16384 | dequantize + SDPA (unfused int8) | 10062.3 | 33.0 | 3 | 8.47e-03 |
| llama-3-8b shape (Hq 32, Hkv 8, D 128) | 16384 | kvwear paged int8 v1 (warp/token) | 1736.7 | 33.0 | 20 | 8.46e-03 |
| llama-3-8b shape (Hq 32, Hkv 8, D 128) | 16384 | kvwear paged int8 v2 (sub-warp/token) | 975.8 | 33.0 | 35 | 8.46e-03 |
| llama-3-8b shape (Hq 32, Hkv 8, D 128) | 16384 | kvwear paged int8 v3 (+ GQA head packing) | 618.0 | 33.0 | 56 | 8.46e-03 |
| llama-3-8b shape (Hq 32, Hkv 8, D 128) | 16384 | kvwear paged int8 v4 (tensor cores, WMMA) | 574.0 | 33.0 | 60 | 8.47e-03 |
| llama-3-8b shape (Hq 32, Hkv 8, D 128) | 16384 | kvwear paged int8 v5 (tensor cores, mma.sync) | 308.6 | 33.0 | 112 | 8.46e-03 |
| llama-3-8b shape (Hq 32, Hkv 8, D 128) | 32768 | torch SDPA fp16 | 7594.0 | 128.0 | 18 | 2.94e-04 |
| llama-3-8b shape (Hq 32, Hkv 8, D 128) | 32768 | kvwear dense fp16 kernel | 2018.8 | 128.0 | 66 | 2.07e-04 |
| llama-3-8b shape (Hq 32, Hkv 8, D 128) | 32768 | dequantize + SDPA (unfused int8) | 20956.7 | 66.0 | 3 | 8.53e-03 |
| llama-3-8b shape (Hq 32, Hkv 8, D 128) | 32768 | kvwear paged int8 v1 (warp/token) | 3431.3 | 66.0 | 20 | 8.53e-03 |
| llama-3-8b shape (Hq 32, Hkv 8, D 128) | 32768 | kvwear paged int8 v2 (sub-warp/token) | 1901.5 | 66.0 | 36 | 8.53e-03 |
| llama-3-8b shape (Hq 32, Hkv 8, D 128) | 32768 | kvwear paged int8 v3 (+ GQA head packing) | 1363.5 | 66.0 | 51 | 8.53e-03 |
| llama-3-8b shape (Hq 32, Hkv 8, D 128) | 32768 | kvwear paged int8 v4 (tensor cores, WMMA) | 1200.6 | 66.0 | 58 | 8.52e-03 |
| llama-3-8b shape (Hq 32, Hkv 8, D 128) | 32768 | kvwear paged int8 v5 (tensor cores, mma.sync) | 582.1 | 66.0 | 119 | 8.53e-03 |
