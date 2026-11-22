# Decode attention over int8 KV (one layer, one decode step)

| model | tokens | impl | time (us) | KV MB read | GB/s | rel. error vs fp32 |
|---|---|---|---|---|---|---|
| qwen2.5-0.5b (Hq 14, Hkv 2) | 1024 | torch SDPA fp16 | 84.0 | 0.5 | 6 | 2.78e-04 |
| qwen2.5-0.5b (Hq 14, Hkv 2) | 1024 | kvwear dense fp16 kernel | 123.0 | 0.5 | 4 | 2.12e-04 |
| qwen2.5-0.5b (Hq 14, Hkv 2) | 1024 | dequantize + SDPA (unfused int8) | 270.2 | 0.3 | 1 | 8.19e-03 |
| qwen2.5-0.5b (Hq 14, Hkv 2) | 1024 | kvwear paged int8 v1 (warp/token) | 148.9 | 0.3 | 2 | 8.16e-03 |
| qwen2.5-0.5b (Hq 14, Hkv 2) | 1024 | kvwear paged int8 v2 (sub-warp/token) | 75.2 | 0.3 | 4 | 8.16e-03 |
| qwen2.5-0.5b (Hq 14, Hkv 2) | 1024 | kvwear paged int8 v3 (+ GQA head packing) | 83.3 | 0.3 | 3 | 8.16e-03 |
| qwen2.5-0.5b (Hq 14, Hkv 2) | 2048 | torch SDPA fp16 | 128.5 | 1.0 | 8 | 2.76e-04 |
| qwen2.5-0.5b (Hq 14, Hkv 2) | 2048 | kvwear dense fp16 kernel | 77.8 | 1.0 | 13 | 2.14e-04 |
| qwen2.5-0.5b (Hq 14, Hkv 2) | 2048 | dequantize + SDPA (unfused int8) | 208.9 | 0.5 | 3 | 8.29e-03 |
| qwen2.5-0.5b (Hq 14, Hkv 2) | 2048 | kvwear paged int8 v1 (warp/token) | 71.6 | 0.5 | 8 | 8.27e-03 |
| qwen2.5-0.5b (Hq 14, Hkv 2) | 2048 | kvwear paged int8 v2 (sub-warp/token) | 63.4 | 0.5 | 9 | 8.27e-03 |
| qwen2.5-0.5b (Hq 14, Hkv 2) | 2048 | kvwear paged int8 v3 (+ GQA head packing) | 90.1 | 0.5 | 6 | 8.27e-03 |
| qwen2.5-0.5b (Hq 14, Hkv 2) | 4096 | torch SDPA fp16 | 136.2 | 2.0 | 15 | 2.70e-04 |
| qwen2.5-0.5b (Hq 14, Hkv 2) | 4096 | kvwear dense fp16 kernel | 70.7 | 2.0 | 30 | 2.04e-04 |
| qwen2.5-0.5b (Hq 14, Hkv 2) | 4096 | dequantize + SDPA (unfused int8) | 321.5 | 1.0 | 3 | 8.85e-03 |
| qwen2.5-0.5b (Hq 14, Hkv 2) | 4096 | kvwear paged int8 v1 (warp/token) | 130.0 | 1.0 | 8 | 8.86e-03 |
| qwen2.5-0.5b (Hq 14, Hkv 2) | 4096 | kvwear paged int8 v2 (sub-warp/token) | 56.3 | 1.0 | 19 | 8.86e-03 |
| qwen2.5-0.5b (Hq 14, Hkv 2) | 4096 | kvwear paged int8 v3 (+ GQA head packing) | 119.8 | 1.0 | 9 | 8.86e-03 |
| qwen2.5-0.5b (Hq 14, Hkv 2) | 8192 | torch SDPA fp16 | 282.6 | 4.0 | 15 | 2.84e-04 |
| qwen2.5-0.5b (Hq 14, Hkv 2) | 8192 | kvwear dense fp16 kernel | 158.5 | 4.0 | 26 | 2.16e-04 |
| qwen2.5-0.5b (Hq 14, Hkv 2) | 8192 | dequantize + SDPA (unfused int8) | 711.2 | 2.1 | 3 | 8.61e-03 |
| qwen2.5-0.5b (Hq 14, Hkv 2) | 8192 | kvwear paged int8 v1 (warp/token) | 273.4 | 2.1 | 8 | 8.61e-03 |
| qwen2.5-0.5b (Hq 14, Hkv 2) | 8192 | kvwear paged int8 v2 (sub-warp/token) | 161.3 | 2.1 | 13 | 8.61e-03 |
| qwen2.5-0.5b (Hq 14, Hkv 2) | 8192 | kvwear paged int8 v3 (+ GQA head packing) | 146.4 | 2.1 | 15 | 8.61e-03 |
| qwen2.5-0.5b (Hq 14, Hkv 2) | 16384 | torch SDPA fp16 | 458.8 | 8.0 | 18 | 3.08e-04 |
| qwen2.5-0.5b (Hq 14, Hkv 2) | 16384 | kvwear dense fp16 kernel | 290.3 | 8.0 | 29 | 2.26e-04 |
| qwen2.5-0.5b (Hq 14, Hkv 2) | 16384 | dequantize + SDPA (unfused int8) | 1390.1 | 4.1 | 3 | 8.54e-03 |
| qwen2.5-0.5b (Hq 14, Hkv 2) | 16384 | kvwear paged int8 v1 (warp/token) | 494.1 | 4.1 | 9 | 8.50e-03 |
| qwen2.5-0.5b (Hq 14, Hkv 2) | 16384 | kvwear paged int8 v2 (sub-warp/token) | 213.5 | 4.1 | 20 | 8.50e-03 |
| qwen2.5-0.5b (Hq 14, Hkv 2) | 16384 | kvwear paged int8 v3 (+ GQA head packing) | 173.6 | 4.1 | 25 | 8.50e-03 |
| qwen2.5-0.5b (Hq 14, Hkv 2) | 32768 | torch SDPA fp16 | 731.0 | 16.0 | 23 | 2.92e-04 |
| qwen2.5-0.5b (Hq 14, Hkv 2) | 32768 | kvwear dense fp16 kernel | 344.4 | 16.0 | 49 | 2.07e-04 |
| qwen2.5-0.5b (Hq 14, Hkv 2) | 32768 | dequantize + SDPA (unfused int8) | 2426.8 | 8.2 | 4 | 9.09e-03 |
| qwen2.5-0.5b (Hq 14, Hkv 2) | 32768 | kvwear paged int8 v1 (warp/token) | 702.9 | 8.2 | 12 | 9.07e-03 |
| qwen2.5-0.5b (Hq 14, Hkv 2) | 32768 | kvwear paged int8 v2 (sub-warp/token) | 309.8 | 8.2 | 28 | 9.07e-03 |
| qwen2.5-0.5b (Hq 14, Hkv 2) | 32768 | kvwear paged int8 v3 (+ GQA head packing) | 256.9 | 8.2 | 34 | 9.07e-03 |
| opt-1.3b (Hq 32, Hkv 32) | 1024 | torch SDPA fp16 | 100.9 | 8.0 | 83 | 2.67e-04 |
| opt-1.3b (Hq 32, Hkv 32) | 1024 | kvwear dense fp16 kernel | 85.9 | 8.0 | 98 | 1.99e-04 |
| opt-1.3b (Hq 32, Hkv 32) | 1024 | dequantize + SDPA (unfused int8) | 210.4 | 4.1 | 21 | 8.22e-03 |
| opt-1.3b (Hq 32, Hkv 32) | 1024 | kvwear paged int8 v1 (warp/token) | 96.6 | 4.1 | 45 | 8.20e-03 |
| opt-1.3b (Hq 32, Hkv 32) | 1024 | kvwear paged int8 v2 (sub-warp/token) | 65.5 | 4.1 | 66 | 8.21e-03 |
| opt-1.3b (Hq 32, Hkv 32) | 1024 | kvwear paged int8 v3 (+ GQA head packing) | 75.3 | 4.1 | 57 | 8.21e-03 |
| opt-1.3b (Hq 32, Hkv 32) | 2048 | torch SDPA fp16 | 169.0 | 16.0 | 99 | 2.88e-04 |
| opt-1.3b (Hq 32, Hkv 32) | 2048 | kvwear dense fp16 kernel | 106.3 | 16.0 | 158 | 2.07e-04 |
| opt-1.3b (Hq 32, Hkv 32) | 2048 | dequantize + SDPA (unfused int8) | 408.1 | 8.2 | 21 | 8.45e-03 |
| opt-1.3b (Hq 32, Hkv 32) | 2048 | kvwear paged int8 v1 (warp/token) | 133.1 | 8.2 | 65 | 8.45e-03 |
| opt-1.3b (Hq 32, Hkv 32) | 2048 | kvwear paged int8 v2 (sub-warp/token) | 95.2 | 8.2 | 91 | 8.45e-03 |
| opt-1.3b (Hq 32, Hkv 32) | 2048 | kvwear paged int8 v3 (+ GQA head packing) | 77.5 | 8.2 | 112 | 8.45e-03 |
| opt-1.3b (Hq 32, Hkv 32) | 4096 | torch SDPA fp16 | 315.9 | 32.0 | 106 | 2.82e-04 |
| opt-1.3b (Hq 32, Hkv 32) | 4096 | kvwear dense fp16 kernel | 232.4 | 32.0 | 144 | 2.09e-04 |
| opt-1.3b (Hq 32, Hkv 32) | 4096 | dequantize + SDPA (unfused int8) | 829.4 | 16.5 | 21 | 9.11e-03 |
| opt-1.3b (Hq 32, Hkv 32) | 4096 | kvwear paged int8 v1 (warp/token) | 266.2 | 16.5 | 65 | 9.09e-03 |
| opt-1.3b (Hq 32, Hkv 32) | 4096 | kvwear paged int8 v2 (sub-warp/token) | 112.0 | 16.5 | 154 | 9.09e-03 |
| opt-1.3b (Hq 32, Hkv 32) | 4096 | kvwear paged int8 v3 (+ GQA head packing) | 149.4 | 16.5 | 116 | 9.09e-03 |
| opt-1.3b (Hq 32, Hkv 32) | 8192 | torch SDPA fp16 | 794.6 | 64.0 | 84 | 2.88e-04 |
| opt-1.3b (Hq 32, Hkv 32) | 8192 | kvwear dense fp16 kernel | 410.6 | 64.0 | 163 | 2.02e-04 |
| opt-1.3b (Hq 32, Hkv 32) | 8192 | dequantize + SDPA (unfused int8) | 1794.0 | 33.0 | 19 | 8.68e-03 |
| opt-1.3b (Hq 32, Hkv 32) | 8192 | kvwear paged int8 v1 (warp/token) | 484.8 | 33.0 | 71 | 8.68e-03 |
| opt-1.3b (Hq 32, Hkv 32) | 8192 | kvwear paged int8 v2 (sub-warp/token) | 243.2 | 33.0 | 142 | 8.68e-03 |
| opt-1.3b (Hq 32, Hkv 32) | 8192 | kvwear paged int8 v3 (+ GQA head packing) | 253.3 | 33.0 | 137 | 8.68e-03 |
| opt-1.3b (Hq 32, Hkv 32) | 16384 | torch SDPA fp16 | 1570.3 | 128.0 | 85 | 2.93e-04 |
| opt-1.3b (Hq 32, Hkv 32) | 16384 | kvwear dense fp16 kernel | 749.6 | 128.0 | 179 | 2.05e-04 |
| opt-1.3b (Hq 32, Hkv 32) | 16384 | dequantize + SDPA (unfused int8) | 3331.1 | 66.0 | 21 | 8.48e-03 |
| opt-1.3b (Hq 32, Hkv 32) | 16384 | kvwear paged int8 v1 (warp/token) | 843.3 | 66.0 | 82 | 8.48e-03 |
| opt-1.3b (Hq 32, Hkv 32) | 16384 | kvwear paged int8 v2 (sub-warp/token) | 429.1 | 66.0 | 161 | 8.48e-03 |
| opt-1.3b (Hq 32, Hkv 32) | 16384 | kvwear paged int8 v3 (+ GQA head packing) | 432.1 | 66.0 | 160 | 8.48e-03 |
| opt-1.3b (Hq 32, Hkv 32) | 32768 | torch SDPA fp16 | 3245.0 | 256.0 | 83 | 2.87e-04 |
| opt-1.3b (Hq 32, Hkv 32) | 32768 | kvwear dense fp16 kernel | 1562.6 | 256.0 | 172 | 2.05e-04 |
| opt-1.3b (Hq 32, Hkv 32) | 32768 | dequantize + SDPA (unfused int8) | 6680.5 | 132.0 | 21 | 8.32e-03 |
| opt-1.3b (Hq 32, Hkv 32) | 32768 | kvwear paged int8 v1 (warp/token) | 1642.9 | 132.0 | 84 | 8.33e-03 |
| opt-1.3b (Hq 32, Hkv 32) | 32768 | kvwear paged int8 v2 (sub-warp/token) | 802.8 | 132.0 | 172 | 8.33e-03 |
| opt-1.3b (Hq 32, Hkv 32) | 32768 | kvwear paged int8 v3 (+ GQA head packing) | 804.8 | 132.0 | 172 | 8.33e-03 |
