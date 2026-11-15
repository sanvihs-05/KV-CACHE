# KV offload SSD wear report

- write counter used: `smart_kv_bytes`
- baseline: `write_through`; speed metric: `latency_mean_s`
- wear projection: 50 chats/day, SSD rated 300 TBW
- idle background writes subtracted: 0.917 GB/hour
- KV-attributable writes = SMART minus the `no_ssd` control run (0.86 ± 1.47 GB per run: everything else the LLM run writes, mostly Windows paging)

| policy        | family         |   runs |   gb_per_1k_tokens |   gb_per_chat |   writes_saved_pct |   slowdown_pct |   ttft_mean_s |   speed_mean |   years_to_tbw |   pct_tbw_per_year |   virt_amplification |
|:--------------|:---------------|-------:|-------------------:|--------------:|-------------------:|---------------:|--------------:|-------------:|---------------:|-------------------:|---------------------:|
| write_through | baseline       |      3 |              0.064 |         0.199 |              0.000 |          0.000 |         0.348 |        1.410 |         82.680 |              1.209 |                1.998 |
| spill         | baseline       |      3 |              0.049 |         0.153 |             23.247 |         -0.774 |         0.335 |        1.400 |        107.721 |              0.928 |                1.562 |
| reuse         | reuse-aware    |      3 |              0.041 |         0.126 |             36.413 |         -3.704 |         0.305 |        1.358 |        130.026 |              0.769 |                1.820 |
| oracle        | oracle (bound) |      3 |              0.034 |         0.106 |             46.726 |         -4.147 |         0.300 |        1.352 |        155.197 |              0.644 |                1.766 |
| spill_int8    | quantize only  |      3 |              0.028 |         0.086 |             56.724 |         -2.401 |         0.313 |        1.377 |        191.054 |              0.523 |                1.701 |
| reuse_int8    | reuse-aware    |      3 |              0.022 |         0.068 |             65.840 |         -3.465 |         0.303 |        1.362 |        242.037 |              0.413 |                1.893 |
| spill_int4    | quantize only  |      3 |              0.016 |         0.051 |             74.458 |          0.581 |         0.364 |        1.419 |        323.695 |              0.309 |                1.837 |
| reuse_int4    | reuse-aware    |      3 |              0.001 |         0.004 |             97.978 |         -1.098 |         0.331 |        1.395 |       4089.368 |              0.024 |                0.221 |
| no_ssd        | baseline       |      3 |             -0.000 |        -0.000 |            100.000 |          8.295 |         0.299 |        1.527 |        inf     |             -0.000 |              nan     |
