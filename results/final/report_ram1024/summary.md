# KV offload SSD wear report

- write counter used: `smart_kv_bytes`
- baseline: `write_through`; speed metric: `latency_mean_s`
- wear projection: 50 chats/day, SSD rated 300 TBW
- idle background writes subtracted: 1.109 GB/hour
- KV-attributable writes = SMART minus the `no_ssd` control run (1.19 ± 0.52 GB per run: everything else the LLM run writes, mostly Windows paging)

| policy        | family         |   runs |   gb_per_1k_tokens |   gb_per_chat |   writes_saved_pct |   slowdown_pct |   ttft_mean_s |   speed_mean |   years_to_tbw |   pct_tbw_per_year |   virt_amplification |
|:--------------|:---------------|-------:|-------------------:|--------------:|-------------------:|---------------:|--------------:|-------------:|---------------:|-------------------:|---------------------:|
| write_through | baseline       |      3 |              0.055 |         0.172 |              0.000 |          0.000 |         0.241 |        1.107 |         95.627 |              1.046 |                1.726 |
| spill         | baseline       |      3 |              0.021 |         0.065 |             62.188 |        -13.830 |         0.194 |        0.954 |        252.899 |              0.395 |                1.413 |
| spill_int8    | quantize only  |      3 |              0.005 |         0.016 |             90.635 |        -13.650 |         0.199 |        0.956 |       1021.099 |              0.098 |                0.674 |
| spill_int4    | quantize only  |      3 |              0.003 |         0.008 |             95.354 |        -11.484 |         0.207 |        0.979 |       2058.310 |              0.049 |                0.613 |
| no_ssd        | baseline       |      3 |             -0.000 |        -0.000 |            100.000 |          0.955 |         0.234 |        1.117 |        inf     |             -0.000 |              nan     |
| reuse_int8    | reuse-aware    |      3 |             -0.000 |        -0.001 |            100.717 |         -2.839 |         0.207 |        1.075 |        inf     |             -0.007 |              nan     |
| reuse         | reuse-aware    |      3 |             -0.001 |        -0.005 |            102.658 |        -11.461 |         0.201 |        0.980 |        inf     |             -0.028 |              nan     |
| oracle        | oracle (bound) |      3 |             -0.003 |        -0.010 |            105.811 |         -1.244 |         0.225 |        1.093 |        inf     |             -0.061 |              nan     |
| reuse_int4    | reuse-aware    |      3 |             -0.010 |        -0.031 |            118.162 |         -8.254 |         0.207 |        1.015 |        inf     |             -0.190 |              nan     |
