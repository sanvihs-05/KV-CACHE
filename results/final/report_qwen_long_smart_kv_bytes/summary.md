# KV offload SSD wear report

- write counter used: `smart_kv_bytes`
- baseline: `write_through`; speed metric: `latency_mean_s`
- wear projection: 50 chats/day, SSD rated 300 TBW
- idle background writes subtracted: 2.382 GB/hour
- KV-attributable writes = SMART minus the `no_ssd` control run (0.28 ± 0.50 GB per run: everything else the LLM run writes, mostly Windows paging)

| policy        | family         |   runs |   gb_per_1k_tokens |   gb_per_chat |   writes_saved_pct |   slowdown_pct |   ttft_mean_s |   speed_mean |   years_to_tbw |   pct_tbw_per_year |   virt_amplification |
|:--------------|:---------------|-------:|-------------------:|--------------:|-------------------:|---------------:|--------------:|-------------:|---------------:|-------------------:|---------------------:|
| write_through | baseline       |      3 |              0.002 |         0.026 |              0.000 |          0.000 |         0.181 |        1.490 |        622.136 |              0.161 |                2.105 |
| spill         | baseline       |      3 |              0.001 |         0.008 |             71.428 |          1.420 |         0.175 |        1.511 |       2177.409 |              0.046 |                2.110 |
| reuse_int8    | reuse-aware    |      3 |              0.000 |         0.004 |             84.060 |         -3.267 |         0.190 |        1.441 |       3902.899 |              0.026 |              nan     |
| no_ssd        | baseline       |      3 |             -0.000 |        -0.000 |            100.000 |         42.634 |         0.793 |        2.125 |        inf     |             -0.000 |              nan     |
| reuse         | reuse-aware    |      3 |             -0.000 |        -0.002 |            107.027 |         -2.382 |         0.175 |        1.455 |        inf     |             -0.011 |              nan     |
| oracle        | oracle (bound) |      3 |             -0.000 |        -0.006 |            123.461 |         -2.809 |         0.166 |        1.448 |        inf     |             -0.038 |              nan     |
