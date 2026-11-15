# KV offload SSD wear report

- write counter used: `smart_kv_bytes`
- baseline: `write_through`; speed metric: `latency_mean_s`
- wear projection: 50 chats/day, SSD rated 300 TBW
- idle background writes subtracted: 1.448 GB/hour
- KV-attributable writes = SMART minus the `no_ssd` control run (0.24 ± 0.40 GB per run: everything else the LLM run writes, mostly Windows paging)

| policy        | family         |   runs |   gb_per_1k_tokens |   gb_per_chat |   writes_saved_pct |   slowdown_pct |   ttft_mean_s |   speed_mean |   years_to_tbw |   pct_tbw_per_year |   virt_amplification |
|:--------------|:---------------|-------:|-------------------:|--------------:|-------------------:|---------------:|--------------:|-------------:|---------------:|-------------------:|---------------------:|
| write_through | baseline       |      3 |              0.195 |         0.488 |              0.000 |          0.000 |         0.224 |        1.203 |         33.698 |              2.968 |                1.640 |
| spill         | baseline       |      3 |              0.176 |         0.439 |             10.048 |          0.303 |         0.227 |        1.206 |         37.462 |              2.669 |                1.517 |
| reuse         | reuse-aware    |      3 |              0.152 |         0.379 |             22.387 |          1.528 |         0.225 |        1.221 |         43.418 |              2.303 |                2.767 |
| reuse_int8    | reuse-aware    |      3 |              0.063 |         0.158 |             67.607 |          6.450 |         0.246 |        1.280 |        104.030 |              0.961 |                2.238 |
| oracle        | oracle (bound) |      3 |              0.053 |         0.132 |             72.962 |         -1.987 |         0.215 |        1.179 |        124.633 |              0.802 |                2.459 |
| no_ssd        | baseline       |      3 |              0.000 |         0.000 |            100.000 |          1.281 |         0.226 |        1.218 |        inf     |              0.000 |              nan     |
