# KV offload SSD wear report

- write counter used: `smart_kv_bytes`
- baseline: `write_through`; speed metric: `latency_mean_s`
- wear projection: 50 chats/day, SSD rated 300 TBW
- idle background writes subtracted: 1.581 GB/hour
- KV-attributable writes = SMART minus the `no_ssd` control run (0.01 ± 0.03 GB per run: everything else the LLM run writes, mostly Windows paging)

| policy        | family         |   runs |   gb_per_1k_tokens |   gb_per_chat |   writes_saved_pct |   slowdown_pct |   ttft_mean_s |   speed_mean |              years_to_tbw |   pct_tbw_per_year |   virt_amplification |
|:--------------|:---------------|-------:|-------------------:|--------------:|-------------------:|---------------:|--------------:|-------------:|--------------------------:|-------------------:|---------------------:|
| write_through | baseline       |      3 |              0.002 |         0.007 |              0.000 |          0.000 |         0.061 |        1.218 |                  2303.770 |              0.043 |                  nan |
| spill         | baseline       |      3 |              0.002 |         0.007 |              3.358 |         -3.993 |         0.059 |        1.170 |                  2383.821 |              0.042 |                  nan |
| reuse         | reuse-aware    |      3 |              0.002 |         0.005 |             30.023 |         -6.079 |         0.061 |        1.144 |                  3292.160 |              0.030 |                  nan |
| oracle        | oracle (bound) |      3 |              0.001 |         0.004 |             49.650 |         -3.742 |         0.060 |        1.173 |                  4575.512 |              0.022 |                  nan |
| spill_int8    | quantize only  |      3 |              0.001 |         0.003 |             60.283 |          3.354 |         0.079 |        1.259 |                  5800.479 |              0.017 |                  nan |
| reuse_int8    | reuse-aware    |      3 |              0.001 |         0.002 |             78.132 |         -1.176 |         0.080 |        1.204 |                 10534.929 |              0.009 |                  nan |
| no_ssd        | baseline       |      3 |              0.000 |         0.000 |            100.000 |         -0.772 |         0.108 |        1.209 | 227425611867652030464.000 |              0.000 |                  nan |
