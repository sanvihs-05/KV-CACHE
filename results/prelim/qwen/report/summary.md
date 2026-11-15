# KV offload SSD wear report

- write counter used: `linux_bytes_written`
- baseline: `write_through`; speed metric: `latency_mean_s`
- wear projection: 50 chats/day, SSD rated 300 TBW

| policy        | family         |   runs |   gb_per_1k_tokens |   gb_per_chat |   writes_saved_pct |   slowdown_pct |   ttft_mean_s |   speed_mean |   years_to_tbw |   pct_tbw_per_year |
|:--------------|:---------------|-------:|-------------------:|--------------:|-------------------:|---------------:|--------------:|-------------:|---------------:|-------------------:|
| write_through | baseline       |      2 |              0.003 |         0.008 |              0.000 |          0.000 |         0.069 |        1.386 |       1989.156 |              0.050 |
| reuse_int8    | reuse-aware    |      2 |              0.002 |         0.008 |              6.357 |          5.914 |         0.087 |        1.468 |       2124.182 |              0.047 |
| spill         | baseline       |      2 |              0.002 |         0.006 |             23.748 |         14.715 |         0.081 |        1.590 |       2608.645 |              0.038 |
| oracle        | oracle (bound) |      2 |              0.002 |         0.006 |             31.604 |         19.250 |         0.102 |        1.653 |       2908.272 |              0.034 |
| reuse         | reuse-aware    |      2 |              0.001 |         0.005 |             44.617 |          5.680 |         0.078 |        1.465 |       3591.615 |              0.028 |
| spill_int8    | quantize only  |      2 |              0.001 |         0.004 |             56.342 |          7.164 |         0.088 |        1.486 |       4556.263 |              0.022 |
| no_ssd        | baseline       |      2 |              0.000 |         0.000 |             98.214 |         15.012 |         0.128 |        1.595 |     111402.375 |              0.001 |
