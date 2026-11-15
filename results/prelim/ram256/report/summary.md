# KV offload SSD wear report

- write counter used: `linux_bytes_written`
- baseline: `write_through`; speed metric: `latency_mean_s`
- wear projection: 50 chats/day, SSD rated 300 TBW

| policy         | family         |   runs |   gb_per_1k_tokens |   gb_per_chat |   writes_saved_pct |   slowdown_pct |   ttft_mean_s |   speed_mean |   years_to_tbw |   pct_tbw_per_year |
|:---------------|:---------------|-------:|-------------------:|--------------:|-------------------:|---------------:|--------------:|-------------:|---------------:|-------------------:|
| write_through  | baseline       |      2 |              0.032 |         0.100 |              0.000 |          0.000 |         0.260 |        1.019 |        165.057 |              0.606 |
| spill          | baseline       |      2 |              0.031 |         0.098 |              1.864 |          2.032 |         0.271 |        1.040 |        168.192 |              0.595 |
| reuse@0.3/fp16 | reuse-aware    |      2 |              0.028 |         0.087 |             12.645 |          4.646 |         0.281 |        1.066 |        188.950 |              0.529 |
| reuse          | reuse-aware    |      2 |              0.022 |         0.069 |             30.369 |         -0.983 |         0.241 |        1.009 |        237.047 |              0.422 |
| oracle         | oracle (bound) |      2 |              0.019 |         0.060 |             39.853 |         -2.045 |         0.241 |        0.998 |        274.423 |              0.364 |
| reuse@0.7/fp16 | reuse-aware    |      2 |              0.019 |         0.059 |             40.478 |         -0.513 |         0.246 |        1.014 |        277.305 |              0.361 |
| spill_int8     | quantize only  |      2 |              0.016 |         0.050 |             49.321 |          4.106 |         0.303 |        1.061 |        325.689 |              0.307 |
| reuse_int8     | reuse-aware    |      2 |              0.012 |         0.036 |             64.023 |         -1.401 |         0.256 |        1.005 |        458.785 |              0.218 |
| spill_int4     | quantize only  |      2 |              0.009 |         0.028 |             72.252 |          6.762 |         0.334 |        1.088 |        594.849 |              0.168 |
| reuse_int4     | reuse-aware    |      2 |              0.006 |         0.020 |             80.296 |          6.369 |         0.326 |        1.084 |        837.691 |              0.119 |
| no_ssd         | baseline       |      2 |              0.000 |         0.000 |             99.956 |          8.198 |         0.218 |        1.103 |     375950.403 |              0.000 |
