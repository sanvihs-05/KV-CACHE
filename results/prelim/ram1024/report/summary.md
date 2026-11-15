# KV offload SSD wear report

- write counter used: `linux_bytes_written`
- baseline: `write_through`; speed metric: `latency_mean_s`
- wear projection: 50 chats/day, SSD rated 300 TBW

| policy         | family         |   runs |   gb_per_1k_tokens |   gb_per_chat |   writes_saved_pct |   slowdown_pct |   ttft_mean_s |   speed_mean |   years_to_tbw |   pct_tbw_per_year |
|:---------------|:---------------|-------:|-------------------:|--------------:|-------------------:|---------------:|--------------:|-------------:|---------------:|-------------------:|
| write_through  | baseline       |      2 |              0.032 |         0.100 |              0.000 |          0.000 |         0.244 |        1.021 |        164.939 |              0.606 |
| spill          | baseline       |      2 |              0.015 |         0.046 |             53.810 |         -0.123 |         0.224 |        1.020 |        357.087 |              0.280 |
| spill_int8     | quantize only  |      2 |              0.008 |         0.024 |             76.133 |         -0.706 |         0.222 |        1.014 |        691.076 |              0.145 |
| spill_int4     | quantize only  |      2 |              0.004 |         0.013 |             86.939 |         -2.767 |         0.213 |        0.993 |       1262.839 |              0.079 |
| reuse@0.3/fp16 | reuse-aware    |      2 |              0.000 |         0.000 |             99.954 |          1.152 |         0.244 |        1.033 |     359934.578 |              0.000 |
| reuse_int4     | reuse-aware    |      2 |              0.000 |         0.000 |             99.959 |         -3.111 |         0.206 |        0.989 |     398339.508 |              0.000 |
| reuse          | reuse-aware    |      2 |              0.000 |         0.000 |             99.961 |         -2.454 |         0.208 |        0.996 |     423564.174 |              0.000 |
| reuse_int8     | reuse-aware    |      2 |              0.000 |         0.000 |             99.965 |         -3.821 |         0.200 |        0.982 |     465306.730 |              0.000 |
| reuse@0.7/fp16 | reuse-aware    |      2 |              0.000 |         0.000 |             99.968 |         -1.229 |         0.209 |        1.008 |     521203.967 |              0.000 |
| no_ssd         | baseline       |      2 |              0.000 |         0.000 |             99.973 |         -2.669 |         0.211 |        0.994 |     610383.353 |              0.000 |
| oracle         | oracle (bound) |      2 |              0.000 |         0.000 |             99.975 |         -2.678 |         0.209 |        0.994 |     655227.845 |              0.000 |
