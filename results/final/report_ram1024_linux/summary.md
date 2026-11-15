# KV offload SSD wear report

- write counter used: `linux_bytes_written`
- baseline: `write_through`; speed metric: `latency_mean_s`
- wear projection: 50 chats/day, SSD rated 300 TBW
- idle background writes subtracted: 1.109 GB/hour

| policy        | family         |   runs |   gb_per_1k_tokens |   gb_per_chat |   writes_saved_pct |   slowdown_pct |   ttft_mean_s |   speed_mean |   years_to_tbw |   pct_tbw_per_year |   virt_amplification |
|:--------------|:---------------|-------:|-------------------:|--------------:|-------------------:|---------------:|--------------:|-------------:|---------------:|-------------------:|---------------------:|
| write_through | baseline       |      3 |              0.032 |         0.100 |              0.000 |          0.000 |         0.241 |        1.107 |        164.632 |              0.607 |                1.726 |
| spill         | baseline       |      3 |              0.015 |         0.046 |             53.729 |        -13.830 |         0.194 |        0.954 |        355.803 |              0.281 |                1.413 |
| spill_int8    | quantize only  |      3 |              0.008 |         0.024 |             75.982 |        -13.650 |         0.199 |        0.956 |        685.463 |              0.146 |                0.674 |
| spill_int4    | quantize only  |      3 |              0.004 |         0.013 |             86.716 |        -11.484 |         0.207 |        0.979 |       1239.371 |              0.081 |                0.613 |
| reuse_int8    | reuse-aware    |      3 |              0.000 |         0.000 |             99.680 |         -2.839 |         0.207 |        1.075 |      51485.190 |              0.002 |              nan     |
| oracle        | oracle (bound) |      3 |              0.000 |         0.000 |             99.707 |         -1.244 |         0.225 |        1.093 |      56208.271 |              0.002 |              nan     |
| reuse_int4    | reuse-aware    |      3 |              0.000 |         0.000 |             99.766 |         -8.254 |         0.207 |        1.015 |      70470.071 |              0.001 |              nan     |
| no_ssd        | baseline       |      3 |              0.000 |         0.000 |             99.774 |          0.955 |         0.234 |        1.117 |      72990.674 |              0.001 |              nan     |
| reuse         | reuse-aware    |      3 |              0.000 |         0.000 |             99.812 |        -11.461 |         0.201 |        0.980 |      87371.637 |              0.001 |              nan     |
