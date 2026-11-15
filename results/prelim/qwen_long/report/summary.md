# KV offload SSD wear report

- write counter used: `linux_bytes_written`
- baseline: `write_through`; speed metric: `latency_mean_s`
- wear projection: 50 chats/day, SSD rated 300 TBW

| policy        | family         |   runs |   gb_per_1k_tokens |   gb_per_chat |   writes_saved_pct |   slowdown_pct |   ttft_mean_s |   speed_mean |   years_to_tbw |   pct_tbw_per_year |
|:--------------|:---------------|-------:|-------------------:|--------------:|-------------------:|---------------:|--------------:|-------------:|---------------:|-------------------:|
| write_through | baseline       |      2 |              0.001 |         0.013 |              0.000 |          0.000 |         0.206 |        1.651 |       1241.718 |              0.081 |
| spill         | baseline       |      2 |              0.001 |         0.013 |              1.796 |          4.968 |         0.200 |        1.733 |       1264.433 |              0.079 |
| reuse         | reuse-aware    |      2 |              0.001 |         0.012 |             10.407 |          6.143 |         0.209 |        1.752 |       1385.947 |              0.072 |
| oracle        | oracle (bound) |      2 |              0.001 |         0.011 |             18.409 |         13.863 |         0.216 |        1.880 |       1521.890 |              0.066 |
| reuse_int8    | reuse-aware    |      2 |              0.000 |         0.006 |             52.855 |          9.302 |         0.265 |        1.805 |       2633.812 |              0.038 |
| no_ssd        | baseline       |      2 |              0.000 |         0.000 |             98.463 |         42.402 |         0.826 |        2.351 |      80770.225 |              0.001 |
