# KV offload SSD wear report

- write counter used: `linux_bytes_written`
- baseline: `write_through`; speed metric: `latency_mean_s`
- wear projection: 50 chats/day, SSD rated 300 TBW
- idle background writes subtracted: 2.382 GB/hour

| policy        | family         |   runs |   gb_per_1k_tokens |   gb_per_chat |   writes_saved_pct |   slowdown_pct |   ttft_mean_s |   speed_mean |   years_to_tbw |   pct_tbw_per_year |   virt_amplification |
|:--------------|:---------------|-------:|-------------------:|--------------:|-------------------:|---------------:|--------------:|-------------:|---------------:|-------------------:|---------------------:|
| write_through | baseline       |      3 |              0.001 |         0.014 |              0.000 |          0.000 |         0.181 |        1.490 |       1208.536 |              0.083 |                2.105 |
| spill         | baseline       |      3 |              0.001 |         0.013 |              3.761 |          1.420 |         0.175 |        1.511 |       1255.766 |              0.080 |                2.110 |
| reuse         | reuse-aware    |      3 |              0.001 |         0.013 |              7.389 |         -2.382 |         0.175 |        1.455 |       1304.961 |              0.077 |              nan     |
| oracle        | oracle (bound) |      3 |              0.001 |         0.011 |             20.677 |         -2.809 |         0.166 |        1.448 |       1523.557 |              0.066 |              nan     |
| reuse_int8    | reuse-aware    |      3 |              0.001 |         0.006 |             52.213 |         -3.267 |         0.190 |        1.441 |       2529.007 |              0.040 |              nan     |
| no_ssd        | baseline       |      3 |              0.000 |         0.001 |             95.787 |         42.634 |         0.793 |        2.125 |      28687.563 |              0.003 |              nan     |
