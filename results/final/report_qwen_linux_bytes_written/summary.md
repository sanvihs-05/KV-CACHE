# KV offload SSD wear report

- write counter used: `linux_bytes_written`
- baseline: `write_through`; speed metric: `latency_mean_s`
- wear projection: 50 chats/day, SSD rated 300 TBW
- idle background writes subtracted: 1.581 GB/hour

| policy        | family         |   runs |   gb_per_1k_tokens |   gb_per_chat |   writes_saved_pct |   slowdown_pct |   ttft_mean_s |   speed_mean |   years_to_tbw |   pct_tbw_per_year |   virt_amplification |
|:--------------|:---------------|-------:|-------------------:|--------------:|-------------------:|---------------:|--------------:|-------------:|---------------:|-------------------:|---------------------:|
| write_through | baseline       |      3 |              0.002 |         0.007 |              0.000 |          0.000 |         0.061 |        1.218 |       2410.879 |              0.041 |                  nan |
| spill         | baseline       |      3 |              0.002 |         0.007 |              0.973 |         -3.993 |         0.059 |        1.170 |       2434.572 |              0.041 |                  nan |
| reuse         | reuse-aware    |      3 |              0.002 |         0.005 |             25.652 |         -6.079 |         0.061 |        1.144 |       3242.698 |              0.031 |                  nan |
| oracle        | oracle (bound) |      3 |              0.001 |         0.004 |             37.174 |         -3.742 |         0.060 |        1.173 |       3837.390 |              0.026 |                  nan |
| spill_int8    | quantize only  |      3 |              0.001 |         0.004 |             48.046 |          3.354 |         0.079 |        1.259 |       4640.424 |              0.022 |                  nan |
| reuse_int8    | reuse-aware    |      3 |              0.001 |         0.003 |             61.935 |         -1.176 |         0.080 |        1.204 |       6333.576 |              0.016 |                  nan |
| no_ssd        | baseline       |      3 |              0.000 |         0.000 |             93.533 |         -0.772 |         0.108 |        1.209 |      37280.730 |              0.003 |                  nan |
