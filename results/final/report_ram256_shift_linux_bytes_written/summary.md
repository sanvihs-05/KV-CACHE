# KV offload SSD wear report

- write counter used: `linux_bytes_written`
- baseline: `write_through`; speed metric: `latency_mean_s`
- wear projection: 50 chats/day, SSD rated 300 TBW
- idle background writes subtracted: 2.345 GB/hour

| policy     | family      |   runs |   gb_per_1k_tokens |   gb_per_chat |   ttft_mean_s |   speed_mean |   years_to_tbw |   pct_tbw_per_year |   virt_amplification |
|:-----------|:------------|-------:|-------------------:|--------------:|--------------:|-------------:|---------------:|-------------------:|---------------------:|
| reuse      | reuse-aware |      3 |              0.018 |         0.058 |         0.230 |        1.014 |        285.417 |              0.350 |                3.336 |
| reuse_int8 | reuse-aware |      2 |              0.010 |         0.030 |         0.263 |        1.031 |        548.595 |              0.182 |                4.861 |
