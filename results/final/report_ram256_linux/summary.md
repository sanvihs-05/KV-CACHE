# KV offload SSD wear report

- write counter used: `linux_bytes_written`
- baseline: `write_through`; speed metric: `latency_mean_s`
- wear projection: 50 chats/day, SSD rated 300 TBW
- idle background writes subtracted: 0.917 GB/hour

| policy        | family         |   runs |   gb_per_1k_tokens |   gb_per_chat |   writes_saved_pct |   slowdown_pct |   ttft_mean_s |   speed_mean |   years_to_tbw |   pct_tbw_per_year |   virt_amplification |
|:--------------|:---------------|-------:|-------------------:|--------------:|-------------------:|---------------:|--------------:|-------------:|---------------:|-------------------:|---------------------:|
| write_through | baseline       |      3 |              0.032 |         0.100 |              0.000 |          0.000 |         0.348 |        1.410 |        164.884 |              0.606 |                1.998 |
| spill         | baseline       |      3 |              0.031 |         0.098 |              1.842 |         -0.774 |         0.335 |        1.400 |        167.977 |              0.595 |                1.562 |
| reuse         | reuse-aware    |      3 |              0.022 |         0.070 |             30.144 |         -3.704 |         0.305 |        1.358 |        236.035 |              0.424 |                1.820 |
| oracle        | oracle (bound) |      3 |              0.019 |         0.060 |             39.696 |         -4.147 |         0.300 |        1.352 |        273.422 |              0.366 |                1.766 |
| spill_int8    | quantize only  |      3 |              0.016 |         0.051 |             49.127 |         -2.401 |         0.313 |        1.377 |        324.109 |              0.309 |                1.701 |
| reuse_int8    | reuse-aware    |      3 |              0.012 |         0.036 |             63.853 |         -3.465 |         0.303 |        1.362 |        456.147 |              0.219 |                1.893 |
| spill_int4    | quantize only  |      3 |              0.009 |         0.028 |             71.990 |          0.581 |         0.364 |        1.419 |        588.665 |              0.170 |                1.837 |
| reuse_int4    | reuse-aware    |      3 |              0.006 |         0.020 |             80.013 |         -1.098 |         0.331 |        1.395 |        824.944 |              0.121 |                0.221 |
| no_ssd        | baseline       |      3 |              0.000 |         0.000 |             99.846 |          8.295 |         0.299 |        1.527 |     107115.762 |              0.001 |              nan     |
