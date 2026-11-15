# KV offload SSD wear report

- write counter used: `linux_bytes_written`
- baseline: `write_through`; speed metric: `latency_mean_s`
- wear projection: 50 chats/day, SSD rated 300 TBW
- idle background writes subtracted: 1.448 GB/hour

| policy        | family         |   runs |   gb_per_1k_tokens |   gb_per_chat |   writes_saved_pct |   slowdown_pct |   ttft_mean_s |   speed_mean |   years_to_tbw |   pct_tbw_per_year |   virt_amplification |
|:--------------|:---------------|-------:|-------------------:|--------------:|-------------------:|---------------:|--------------:|-------------:|---------------:|-------------------:|---------------------:|
| write_through | baseline       |      3 |              0.119 |         0.298 |              0.000 |          0.000 |         0.224 |        1.203 |         55.244 |              1.810 |                1.640 |
| spill         | baseline       |      3 |              0.116 |         0.289 |              2.737 |          0.303 |         0.227 |        1.206 |         56.799 |              1.761 |                1.517 |
| reuse         | reuse-aware    |      3 |              0.055 |         0.137 |             53.976 |          1.528 |         0.225 |        1.221 |        120.035 |              0.833 |                2.767 |
| reuse_int8    | reuse-aware    |      3 |              0.028 |         0.071 |             76.229 |          6.450 |         0.246 |        1.280 |        232.405 |              0.430 |                2.238 |
| oracle        | oracle (bound) |      3 |              0.022 |         0.054 |             81.929 |         -1.987 |         0.215 |        1.179 |        305.713 |              0.327 |                2.459 |
| no_ssd        | baseline       |      3 |              0.000 |         0.000 |             99.957 |          1.281 |         0.226 |        1.218 |     128905.906 |              0.001 |              nan     |
