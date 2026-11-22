# End-to-end: CPU codec vs CUDA codec

encode = total codec time in the SSD writer thread; load+read = SSD read + decode + host-to-GPU cache assembly, summed over the run (s).

| config    | policy        |   reps | bytes_identical   |   ssd_hits |   encode_cpu_s |   encode_cuda_s |   load_plus_read_cpu_s |   load_plus_read_cuda_s |   ttft_delta_pct |   latency_delta_pct |   latency_noise_pct |   wall_cpu_s |   wall_cuda_s |
|:----------|:--------------|-------:|:------------------|-----------:|---------------:|----------------:|-----------------------:|------------------------:|-----------------:|--------------------:|--------------------:|-------------:|--------------:|
| default   | reuse_int8    |      3 | True              |        391 |          5.632 |           1.323 |                 13.729 |                   4.545 |          -47.006 |             -12.630 |               1.611 |       75.220 |        65.720 |
| default   | spill_int4    |      3 | True              |        414 |          9.558 |           1.686 |                 17.578 |                   3.555 |          -60.452 |             -18.780 |               2.542 |       78.204 |        63.517 |
| default   | spill_int8    |      3 | True              |        414 |          7.634 |           1.946 |                 15.976 |                   4.836 |          -51.949 |             -16.194 |               1.778 |       77.835 |        65.231 |
| default   | write_through |      3 | True              |        414 |          0.541 |           0.494 |                 12.541 |                  12.634 |            2.275 |              -0.077 |               3.034 |       73.557 |        73.525 |
| qwen_long | reuse_int8    |      3 | True              |       2675 |          0.929 |           1.030 |                  3.644 |                   2.243 |          -14.026 |              -0.212 |               2.975 |       72.494 |        72.340 |
| qwen_long | spill_int4    |      3 | True              |       2696 |          1.673 |           0.916 |                  4.990 |                   1.705 |          -32.743 |              -6.266 |               2.837 |       75.515 |        70.783 |
