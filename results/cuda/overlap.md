# Writer/model overlap (CUDA streams + pinned memory)

30 model steps; 2 OPT-1.3B blocks (12 MiB each) encoded to int8 per step.

model time = until the last model step finishes; total = until every block is also encoded.

| schedule | model time (s) | total incl. drain (s) | model slowdown | total vs model-only |
|---|---|---|---|---|
| model only (no KV writes) | 0.612 | 0.612 | +0.0% | +0.0% |
| serial | 0.911 | 0.911 | +48.9% | +48.9% |
| cpu-thread | 0.615 | 1.427 | +0.6% | +133.4% |
| gpu-stream | 0.624 | 0.625 | +2.1% | +2.2% |
