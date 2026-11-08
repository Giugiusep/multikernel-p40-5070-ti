# Decode RPC accounting

Debug source commit e92544d, default off. DS4_MK_PROFILE records cumulative request wall time, CUDA event times already returned by the worker, and host preparation time. Model::stats checkpoints separate prompt registration from decode. No new CUDA synchronizations or per-chunk logging.

64-token code run, 512 MiB original three-role cache:

- Decode: 97 down requests, 2,869,852 application-protocol bytes, 63.501001 ms request wall time, 29.658080 ms worker GPU events, 0.520354 ms host preparation. No registration during decode.
- Per 63 generated-token intervals: 1.008 ms RPC, 0.471 ms GPU events, 0.0083 ms preparation. GPU events exclude H2D/D2H, and RPC wall time includes transport/worker staging; these categories overlap and must not be added.
- Controls 4.49 / 5.13 tok/s; profiled offload 5.26, unprofiled offload 4.42. Outputs exact. Variation prevents a precise overhead or speedup claim. Timers do not explain the much larger CPU/I/O variation.

The measurement motivates avoiding redundant primary weight reads, rather than further ordinary GEMV or socket tuning. Production unchanged. Benchmark caps and automatic API restoration match preceding experiments.
