# P40 large-cache experiment, 2026-11-13

Opt-in port6012 worker and client branch experiment/shiba-deepseek-large-cache. Original6011 worker, RPC5000/5002, primary pacing, kernels, NVIDIA and API settings unchanged. Original private coordinator binary restored byte-for-byte (SHA e3e4e048b87649462b6a74ef4ec18a600b436141877af158b3555535765b9622).

## Capacity and layout

The previous1GiB worker and client limits were experimental software limits, not P40 capacity. Client now accepts explicit16..22528MiB budgets with the old1024MiB default and optional DS4_MK_PORT (default6011). Private worker6012 allocates a22GiB device arena per connection and places weights at256-byte aligned offsets. All real expert byte sizes are alignment-checked. No materialized dequantized weights and no corresponding host allocation. CLEAR reuses the arena; disconnect frees it. It reserves22GiB even for the512MiB control, so payload residency and NVML reservation must not be conflated. Original worker remains available.

The initial separate-cudaMalloc-per-expert attempt failed with CUDA out-of-memory after the17.9GiB progress checkpoint; the connection freed its allocations, and no new GPU/kernel error occurred. The packed arena then loaded6927 repeated real IQ2_XS down-weight fixtures:23620515840bytes (21.998GiB). Three first/middle/last computation results were bit-exact against CPU. NVML showed23133MiB used/1307MiB free; secondary RAM930MiB used/628MiB available. This is a synthetic capacity test, not real model residency. All24 format/width cases also passed bit-exact. CUDA sm61 worker and CUDA sm120 coordinator built; HIP/SYCL not tested (experiment uses the existing Linux CUDA-only adapter).

## Actual model results

Same64-token code prompt, Nemotron infrastructure untouched, DeepSeek six experts/CPU32GB/DSpark5/conf0.65/greedy42, original primary write pacing. Every run matched the CPU reference exactly.

|budget|actual packed model weights|first-pass generation|
|---|---:|---:|
|512MiB|535265280B (0.498GiB)|4.73tok/s|
|8GiB|8588851200B (7.999GiB)|1.98tok/s|
|22GiB|14538700800B (13.54GiB)|0.66tok/s|

The short workload encountered4391 distinct admitted experts at the largest budget; it did not need22GiB of distinct weights. Model load/cold decode includes large uploads: largest-budget decode had77.065s LOAD RPC and12.502s down-RPC, compared to512MiB0s LOAD and0.283s down-RPC. These are nested request timers, not additive GPU critical-path proof.

Persistent --serve replay, same64-token prompt, zero prompt reuse verified in DONE, two warm requests:

|budget|warm1|warm2|mean tok/s|
|---|---:|---:|---:|
|512MiB|4.7045|5.0450|4.8748|
|22GiB (13.54GiB real weights)|4.4057|4.4915|4.4486|

All six persistent requests were exact. Throughput is63 generated intervals divided by decode milliseconds, matching the CLI convention. Two warm samples do not establish broad performance. No gain demonstrated; retain baseline.

The first persistent harness completed valid512MiB measurements but timed out on shutdown because it did not close stdin after QUIT. The separate large-cache repeat closes stdin; it completed and restored the prior active qwen3.8-flash-next-strata API profile. Initial manual/out-of-memory attempts remain recorded as failures, not passes.

## Implication

Large VRAM residency is technically working. The qualified path offloads only down projections. Gate/up computation and much expert disk traffic remain on CPU. More admitted experts also create many more per-expert RPCs (cold decode down calls373 at512MiB vs10426 at22GiB). A larger cache alone is not an accepted speed optimization. Further work should evaluate batching RPCs and independently correctness-qualifying full-expert GPU execution rather than assuming VRAM residency alone removes CPU work. Full-expert replacement remains unqualified; no production promotion.

Final logs: no new primary ring-full/RCU/Xid reports since18:38Z; secondary dmesg unchanged. P40 returned to601MiB idle use. API restored. Both source branches clean after focused experimental commits.
