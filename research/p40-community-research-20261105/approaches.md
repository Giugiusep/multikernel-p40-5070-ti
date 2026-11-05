# Qwen Flash Next on P40: community evidence and local implications

Researched 2026-11-05. Model scope: Qwen3.8-Flash-Next GSQ-RCO IQ2_XS; dense27B and DeepSeek results are not equivalent evidence. No runtime/configuration changes made during research.

## Direct reports

1. Same IQ2_XS artifact, one/two P40s:19.6/34.8tok/s short-prompt medians, with substantial published ranges. Layer split, CUDA12 SM61 build, host RAM. Reproduction and source provenance:
https://github.com/ooandychen/strata/blob/main/bench/results/2026-10-05-community-2x-p40/README.md
https://github.com/Niko1221/Strata/pull/1028

2. Single P40 passthrough Proxmox VM: SM61 mode8 and dense-kernel changes; report lists approximately34–35→37–38tok/s short-prompt decode. **Different IQ3_S quant and experimental speed projection enabled**. Not an exact-model throughput comparison with us. Kernel-level optimizations can be considered independently:
https://github.com/Niko1221/Strata/blob/main/bench/results/2026-10-12-p40-sm61-kernels/README.md

3. Two P40 IQ3_S report:about35tok/s,131K configured context. Extensive resident expert cache,layer split,MTP,int8 KV; calibration modestly improved decode. Different quant/hardware workload:
https://github.com/T8mars/Strata-T8/blob/main/bench/results/2026-10-10-community-2x-p40-iq3_s/README.md

## Implementation references

Strata SM61 defaults: mode8 resident experts,interleaved small-batch matvec table,wide dense dequantization. Optional FP32 cuBLAS GEMM needs parity qualification against the actual library. Dense/prefill tuning does not accelerate our expert-only P40 worker directly. Do not transplant P100 half arithmetic or Volta Tensor Cores.
https://github.com/Niko1221/Strata/blob/main/docs/OLDER_GPUS.md

Strata helper architecture keeps dense/KV/MTP primary, caches experts on helpers, overlaps CPU work, returns through pinned staging. Optimized helper path learns routing and reduces outputs before returning, but changes summation order. Those flags assume devices visible to one engine; they cannot discover a GPU owned by another kernel.
https://github.com/Niko1221/Strata/blob/main/docs/SECOND_GPU.md

Cross-process AMD helper prototype is architecturally relevant: primary CUDA + helper process + expert ownership + bounded shared messaging. Its reported speedup largely removes SSD expert reads on a RAM-limited machine. Its memfd cannot span our kernels unchanged; VSOCK/shared Multikernel transport remains necessary. Not an SM61 validation.
https://github.com/Niko1221/Strata/issues/3127

PXA targets SM61 as well as SM60/70, with packed formats,expert caching and hardware-based dispatch. Published Flash Next evidence examined is primarily P100,not a reproduced5070Ti/P40 Multikernel result. PXQN requires a compiled library in source builds; some artifacts are gated. It is an alternative quant/runtime study,not a drop-in acceleration of current IQ2_XS.
https://github.com/poisonxa16/pxa

Independent P40 bakeoff also reports temperature0 nondeterminism in Strata. This reinforces measuring correctness and speed separately; it does not diagnose our exact divergent token.
https://github.com/sarge18/p40-llm-engine-bakeoff

## Local code comparison

Pinned native CUDA source c62d5e3a15b40ea51c96aae9e8d36928965f53ab contains SM61 mode8 dispatch in src/kernels/cuda/iq_kernels.cu. Worker constructor already sets native mode8. However, qualified CPU_EXACT_IQ2 branch calls mk_exact_iq2_gu_launch and mk_cpu_pool_down_launch instead of native grouped execution. Thus setting mode8 again will not accelerate the actual qualified path. Our custom math preserves CPU Q8_K input,CPU SwiGLU/ActQ and CPU reduction ordering; native GPU path uses a different arithmetic contract and previously diverged. No safe claim that simply updating the engine fixes that.

Current custom worker has intermediate GPU→CPU→GPU staging and returns individual expert rows. Existing512MiB/256-pair caps are experiment safeguards,not P40 capacity. Last test used84 pairs at128 selection; reports commonly keep thousands resident. Capacity alone did not establish a speedup in our tests.

## Proposed order

1. Isolated SM61 kernel study: adapt native mode8 packed decode/coalescing/DP4A principles to our CPU-Q8_K exact contract. Compare against actual-pool oracle with real routed inputs,not generic dequantization alone. Measure by expert count and verification width; retain previous worker fallback.
2. Place experts in layer-local groups rather than sparsely spreading small hits across48 layers. Compare CPU time removed against remote wall time added. Sending one request for a useful group is preferable only if measured faster.
3. Grow resident cache in guarded stages (e.g.512MiB→2GiB→4GiB), checking actual VRAM and leaving headroom. Stream individual bounded blobs from primary; do not restore model FUSE bridge or duplicate the full RAM arena. Warm-load time measured separately.
4. Preserve exact primary accumulation initially. GPU partial-sum return is a separate numerical-contract experiment; do not call it bit-identical.
5. If expert-only helpers still lose on latency, investigate a complete layer-stage worker with one activation handoff per stage/window. This needs Qwen recurrent state,attention and MTP verification/rollback handling. Upstream same-process layer split is a reference,not a working cross-kernel implementation. No topology/driver move required by the proposal.

No directly reproduced5070Ti+P40 Multikernel recipe was found in the sources reviewed. Community card counts and advertised throughput must not be treated as controlled predictions for our VM.
