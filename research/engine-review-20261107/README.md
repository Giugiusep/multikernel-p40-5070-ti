# Speedy and DeepSeek/P40 engineering review — 2026-11-07

Production API, binaries, NVIDIA drivers, kernel, write pacing and fallback listeners are unchanged. This directory contains isolated research, not a deployment.

## Sources and pinned revisions

* Speedy: https://github.com/DevDock-AI/Speedy at 1c7a0b629b4f6a318480e6bb3692b595ff864334; inspect SPEEDY.md and docs/TUNING-HISTORY.md.
* Siege-Ladder: https://github.com/MasherDon/Siege-Ladder at 484d9b77476f927a08e4164e7f955139910dd6e4; inspect docs/techniques.md and docs/switches.md.
* Author report: https://www.reddit.com/user/MasherDon/comments/1xf906y/deepseek_v41flash_1017_at_20_toks_on_a_single_16/
* Local DeepSeek base: compile-manifest.json, preserved source checkout Strata-dsv41-experiment. Current upstream ds4_iq.cu was also inspected online; it still includes WMMA code without a Pascal build guard.

## Useful ideas, with correctness distinctions

Speedy remains primarily a Qwen RTX engine, documented CUDA minimum SM80. It does not supply a turnkey Pascal or Multikernel backend. Its smaller prefill scratch, KV prefetch and context-sensitive cache planning merit RTX-only tests. Its verification CPU deferral changes contribution ordering and requires separate numerical qualification; do not copy it blindly into our exact baseline. Its measured machine has much more RAM bandwidth than this VM. Upstream GPU expert-cache correctness in our deployment remains a separate gate. Do not simply enable all defaults or disable our diagnostic cache bypass.

Siege-Ladder's author reports about20tok/s on a5070Ti,64GBRAM and two NVMes. This uses another quant mix and a modified llama.cpp engine, not our existing AtomicChat pack/Strata engine. Even its default preset includes a lossy low-weight expert drop rule and YoCo truncation; exact is a separate preset. The faster23–25tok/s rows add more approximations. Rates are author reports, not reproduced locally. Useful exact directions: bounded VRAM/RAM/NVMe tiers; mapped pinned router/reply mailboxes; fewer CPU wakeups and graph splits; overlap copies with ready experts; prediction used solely for prefetch, with all actual routed contributions still computed. Windows unbuffered-file findings are not automatically Linux improvements.

## Local DeepSeek boundary

Pack metadata:40layers,384experts/layer,top6,hidden5120,FFN2304. expert-layout.json records actual per-layer formats and complete expert sizes9,123,840–11,335,680bytes (expert0 per layer). Gate/up and down are independently mixed IQ2_XXS/IQ2_XS/IQ2_S, with an IQ3_XXS down layer. The qualified Qwen worker assumes hidden2560,FFN640,top10 and Q2_0 down; it cannot accept these blobs or activation semantics unchanged.

The isolated STRATA_DS4_SCALAR_ONLY build excludes WMMA declarations/dispatch while retaining existing scalar and DP4A paths. compile-result.json confirms an sm_61 ELF, DP4A instructions and zero HMMA/IMMA instructions. The existing independent iq_expert_test ran on P40 and passed all four actual expert formats (types16,17,22,18), including token gate/up, token down, set/prompt fallback and CPU reference checks. These are tolerance checks, not byte-exact equivalence to the deployed RTX engine. Mixed-role real-model validation remains outstanding. It is not a complete DeepSeek SM61 build, model correctness or speed measurement. Never run the full model in secondary RAM or revive the disabled FUSE model bridge.

## Implementation order

1. Run existing iq_expert_test independently against ggml dequantization/double arithmetic on P40, all actual expert types. Include grouped rows, prompt fallback and repeated runs. Then add real mixed gate/down fixture coverage from this pack; homogeneous type tests alone are insufficient.
2. Create a separate versioned DeepSeek worker contract, not a Qwen protocol reinterpretation. Keep primary ownership of Engram, NVMe reads, drafting and routing. Explicit shapes, independent gate/down type IDs, quantized activation format, bounds and error replies. Preload a bounded hot set, initially<=1GiB VRAM, and reject uncached IDs. No synchronous multi-megabyte cache fills on the token critical path.
3. Shadow actual DeepSeek expert requests against the deployed engine's arithmetic; qualify sparse/grouped routing and output accumulation. Retain all six routed experts, current no-expert-cache settings and existing DSpark options for the first comparison.
4. Measure resident hit rate, saved RTX weight-upload bytes, P40 compute, request/return and wait time. Only expand cache after exact shadow passes and resource limits remain safe. A P40 helps only when avoided CPU/SSD/RTX-upload work exceeds added service/wait costs.
5. Full model paired RTX-only/P40 tests, same prompt IDs, quant, live context and speculative policy. Record actual committed tokens, acceptance and end-to-end decode. No generation-speed claim from an isolated kernel probe.

Separate Qwen next step: remove duplicate Q8_K activation quantization by sharing the existing CPU buffer with the packed request, then exact shadow and paired benchmark. This review has not implemented or measured that change.

## Completed P40 probe

Executable SHA256:83ae980e24893cb5afe161844954bf26d18ec06d15b6672a849aa7beb90dbf91. Paced22,688,024-byte executable transfer, hash verified in secondary. Existing runtime loader/libraries used; no driver/module/kernel modifications. iq_expert_test returned0 and ALL OK for types16,17,22,18. Full raw output:p40-results.log. Maximum token gate/up normalized error5.21e-8; token down4.60e-8, within test tolerances. Prompt/set F16 tolerances are wider and are reported explicitly in the raw log; do not label this bit-exact.

The inherited --bench diagnostic also ran (six experts,5120x2304):959–1374us per layer across types. Its40-layer extrapolation is not a full-model runtime. The inherited benchmark fills arbitrary weight bytes without repairing scales, uses fixed independent down activations and lacks timing-output correctness checks. These figures are exploratory only and unsuitable as a performance gate; do not infer model tok/s. Build no longer uses an incompatible deployed CPU archive: standalone probe links the matching source CPU support and existing ggml libraries. This avoids changing the production archive.

Next performance harness must use finite legal or real packed weights, real mixed gate/down formats, correctness before/after timing, warmed clocks and repeated median/p95. The existing generic correctness test has cleared the first format gate; it has not cleared the full inference gate. No model benchmark or P40 speedup is claimed.
