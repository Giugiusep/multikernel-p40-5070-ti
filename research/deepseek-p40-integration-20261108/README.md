# DeepSeek P40 integration checkpoint — NOT deployment-qualified

Primary source branch `experiment/shiba-deepseek-decode-release`, commit `6c125cf`; deployed release base `99866a47a`. Source change is default off. `DS4_MK_EXPERT=shadow` compares actual CPU-lane expert contributions; `replace` experimentally substitutes worker contributions. Neither is enabled in the production API.

Primary retains all model files, Engram, six-expert routing, shared experts, DSpark5/conf0.65, original cache32/CPU8 and context8192. Secondary owns only a bounded 1GiB GPU expert cache, <=16MiB frames and widths1–8. No model filesystem/secondary RAM expansion, driver changes, kernel changes or changes to RPC5000/5002. Worker6011 contract `deepseek-iq-q8k-v2` is separate from original worker6010. AF_VSOCK option9/type1, 32KiB/100us pacing, partial I/O and bounded timeouts retained.

## Correctness findings

1. Original worker per32 activation format differs from CPU lane Q8_K. Real-input shadow had max relative-L1 .0251479511. All32code output tokens matched because CPU outputs were still used. Replacement was not attempted for this contract. Evidence `q8-32-shadow`.
2. New worker preserves actual CPU input Q8_K codes and repeats its signed per256 scale into the native GPU kernel's per32 view. Intermediate hidden activation uses matching signed-max, -127/max, nearest-even Q8_K quantization. Weights remain packed; scalar/DP4A SM61 native IQ kernels reused. No Tensor Core implementation.
3. Independent real expert0 mixed-format oracle:10format combinations x widths1–8 =80cases,480verified repeats. GU/down against float64 decode, intermediate quantization codes/scales exact, batched vs individual rows bit-equal. Maximum complete-path relative-L1 1.6775e-7. This does not prove full-model correctness.
4. Initial real-engine Q8_K run idled out between calibration and inference. CPU fallback preserved output. Fixed worker receive timeout10→120s and suppressed remote participation during hardware calibration. Earlier results preserved under `q8k-idle-timeout`.
5. Corrected real-engine shadow:117resident experts,274comparisons, max relative-L1 .000178128796;32committed code tokens match control. Control4.68tok/s,shadow4.82tok/s. Shadow is not a speedup claim.
6. Actual replacement:4.17tok/s, first committed mismatch at token1, later socket I/O failure/CPU fallback. Not accepted, no prose/full performance promotion. Exact cause of divergence is unresolved.
7. Static inspection also found entry lookup used expert ID alone, which is insufficient when a long prompt splits one expert into several slot/token groups. Fixed to match ID/count/slot/token rows. Host regression test passes and CUDA primary rebuild passes. This correction was NOT GPU-retested and is NOT established as the cause of token1 divergence.

## Recovery state

After replacement the secondary stopped answering console and both5000/6011connect probes timed out. A fresh console attach followed by writing returned ENOSPC. Instance sysfs still says active. Primary kernel warning history showed no new warnings at current uptime; old boot warnings/driver mismatch were present. Secondary pre-failure dmesg tail was the old758.641s ring-full burst. Unable to read post-failure secondary dmesg. This is consistent with a stalled/full Multikernel messaging path, but exact lockup root cause and NVIDIA state are unconfirmed. Do not classify as an identified NVIDIA or outer-KVM failure.

No secondary kill/reset/module reload attempted. Primary API automatically restored, health ok and active qwen3.8-flash-next-strata. Other P40-dependent services cannot be assumed usable while secondary is unresponsive. No Funnel changes or external validation. VM reboot needed before further P40 work; do not reboot automatically.

## Next gate after recovery

Keep source adapter off while checking original services and NVIDIA state. Relaunch only private worker6011 via preserved runtime after paced/hash-checked transfer. First add replacement-with-CPU-reference diagnostic/forced-token logits to locate first divergence, and instrument request/sequence/timestamps to locate secondary failure. Rerun exact32code and prose gates before longer correctness/performance tests. Do not expand cache or deploy based on these failed replacement results. Current adapter waits for existing StoreHold reads, so it does not yet avoid primary SSD reads on P40 cache hits; no claim of exploiting all24GB or saving disk traffic.

Original API/executable/configs remain untouched. Primary patch, worker source, build recipe, oracle and failure evidence are committed separately from binaries/model fixtures. CUDA sm120/SM61 builds tested; HIP/Metal not tested.
