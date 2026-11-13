# Exact P40 expert compute experiment, 2026-11-13

This is DeepSeekV4.1 IQ2_XS, not the known-good Nemotron configuration. Experimental primary branch experiment/shiba-deepseek-exact-full: a5e2114 adds exact full-matvec execution; b0e1ead adds opt-in single-request dispatch. Neither is deployed to the API.

## Correctness architecture

The earlier full replacement used native GPU accumulation/activation arithmetic that did not match actual CPU expert contributions. The qualified down kernel was generalized without changing its integer groups, FP32 FMA/reduction order or final scale: K5120/N2304 for gate/up, K2304/N5120 for down. Weights stay packed IQ2_XXS/IQ2_XS/IQ2_S on SM61, no Tensor Cores, no FP16 dequantized model. Unsupported IQ3 down formats and widths>8 remain CPU.

Port6013 two-request path: primary lossless actual CPU Q8_K input -> GPU exact gate/up -> primary CPU fmin/fmax/clipped SiLU/std::exp and Q8_K -> GPU exact down -> primary combines in original order. DS4_MK_EXPERT=full-exact-shadow requires raw output bytes equal actual CPU-lane results; full-exact substitutes only successful entries and retains CPU fallback. Existing modes unchanged.

Independent ggml-cpu oracle checked three real weight-format combinations x widths1..8:24 cases, all three matrix-product stages bit-exact. Real128MiB shadow gate:36 code +26 prose comparisons with zero differing bits; actual32+32 token outputs exact. Persistent22GiB full-weight cache then matched64-token CPU reference on all three requests. This validates these prompts/shapes, not arbitrary contexts or sampling.

The adapter still reads primary expert-store slots and performs remote entries synchronously before CPU remaining entries. It does not overlap independent CPU/P40 experts or suppress primary reads of full remotely resident blobs.

## Full-model warm results

Same64-token code prompt,21-token input, maximum context8192, all6 experts, CPU8/cache32GB, DSpark5/conf0.65, greedy42, zero prompt reuse verified from DONE. Persistent process: one cold and two warm requests.63 generated intervals/decode_ms is the reported rate.

|campaign|CPU-only warm tok/s|P40 warm tok/s|
|---|---|---|
|two-request|4.0333 /5.0205|3.0656 /3.4710|
|one-request|4.4057 /4.0450|3.8858 /3.4782|

CPU control means4.5269 and4.2254 respectively; P40 means3.2683 and3.6820. Only two warm samples each with visible storage variation. No accepted overall speedup; production remains unchanged. Both P40 runs held23613972480 bytes (~21.99GiB) of complete gate/up/down weights.

Two-request warm requests each made18134 calls and transferred689406792 protocol bytes, including prompt+decode. GPU event time~8.98s was included within17.08–18.27s RPC time. One-request requests each made9067 calls and transferred382652404bytes (~44.5% reduction); GPU~9.14s was included within14.67–14.91s RPC. These whole-request counters include prompt and decode and are nested, not an additive per-token critical-path decomposition.

## DP4A candidate (rejected as a global replacement)

Private port6014 candidate replaces four scalar integer products with SM61 __dp4a on signed packed grid bytes, then multiplies their common integer factor. A deterministic table audit checked all three IQ2 grids fit signed int8 and worst-case int32 accumulation cannot overflow.23 repeated checks per case (3 warmup +20 measured) on each of24 cases per worker remained exact. M1 gate+up GPU microseconds: IQ2_XXS471->587 (regression), IQ2_XS582->568, IQ2_S502->283. IQ2_XXS M8 regressed >2x. Therefore NOT used by the full-model one-request path or selected globally. Synthetic improvements to IQ2_S are not inference claims. Combined microbenchmark timing includes Python packing/reference-file reads; CUDA event times isolate kernels.

## One-request exact path

Private port6015 worker retains the original exact matvec kernel. GPU gate/up results are copied to secondary host memory, which runs CPU-compatible clipped SiLU and signed-max Q8_K quantization; GPU exact down follows. DS4_MK_FUSED=1 uses this path only with full-exact/full-exact-shadow; old two-request behavior remains available. This is all three large matmuls on P40, NOT an all-GPU activation implementation.24 isolated complete outputs exact; real code/prose shadow again36+26 exact comparisons, then32+32 exact substitutions. Persistent3x64-token outputs exact, zero prompt reuse.

## Preserved state and next dependency work

Original binary SHA e3e4e048b87649462b6a74ef4ec18a600b436141877af158b3555535765b9622 preserved at the original build path. Original workers6011/6012 and RPC5000/5002 retained; new test workers6013/6014/6015 stopped after experiments, P40 idle601MiB. API restored to original active qwen3.8-flash-next-strata. Kernels #12-rxsafe1, NVIDIA stacks and all primary write pacing unchanged. No primary ring-full/RCU/Xid events since19:09Z; secondary dmesg unchanged. No secondary reset/unload. CUDA sm61 worker and sm120 coordinator built; HIP/SYCL not run for this CUDA-only experiment.

Next useful scheduling experiment: independent CPU and remote experts can contribute to the same MoE output, but the current adapter serializes them. Before overlap, snapshot ALL remote FP32 inputs: CPU quantization aliases the original c_xh buffer. Preserve disjoint slot ownership, exact entry identity, final combine order, CPU scratch ownership and failure recovery ordering. Separately, suppressing full primary reads for remotely resident experts requires a full-role lazy-refill/failure gate, beyond the previously validated down-only deferral. Neither change is implemented here.
