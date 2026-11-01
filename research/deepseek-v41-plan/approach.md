# DeepSeek V4.1 Flash IQ2_XS: local execution approach

Research date: 2026-11-01. Recommendation: use a separate current Strata DeepSeek engine on the RTX5070Ti first. Do not update the working Qwen engine in place, change NVIDIA/Multikernel, move the P40 or expose an untested model through the shared API.

## What the 12 GB claim means

AtomicChat publishes the edited weights; its model card directs this model to Niko1221/Strata. The full weights do not reside on a12GB GPU: the engine keeps a working subset in VRAM/RAM and reads missing data from storage. The downloaded four-shard IQ2_XS set is not the complete runnable bundle: the shared Engram lookup table is separate. [AtomicChat model card](https://huggingface.co/AtomicChat/DeepSeek-V4.1-Flash-1017-Uncensored-GSQ-RCO-GGUF).

The newer Strata tree has a distinct `strata-dsv41` executable; our existing deployed Qwen Strata fork does not establish DeepSeek support. Upstream supplies Linux/CUDA builds including SM120. [DeepSeek engine documentation](https://github.com/Niko1221/Strata/blob/c62d5e3a15b40ea51c96aae9e8d36928965f53ab/docs/DEEPSEEK41.md).

## Checked on this VM

- RTX5070Ti: previously CUDA-verified16GB-class GPU; no GPU workload started by this investigation.
- Guest RAM: about50GiB total,46GiB available at inspection;8GiB swap. The guest exposes12logical CPUs, AVX512, and Ryzen79700X identity. VM topology is not proof of12physical host cores.
- Disk: about118GiB free after the144.565GiB download. Guest sees a QEMU HARDDISK, not a physical NVMe identity. No Proxmox host inspection/modification was performed.
- Read-only direct-I/O spot tests:1GiB sequential in1.241s=0.806GiB/s;256MiB scattered across64aligned4MiB reads in0.659s=0.380GiB/s. Short guest-path observations, not physical-drive limits, sustained benchmarks or4KB Engram lookup latency. Host caching and other activity remain uncontrolled. Raw results: guest-storage-read-probe.json.
- All four local GGUF headers parsed without loading tensors. Architecture deepseek41, context metadata1048576. The120routed-expert tensors comprise76IQ2_XXS,16IQ2_S,27IQ2_XS and1IQ3_XXS; none is MXFP4. Required Engram token-map metadata is absent in the first shard. Raw results: local-model-inspection.json. Context metadata is not evidence this machine can serve a1M prompt.

## Storage plan: reuse the download

At the measured free-space snapshot:

| Addition | Bytes | Role |
|---|---:|---|
| Engram IQ4_NL |110595290816| Required lookup weights |
| Metadata GGUF |5780589| Engram hash/token-map parameters |
| DSpark Q8 drafter |7976914240| Optional later speculative experiment |

The Engram currently listed by ISTA has SHA256 `3dac4c54893ad7aa89103c0028500eadc78e74112cef3a2930a97e67dc9fb4f3`, revision `65d53bc8be65123e255ff02f3ded958b6dbf54fd`. Pin a consistent engine/companion set and verify every download. Upstream supports more than one Engram layout/hash; do not silently mix assumptions. Existing AtomicChat shard provenance stays authoritative for its edited files, rather than pretending they match official unedited shard hashes.

After Engram+metadata, calculated free space is14.78GiB. With DSpark too,7.35GiB remains, before engine/build/logs. Arithmetic checked in code. A text-only, no-drafter first run may fit without deleting anything; the complete experiment has little headroom. Prefer reclaiming or adding25–50GiB before DSpark/build expansion, with explicit approval before deleting any model. No extra large downloads were performed in this investigation.

The packer writes tensor filenames/offsets, geometry, tokenizer and small Engram hash metadata; it does not duplicate155GB of weights. Its `engram.bin` is not a second copy of the111GB lookup table. Use the plain packing recipe and preserve edited dense matrices. [Pinned packer](https://github.com/Niko1221/Strata/blob/c62d5e3a15b40ea51c96aae9e8d36928965f53ab/tools/dsv41/pack_gguf.py).

## Experiment sequence and acceptance gates

1. Create an isolated pinned Strata checkout/build/config under research/Strata-dsv41-experiment; leave our deployed fork, Python environments and API configurations alone. Pin the source/submodule revisions. Prefer a verified engine package after checking its manifest and dynamic-library dependencies. Installed toolkit is CUDA12.8; published builds use CUDA13. Do not replace drivers to resolve that difference: check compatible separate runtime libraries or build requirements first. Self-build target: strata-dsv41, CUDA architecture120, native experts enabled.
2. Download only missing Engram and metadata initially; check exact sizes/hashes and create an index-only pack referencing existing shards. Run --pack-info and reject missing/unsupported tensors. Do not use tensor skipping/dense overrides as a workaround.
3. Schedule GPU ownership through the existing API lifecycle: finish requests, unload the resident model and confirm no native Studio loader occupies RTX. Leave P40 and Multikernel untouched. Run the experiment on a separate loopback port, e.g.19085; no Funnel/API catalog publication yet.
4. Initial baseline: one request, RTX only, text only,8K context, all6routed experts. Explicit --experts6 is important: upstream's Faster/auto mode can select5on cards at or below16GiB, changing model computation. Use --dspark none initially: source parsing confirms this avoids loading the drafter; off disables drafting but still loads it. [Pinned flag parsing](https://github.com/Niko1221/Strata/blob/c62d5e3a15b40ea51c96aae9e8d36928965f53ab/src/dsv41/ds4_main.cpp).
5. Start with28–32GiB RAM cache, reserve roughly8GiB for OS/services, and measure total process+file-cache use. Budget includes more than expert cache; do not assume mapped weights are free. Use scoped cgroup limits with no swapping for this experimental workload after confirming support, and fail cleanly rather than freezing the VM. Guest memory controller is exposed and io_uring is not globally disabled; actual ring registration still needs a runtime check.
6. Correctness before speed: deterministic smoke prompts, repeated output, tokenizer/template checks, non-repetitive coherent answers and tool-call parsing. Where available compare target logits/tokens against an independent reference for the same quant. An answer naming Paris is only a smoke test. Record any routing/precision approximation; never present5expert output as the6expert baseline.
7. Benchmark4K/8K/32K live contexts, prompt and generation separately, time-to-first-token, cold/warm repetitions, bytes read per generated token, hit rates, I/O waits, RAM/swap/VRAM and GPU/CPU timelines. Increase context only when measurements justify it. Our storage spot measurements do not predict a decode rate without the cache-miss bytes/token.
8. Add DSpark only after the ordinary path is stable and disk headroom is adequate. Measure draft and verification time, acceptance/committed tokens per round and result agreement. Sweep supported depths and contexts. A faster speculative path must pass correctness checks rather than merely produce plausible text.
9. After a soak test, add a distinct backend/profile to our existing API. The old Qwen Strata command cannot simply point to this model: the new upstream server has DeepSeek-specific templating/engine handling. Port or wrap our STT/TTS additions separately with regression tests. Keep rollback to all current aliases and the old engine.

## WiCi: useful architecture, different immediate engine compatibility

InferSSD is WiCi's standalone Apache2 SSD-backed inference software. Its architecture indexes selected expert ranges, keeps bounded RAM/VRAM caches, batches prompt accesses and schedules CPU/GPU work. It can use direct GGUF offsets instead of an optional duplicate expert pack. These are relevant local techniques; its wireless GPU transport is a separate layer we do not need. [Architecture](https://github.com/wici-ai/inferssd/blob/368df4f1a4d49c564ded47240b45fbd0dca8b084/docs/architecture.md).

Its configured reader has concrete I/O/prefetch, admission and CUDA-staging ownership, not merely mmap plus hope. [Reader configuration](https://github.com/wici-ai/inferssd/blob/368df4f1a4d49c564ded47240b45fbd0dca8b084/docs/ssd-reader-configuration.md). Useful measurements to borrow are request/byte counts, cache retention, upload waits and compute placement. The VM must prove its actual storage path before adopting physical-NVMe expectations.

Critical compatibility finding: InferSSD's DeepSeek descriptor requires all up/gate/down experts to be MXFP4. Our actual IQ expert types fail that rule. An IQ2 label being understood by the GGUF parser does not establish executable model support. [Pinned model adapter](https://github.com/wici-ai/inferssd/blob/368df4f1a4d49c564ded47240b45fbd0dca8b084/engine/src/ssd/model-adapter.cpp). Adapting it would require quant-kernel/layout and activation validation, not deleting the guard. That is a separate research project; do not defer the supported Strata first run for it.

InferSSD's recorded DeepSeek checkpoint is a different roughly473GiB artifact with nativeMXFP4 experts, not our AtomicChat IQ quant; its tested conversion excludes vision and DSpark. It would not fit our free disk. [Benchmark format notes](https://github.com/wici-ai/inferssd/blob/368df4f1a4d49c564ded47240b45fbd0dca8b084/docs/benchmark.md). Published throughput comparisons use different complete configurations and do not establish speed or quality on our VM.

Conclusion: practical RTX-only Strata feasibility is credible, conditional on companion downloads, runtime compatibility, memory admission and storage performance. VRAM alone is not the blocker. First prioritize a correct6expert/no-drafter Strata baseline; use InferSSD as a design/profiling reference and a later controlled alternative, not a promised drop-in speed upgrade.
