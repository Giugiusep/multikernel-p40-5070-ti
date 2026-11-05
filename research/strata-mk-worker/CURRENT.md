# Multikernel Qwen expert worker — experimental

Updated 2026-11-03 after the recovery reboot. P40 is working and the secondary
is active. No kernel, NVIDIA package, boot entry or production API configuration
changed. No additional VM/secondary reboot was performed during this run.
The failed FUSE model-filesystem bridge remains disabled.

## What now works

The primary retains its original Qwen IQ2_XS files, PLE disk reads, resident
host expert arena, MTP drafting and RTX dense/attention computation. An opt-in
coordinator sends selected expert calculations to P40 over Multikernel VSOCK.
In `offload` mode the corresponding CPU expert jobs are skipped. All other
experts remain on the primary CPU. This is one model, not two models or an
independent draft server.

Qualified pilot: top 32 ranked IQ2_S/IQ2_XXS pairs, 64 MiB maximum packed cache,
maximum eight verification rows. The longer test actually used 13 resident
pairs and 143 expert entries over 87 nonempty callbacks. IQ1_M experts stay on
the CPU in this configuration; none are dropped from the model.

* P40 full/sparse routing tests passed after reboot.
* Eight supported IQ2 cases at M=1/2/4/8 matched the **actual Strata CPU-pool
  output bytes**, including ten exact repeats and sparse row reordering.
* Full-model checked replacement matched all 130 committed tokens (Paris 2,
  bicycle 128), with zero error in every live compared expert output.
* Actual offload, with duplicate CPU jobs skipped, also matched all 130 tokens.
* Tests use greedy sampling, spec capacity 4 and a 2K context. Production API
  retains its existing large-context configuration. These prompts do not prove
  general correctness, long-context reliability or sampling equivalence.

Original Qwen API is restored. Port 5000 remains the conservative GGML RPC
fallback. The unchanged, verified fast RPC library was restored on port 5002:
SHA-256 `5c0fb8e5279b4c1270f209e08f4d0a154cf95fcc77bd8300c4e41a4986bb75b5`.
Qualified expert worker is on 6006; the earlier native worker remains on 6005.
Neither expert worker is exposed by the public API.

## Exact arithmetic contract

The native GPU path quantizes inputs as Q8_1, while the target CPU IQ2 path uses
Q8_K. The live CPU pool's Q2_0 down path **also differs from generic ggml**:
`pool.cpp` uses `act_quant_any`/`ActQ` with FP32 scales, half-away codes and
precomputed correction, followed by `q2_rows_any`'s AVX512 reduction.

The qualified worker matches those boundaries. It uses the same CPU input and
intermediate quantizers, keeps weights packed in P40 VRAM, and performs GU and
down matrix operations on the GPU. IQ2 integer contributions and FP32 FMA/
reduction ordering reproduce the pinned CPU paths, including AVX2 versus
AVX512 GU ordering depending on each expert's verification-row count. No
Tensor Core instructions or materialized full FP32 weight matrices are used.
CPU host helpers currently compute quantization and SwiGLU between GPU stages.

`compat-kernels.inc` is included in a generated translation unit containing
the pinned, otherwise unchanged IQ CUDA source. `legacy_act.h` defines only the
small private intermediate representation. This is specific to the current
CPU/Strata arithmetic, not a universal replacement for CUDA expert kernels.

The primary adapter lives on `experiment/shiba-mk-expert-worker` in the separate
`Strata-mk-worker` worktree. The original deployed source/binary is intact.
`build_primary.py` fingerprints and reuses original GPU/driver objects while
replacing only the host expert-dispatch object. Offload requires the worker's
`strata-pool-iq2-v1` handshake and the existing primary GPU-expert bypass.

## Timing: no demonstrated speedup

Single 128-token correctness runs, same 2K context and sampling:

| Configuration | Decode tok/s |
|---|---:|
| Same-session original RTX/RAM reference | 32.61 |
| Checked P40 replacement (duplicate CPU calculation retained) | 32.82 |
| Actual P40 offload | 32.24 |

These are single runs, not a controlled repeated performance study. The small
pilot has not improved speed and is not promoted to production. An earlier
cold reference was 18.92 tok/s; comparing only against it would falsely imply
a gain. Correctness-first kernels are slow: ten-entry M=1 requests are about
5 ms versus about 0.8 ms for the numerically different native path. Native
versus qualified timings are not equivalent-math performance comparisons.

## Failed variants and oracle correction

The initial native worker passed the independent float-dequantization tolerance
but changed full-model output at the sixth generated bicycle token. Preserving
CPU quantizers while using generic GPU float reductions matched the short test
but diverged at token 88. Reproducing IQ2 GU accumulation while still using
generic ggml Q8_0 down arithmetic diverged at token 112.

The initial `cpu_reference` harness used generic `native_down_rows`, **not the
actual pool's optimized Q2_0 path**. Earlier exact probe claims therefore apply
only to that generic oracle. The corrected oracle is `cpu_reference_pool`,
which calls the actual `act_quant_any`/`q2_rows_any` functions. Its executable
name selects pool mode; `MK_CPU_POOL_ORACLE=1` is an additional explicit switch.
`exact-actual-pool-bytes.json` records the corrected contract and byte equality.

`rounding-diagnostic.json` concerns the earlier generic oracle: host emulation
of GPU GU ordering changed one intermediate quantized byte and reproduced that
GPU result exactly. It does not qualify the generic oracle as the live target.
Failed gate JSONs/logs and invalid-chat-marker initial logs are retained.

## Transport and VM limitations

All sockets select AF_VSOCK option 9, type 1. Request bodies are capped at 2 MiB;
partial transfers and backpressure are handled. Bulk writes retain 32 KiB
chunks and 100 us progress pacing. The worker loads bounded blobs from the
primary, never arbitrary paths or an exported model filesystem.

The first one-shot helper upload logged 47 secondary ring-full reports over
3.273 ms and still passed its SHA-256 check. Inspection found that its outer
file loop bypassed the intended sleep between 32 KiB calls. The separate
`vsock_file_pull_paced.c` corrects that helper only. Subsequent updates used it;
no global transport or primary RPC write-pacing setting changed. Final saved
secondary dmesg has no Xid, OOM, soft lockup or RmInitAdapter error in this boot.
The familiar VSOCK softirq warning remains. This is not long-run reliability
proof, and the old freeze's root cause is not established.

The previously observed P40 reinitialization failure followed a secondary
reload. No PCI reset interface was exposed. After the user's VM reboot this
run updated worker processes through bounded, hash-checked transfers without
restarting either kernel or unloading NVIDIA. The instance remains at 1792 MiB;
only its preboot resource pool was grown, to 5632 MiB. No live instance hot-add.

DeepSeek is still separate: its current expert CUDA source fails SM61 compilation
on WMMA code. This Qwen qualification is not a DeepSeek Pascal port.

## Reproduce the qualified experiment

Build with `build.sh`, `build_compat.sh`, `build_cpu_oracles.sh` and `build_primary.py`.
`prepare_root.py` stages a private copy of the original RPC root and both
workers; no model weights or GPU driver files are changed. `start-secondary.sh`
is for a fresh boot with the P40 unbound. Do not rerun it on the active instance.
The helper script inside the secondary is launched explicitly:

```sh
sh /opt/mk-worker/run_worker_qualified.sh > /tmp/mk-expert-qualified.log 2>&1 &
```

Keep the API model unloaded during standalone coordinator tests. The expert gate:

```sh
env OPENBLAS_NUM_THREADS=1 MK_CPU_POOL_ORACLE=1 \
  ../dsv41-venv/bin/python validate.py --port 6006 --layers 0,1 \
  --cpu-reference ./cpu_reference_pool --require-exact --output NEW-results.json
```

Use `run_primary_gate.py` in a systemd user service with MemoryMax=40G,
MemoryHigh=38G, MemorySwapMax=0 and RuntimeMaxSec=1200. Select
`--mode offload --endpoint 1:6006 --pairs profile-first32-iq2.txt`, and compare
against a same-prompt, same-token-count reference via `--reference`.
Use `--tag` to retain each result. Do not infer a speedup from one cold/warm pair.


## 2026-11-05 checkpoint

Production stays unchanged. Default32-pair packed/async experiment remains available on6007; conservative qualified6006 intact. Broader128-pair offload passed726 exact tokens only in matched --suffix-draft0 tests; default adaptive suffix runs diverged even in byte-exact shadow mode. See audit: timing-dependent verification selection/numerical sensitivity is implicated, not fully traced. Do not promote broader offload.

Row-packing candidate passed exact probes (M1 request about15–18% faster), but no reliable full-model gain; active kernel source reverted and6008 candidate stopped. Candidate patch and all failed/successful evidence kept in perf-20261105. Six/five CPU workers did not beat the first seven-worker control. Identical controls show large timing swings correlated partly with system-wide I/O stalls; no new generation-speed claim. API restored to original Qwen.
