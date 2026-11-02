# Isolated Strata P40 experiment — 2026-11-02

## Scope and preserved state

No kernel, NVIDIA driver, transport pacing, original rootfs, API model configuration or Proxmox host changes. The test rootfs is a copy of rpc-p40-root with a private CUDA12 Strata installation under /opt/strata. The conservative RPC entrypoint remains available. No benchmark result exists yet.

## Verified engine

Downloaded release v0.2.9, strata-engine0.1.55 Linux x86_64 CUDA12. Archive SHA2562455898afd445fa2c814676f0923339707344bc198286f14694b76c9a7baefb6 matches the saved upstream checksums. Upstream manifest signature and every component file passed verification. Manifest explicitly includes sm_61 (also sm_60/70/75/80/86/89,compute_89). Engine --help ran successfully on primary; this is NOT P40 CUDA execution validation. install_verified.py and engine-verification.json record the checks.

## Startup blocker

Primary remains7.0.0-mk2-shiba-mk1. Initially Multikernel filesystem unmounted, no active secondary; P40 PCI00:11.0 present and unbound. Mounted control filesystem and loaded existing lazy_cma/daxfs modules. No secondary was created. Kerf baseline pool reservations failed with ENOMEM for6GiB,3GiB,1GiB and512MiB. Unloaded the resident API Qwen model: available memory rose to47GiB, but reservations still failed. Explicit VM memory compaction did not recover3GiB. Suspected physical-contiguity/unmovable-page fragmentation; not proven. Kernel logs in primary-pool-failures.log. All12primaryCPUs remained online, instance directory empty.

The failures occurred before secondary boot, PCI assignment, reset, interrupts or DMA; they are not evidence of Strata SM61 incompatibility or nested PCI failure. No P40 output correctness, GPU kernel timing, combined GPU throughput or full model load has been measured.

Restored shiba-model-api.service and previously active qwen3.8-flash-next-strata. Authenticated local generation returned READY. API profile and keys unchanged.

## Next clean-boot test

Start before loading a large API model. start-secondary-test.sh uses the unchanged kernel, CPUs8–11,1792MiBsecondaryRAM,6GiBpool and copied test rootfs. It requires the staged correctness probe and stops on errors. The larger pool accommodates private engine/runtime files; it is not a claim that a full Qwen model fits this secondary.

Full Qwen inference also requires access to model weights and enough CPU expert memory. The existing1792MiBsecondary layout was designed for GPU-only ggml RPC. It is insufficient for Strata's normal CPU expert arena. First run the small SM61 kernel correctness probe; then implement a suitable model/memory-sharing arrangement or a resident expert worker. Do not place a full40GBmodel into the rootfs without separately budgeting its pinned DAX allocation.

## Build environment

CUDA12.8 rejects default GCC15; use installed G++13. Ubuntu's newer glibc adds noexcept declarations conflicting with six CUDA12.8 math declarations. build uses an isolated33MiBcopy of toolkit headers with those six declarations adjusted, never modifies /usr/local/cuda or driver packages. Probe is upstream mmvq_multi_parity built forsm_61, same8rowCUDAverification window. Compile success is distinct from runtime correctness.

Final preparation: SM61probe compiled successfully (exit0), native sm_61ELF confirmed with cuobjdump; staged SHA2565bc522a657c0388d0f1a5e911c9b4b99a2b5f3c0d37050b27e6672b142df8e3b. Upstream verify_kernels.cu supplies PDL helpers; no stubs. DAXFSsize2599702528bytes plus1792MiBsecondaryRAM fits6GiBrequestedpool, asserted in code. Runtime still untested. Shell syntax and Python compilation checks passed. Public second-user /v1/models returned200with21models. GRUBstillselects7.0.0-mk2-shiba-mk1; no reboot performed.


## Postboot results — supersedes the startup blocker above

A single6GiBreservation failed again with46GiBavailable/no swap. Scoped kretprobe on alloc_contig_range_noprof observed rejected candidates EBUSY(-16) followed by successful512MiBallocation. Incremental512MiB ->3GiB ->6GiBpool growth succeeded. No allocator/kernel patch or boot change. Startup script now has that fallback. Temporary tracing removed afterward. Secondary booted asCID1,CPUs8–11,1792MiBRAM,P40ownedbyunchanged580.173.02driver; signedCUDA12strata--versionworked. MSI IRQ16active,P0observed1303MHzSM/3615MHzmemory.

Custom probe initially failed before CUDA: host build requires sqrtf@GLIBC_2.43, secondary has olderlibm. Transferred a private loader/libc/libm/libstdc++/libgcc set into/tmp/probe-runtime and invoked only probe processes with it; original libraries and driver stacks unchanged. Runtime hashes recorded. Future test root preparation includes this isolated runtime, both probes and small fixtures.

P40 mmvq_multi_parity:148480outputs compared,0bitwise differences,0nonfinite. Negative control:63830finite differences, showing comparison power. Tested Q4_0,Q8_0,Q4_K,Q5_K,Q6_K,IQ4_XS plus widerQ5_K at widths1,2,3,4,5,6,8. Independent iq_parity tests passed all10formats(IQ2_XXS,IQ2_XS,IQ2_S,IQ3_XXS,IQ3_S,IQ1_M,IQ4_NL,IQ4_XS,Q2_0,Q3_K), widths1–8, first32x256then640x2560(realQwen gate/up shape; synthetic weights, not checkpoint tensors). Dequantization relerror0; matvec relativeL1~0.0046–0.0057within0.02activationquantizationtolerance; multi-column outputs bitwise equal to single-column calls. No token-generation or inference-throughput claim.

Separate DeepSeek-specific ds4_iq.cu compile forsm_61 failed at818and848:WMMA namespace/fragments instantiated for a card without TensorCores. This is a source-level capability gate, not a PCI or driver failure. Qwen's generic kernels passing does not qualify the DeepSeek-specific kernels. No upstream source changes made.

Transport caveat: first native file transfer emitted secondary__local_bh_enable_ipWARNING with mk_vsock_ipi_handler in hardIRQ path. Transfer and later probes completed. No ring-full or NVIDIA Xid in saved secondarydmesg. This is a Multikernel/VSOCKcontext warning; cannot attribute it to nested PCI from this evidence. Do not hide it or change kernel to suppress it.

Restored5002using exact stable libggml-rpcSHA2565c0fb8e5279b4c1270f209e08f4d0a154cf95fcc77bd8300c4e41a4986bb75b5, unchanged primarywritepacing, secondary NO_READ_SLEEP/NO_WRITE_SLEEP opt-ins.5000untouched. Exact remoteADD/allocation/copy passed1KiB,8KiB,64KiB,1MiB,16MiB;16MiBread83.214ms. Primary validator also enumeratedRTX5070Ti. APIhealthok,noresidentmodelafterreboot. No full Nemotron load rerun.

Full Qwen inference still needs model access and enough CPU expert memory or an expert-only worker. Normal Strata multi-GPU code requires devices in one CUDA process, while ours span kernels. Neither fullQwenP40inference nor dualGPUStrata execution has been tested. The successful kernels establish a useful prerequisite, not a performance improvement.
