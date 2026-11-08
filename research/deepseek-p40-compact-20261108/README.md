# Compact P40 down weights and deferred primary reads — experimental

Source commit 0a34805 follows debug commit e92544d. Nothing deployed to the API. Kernel, NVIDIA stacks, primary upload pacing and conservative/fast RPC configurations remain unchanged.

The private worker adds op5 LOAD_DOWN to the existing version2 protocol on6011. It stores only checkpoint-native down weights, leaving the worker's1GiB hard GPU cap and16MiB frame cap unchanged. Modes down-compact-shadow and down-compact use512MiB admission. The same cache now holds176experts /535265280bytes instead of58whole expert blobs. Packed CPU-compatible down arithmetic is unchanged; all24format/width oracle cases remain byte-exact.

ExpertStore gains an optional defer_down callback and per-slot role-validity mask. Only a down role already resident remotely can be omitted from a primary fill. Accessing a missing role loads it lazily before exposing its bytes. Full-blob access repairs all missing roles, and incomplete slots cannot be published as GPU-visible. Remote membership is exposed through atomic flags, avoiding a client/store mutex inversion. Worker failure invalidates remote residency; existing partial slots remain repairable for CPU fallback. Slot allocation/primary RAM capacity is unchanged.

Host deterministic test covers exact bytes, unaligned role offsets, twelve concurrent fallback readers with only one repair read, unchanged gate/up data, full-blob repair, GPU visibility and slot eviction. ASAN/UBSAN pass with detect_leaks0; LSAN cannot run under this sandbox's ptrace. CUDA sm120 engine and SM61 worker compile. HIP/Metal not built. An actual worker-failure test with live partial slots has NOT yet run.

32-token code: baseline4.83, compact shadow3.32, actual compact offload4.43tok/s. All tokens exact;424live shadow comparisons have zero differing bits. Baseline attempted primary read bytes54,321,039,360; compact54,245,007,360. Difference exactly76,032,000bytes, matching deferred-down counter, with zero repair reads. Original store misses/evictions unchanged. This establishes actual avoided I/O, but no speedup at this small cache size.

Prose baseline3.67tok/s. Compact shadow completed252byte-exact comparisons and still matched all32tokens, but a subsequent request hit socket EAGAIN after its15s timeout. CPU fallback completed the run at1.25tok/s; this is a FAILED transport gate, not a qualified benchmark. No actual compact prose offload or cache expansion followed. The controller stops at this failure and preserves every row.

Afterward5000,5002and6011all timed out; console stopped responding, while sysfs still reported active. Primary API restoration succeeded, health ok, active_model null (same pre-test state). P40-dependent services are currently unavailable. No secondary reset, module unload or physical-host action attempted. The shell could not retrieve dmesg after failure; later automatic RCU/NMI output identified same-CPU socket-table-lock re-entrancy. See the transport proposal and its complete backtrace.

Next gate is reliable transport recovery, then actual fallback-with-partial-slots testing and longer exact-token runs. Cache expansion is on hold. See ../mk-vsock-rxsafe-20261108 for a separately prepared receive-context patch; it has NOT been installed.
