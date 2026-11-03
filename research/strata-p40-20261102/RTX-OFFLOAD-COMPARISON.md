# RTX baseline versus failed secondary launch

Source inspection, 2026-11-02. No new GPU throughput measurements.

## Qwen Flash Next

The working API reads `model-api/strata-iq2-xs-safe.json`. It supplies both
`--native` shard 1 and `--ple-gguf` shard 2. In the deployed source,
`src/program/generate.cpp` defaults `ple_io` to `direct` and constructs the
PLE reader separately from the expert source. The retained API log confirms
PLE enabled with 320001536 rows. This is disk-backed n-gram-table access,
not a request to keep the whole GGUF in VRAM.

The working configuration does not select `--mmap-experts`: its expert source
is the resident host arena, confirmed by the retained log's successful
cudaHostRegister message. It selects one GPU expert-cache slot, zero PCIe
miss fraction, and disables adaptation. The API additionally enables
STRATA_DIAG_BYPASS_EXPERT_HITS to avoid the previously reproduced GPU-expert
correctness failure. The MTP draft pack is supplied with capacity 4.

The failed secondary launch supplied shard 2 via `--ple-gguf` too. It selected
`--mmap-experts`, automatic GPU caching, capacity 2 without MTP weights,
three CPU workers and a 2K context. Model files were exported through a new
VSOCK/FUSE bridge. It therefore did not reproduce the working baseline.
It did not request all weights resident in host RAM or VRAM; nevertheless,
GPU-cache initialization caused substantial cross-kernel model traffic.
Direct PLE I/O was requested by default; the lost secondary log prevents
claiming its exact runtime behavior on this FUSE mount.

No missing n-gram flag is established as the hang cause. The recorded RCU
soft lockup, IPI warnings and simultaneous memory additions remain unresolved.
The failed full-model launch stays disabled.

## DeepSeek V4.1

This is a different engine and offload mechanism. The working DSpark API
configuration is `deepseek-v41/strata-deepseek-iq2-xs-dspark.json`:
GPU 0, SSD cache 32, all six selected experts, DSpark enabled with block 5
and confidence 0.65. The index-only pack references the Engram companion.
The API enforces a 42 GiB memory limit. Preserve these settings and its
readonly expert profile during any later integration.

The failed full-model test was Qwen, not this DeepSeek configuration.
DeepSeek's current `ds4_iq.cu` fails SM61 compilation on WMMA code; a working
Qwen Pascal kernel does not establish DeepSeek Pascal support.

## Next implementation boundary

Keep GGUF/Engram/PLE reads, the expert host arena, drafting and request
ownership on the primary, using their existing local paths. Do not export
the complete model filesystem to the secondary again.

An isolated secondary expert worker should first accept a bounded set of
packed expert weights, retain them in P40 VRAM, and exchange activations,
routing metadata and results over the established paced Multikernel VSOCK
transport. Initial correctness tests must compare those results with an
independent reference before changing serving configuration. Bound resident
weights and request sizes; reject missing expert IDs instead of paging
arbitrary model files through the transport. Preserve primary write pacing.

Strata's `src/core/remote_experts.cpp` currently uses cudaGetDeviceCount and
cudaSetDevice: its helper is another device in the same CUDA process, not
a cross-kernel worker. It requires an explicit isolated adapter. This adapter
is not implemented or validated by this investigation. Qwen and DeepSeek
need separate kernel/format qualification. No two-GPU model speed is claimed.
