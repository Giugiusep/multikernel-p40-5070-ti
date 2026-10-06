# Multikernel: Tesla P40 + RTX 5070 Ti

Recovery source snapshot of the working Shiba KVM experiment, captured 2026-10-06 before a planned Ubuntu release upgrade.

Primary: `7.0.0-mk2-shiba-mk1`, RTX 5070 Ti, NVIDIA open 595.84. Secondary: Tesla P40, proprietary NVIDIA 580.173.02. Stable llama.cpp includes native Multikernel AF_VSOCK RPC and the backend sampler optimization. Nemotron Q8 50/50 reached approximately 72 generation tokens/s under the documented benchmark conditions.

## Contents

- `manifest.json`: exact upstream bases, tested source commits, and local boot-artifact hashes.
- `patches/linux/`: ten committed Multikernel kernel changes.
- `patches/llama.cpp/`: fifteen committed changes, including transport and sampler optimization.
- `configs/`: kernel configuration, GRUB selection, package inventory, and API service.
- `scripts/`, `tests/`, `lazy_cma/`: build, lifecycle, transport and profiling sources.
- `model-api/`: model-switching API code and configuration, without its API key.
- `multikernel-engineering-audit.md`: engineering history, correctness evidence, benchmarks and rebuild details.

## Reconstruct source

Clone each upstream listed in `manifest.json`, check out its `patch_base` on a new experimental branch, and apply the corresponding ordered patch series using `git am /path/to/this-repository/patches/linux/*.patch` or `git am /path/to/this-repository/patches/llama.cpp/*.patch`. Kerf and daxfs are pinned directly to their recorded commits. Use the saved kernel configuration and the audit/build scripts as the recovery guide; inspect paths and environment assumptions before running scripts on a new installation.

The custom primary NVIDIA modules require the matching NVIDIA open source and kernel source/output directories. They are not currently registered in DKMS for the custom kernel. Do not update driver userspace independently of those modules. The audit documents the correct build-directory arguments.

## Recovery limits

This is a source/configuration backup, not a bootable VM snapshot. Model weights, NVIDIA installers, compiled kernels/modules, secondary root filesystem and credentials are intentionally excluded. Boot-artifact hashes identify the local binaries but do not replace them. Retain the VM backup and local build artifacts before upgrading.

Stopping the secondary alone leaves the custom primary kernel active. Temporarily deactivating Multikernel for an OS upgrade means selecting the stock Ubuntu kernel; restoring it afterward may require rebuilding matching driver modules. This snapshot does not perform an upgrade, reboot or boot-selection change.

The repository-level license does not relicense third-party patches or copied sources; their original upstream notices and licenses continue to apply.
