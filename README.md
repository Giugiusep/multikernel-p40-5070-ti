# Multikernel inference: RTX 5070 Ti + Tesla P40

An engineering experiment running one language model across two Linux kernels inside a KVM virtual machine. Each kernel owns a different passed-through GPU and uses the NVIDIA driver appropriate for that GPU. Patched llama.cpp connects the two through native Multikernel AF_VSOCK/IPI RPC.

This repository preserves the tested source changes, configuration, probes and engineering record. It is not a turnkey installer or a bootable VM backup.

## Architecture

```text
KVM VM — Shiba
│
├── Multikernel primary: 7.0.0-mk2-shiba-mk1
│   ├── RTX 5070 Ti → nvidia-open 595.84 → CUDA
│   └── llama.cpp coordinator / local CUDA backend
│                  │
│                  │ native AF_VSOCK / shared-memory IPI
│                  │ mkvsock:1:5002
│                  ▼
└── Multikernel secondary
    ├── Tesla P40 → proprietary NVIDIA 580.173.02 → CUDA
    └── llama.cpp RPC worker
```

The two drivers run in separate kernels. The experiment does not require changes to the physical Proxmox host. NVIDIA versions above describe the tested pre-upgrade configuration, not a compatibility claim for arbitrary newer packages.

## Current status

The VM has now upgraded to **Ubuntu 26.04.1 LTS**. At the README update it was running the stock **6.8.0-142-generic** kernel. Multikernel was deliberately deactivated for upgrade preparation, and the model API was stopped with automatic startup disabled.

**The dual-kernel inference configuration has not yet been revalidated after the release upgrade.** The results below were obtained before it. Matching NVIDIA-open 595.91.07 modules have now been rebuilt and installed for the unchanged custom kernel, Kerf has been repaired for Python 3.14 (330 tests passed), and the custom GRUB default has been restored. Reboot and GPU/transport validation are pending. The P40 stack and stable llama.cpp code remain unchanged. See `configs/resolute-reactivation.json` and the audit for preparation evidence.

## Measured results

| Configuration | Prompt tokens/s | Generation tokens/s |
|---|---:|---:|
| RTX CUDA + host RAM baseline | 55.6 | 15.5 |
| P40 Nouveau/NVK + host RAM baseline | 12.1 | 8.0 |
| RTX CUDA + P40 NVK split baseline | 6.0 | 8.3 |
| Native Multikernel RPC, fast transport, 48/52 split | 69.2 | 66.0 |
| Native Multikernel RPC + backend sampler optimization, 50/50 split | — | 72.0 |

The final 50/50 CLI reference had generation runs of 71.9 and 72.1 tokens/s. These are Nemotron Q8 measurements under the conditions recorded in the [engineering audit](multikernel-engineering-audit.md). API serving, allocated context and split settings affect results: the separately tested 160K-context API profile reached approximately 58 tokens/s and should not be presented as the 72 tokens/s CLI result.

Native transport initially measured a 52.920 µs ping/pong RTT, 135.47 MiB/s for 1 MiB and 129.02 MiB/s for 16 MiB. Removing unnecessary read and secondary-send progress sleeps brought a verified 512 KiB return to 2.452 ms. These measurements come from different isolated tests, not one simultaneous benchmark.

## Transport and safeguards

The socket explicitly selects Multikernel transport:

```c
int transport = 1;
setsockopt(fd, AF_VSOCK, 9, &transport, sizeof(transport));
```

Without this selection, generic AF_VSOCK can choose the VM transport and return `ENODEV` before entering Multikernel. The llama.cpp implementation is isolated in the RPC transport layer and handles partial reads/writes. It uses native sockets; socat is not the inference transport.

- Port **5000** remains the conservative RPC fallback.
- Port **5002** is the tested fast configuration.
- Primary upload/write pacing remains enabled, using approximately 32 KiB chunks. Removing it caused sustained IPI-ring `ENOSPC` during model loading and was rejected.
- Primary read sleeps and secondary read/send sleeps are disabled in the fast configuration.
- The stable backend sampler optimization is included in llama.cpp commit `3e88755aa`.

The experiment runs with virtualized PCI passthrough. Reset support, interrupts, DMA and guest IOMMU facilities need separate inspection when reproducing it. A failed secondary shutdown can require a full VM reboot to recover the P40. Guest transport success is not evidence of DMA isolation suitable for production.

## Repository map

| Path | Contents |
|---|---|
| [manifest.json](manifest.json) | Pinned upstream bases, tested commits, driver versions and local binary hashes |
| [patches/linux](patches/linux) | Ten committed custom-kernel patches |
| [patches/llama.cpp](patches/llama.cpp) | Fifteen committed transport, profiling and sampler patches |
| [configs](configs) | Pre-upgrade kernel/GRUB settings, package inventory and API service |
| [scripts](scripts) | Build, lifecycle, profiling and upgrade-preparation helpers |
| [tests](tests) | CUDA, RPC, transport and P40 correctness/performance probes |
| [lazy_cma](lazy_cma) | Experimental resource-allocation helper sources |
| [model-api](model-api) | Authenticated model-switching API code/configuration, without credentials |
| [multikernel-engineering-audit.md](multikernel-engineering-audit.md) | Detailed experiments, failures, correctness checks and benchmark evidence |

Configuration and package inventory files describe the snapshot date; they are not automatically synchronized with the upgraded VM. Some research scripts refer to experimental branches or models outside the stable patch series.

## Reconstruct the tested source

Use the exact bases in `manifest.json`. For example, from this repository's root:

```bash
snapshot_dir="$PWD"
git clone https://github.com/multikernel/linux.git ../multikernel-linux
cd ../multikernel-linux
git switch -c experiment/recovery 3bdd35b64413da0b4e089ce931bfc2e8b031cbf7
git am "$snapshot_dir"/patches/linux/*.patch

cd "$snapshot_dir"
git clone https://github.com/ggml-org/llama.cpp.git ../llama-multikernel
cd ../llama-multikernel
git switch -c experiment/recovery c841aeeb8bb2fe417038dadfa9b007cf1a9ef950
git am "$snapshot_dir"/patches/llama.cpp/*.patch
```

Kerf and daxfs are pinned directly to their recorded upstream commits. Apply the saved kernel configuration and consult the audit/build scripts for the actual build sequence. Inspect absolute paths and local assumptions before running scripts on another machine. Applying patches reconstructs source; it does not install or boot the environment. Replayed commit IDs may differ from the original IDs.

For the custom primary NVIDIA build, the tested source/output arguments were:

```text
SYSSRC=/home/shiba/multikernel-experiment/source
SYSOUT=/home/shiba/multikernel-experiment/build
```

The pre-upgrade custom-kernel modules were manually installed and were **not registered in DKMS** for that kernel. Updating NVIDIA userspace independently can break the module/userspace match. The audit records the build details and the previously failed DKMS configuration.

## Upgrade and recovery

Before the release upgrade, stock `6.8.0-142-generic` was selected as the default with a 15-second GRUB menu. The custom `7.0.0-mk2-shiba-mk1` entry was retained. Stopping the secondary alone would have left the custom primary active, so it would not have provided a normal Ubuntu kernel/driver upgrade path.

A checksum-verified local recovery archive was saved under:

```text
/home/shiba/multikernel-upgrade-recovery-20261006/
```

It contains boot files, custom-kernel modules and selected GRUB/apt configuration. It is intentionally absent from GitHub. Retain that archive and the VM backup: source patches and binary hashes cannot replace compiled boot artifacts.

After the upgrade, verify the installed kernel, NVIDIA package/module versions, GPU access and secondary lifecycle before restoring the model API. Restore the custom GRUB default only after its driver compatibility is established. The saved original GRUB fragment and the [preparation script](scripts/prepare-stock-upgrade-boot.sh) document the reversible change; do not run that machine-specific script blindly on another installation.

Once the desired runtime has been verified, the previously enabled API can be restored with:

```bash
systemctl --user enable --now shiba-model-api.service
```

## Scope and licensing

Model weights, NVIDIA installers, compiled kernels/modules, secondary root filesystem, API keys and other credentials are excluded. This snapshot preserves the stable Nemotron/Multikernel source branch; it does not claim to include or validate every later Qwen/MTP research branch.

The repository-level license does not relicense third-party patches or copied sources. Their upstream notices and licenses continue to apply.
