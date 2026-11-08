# RX-safe boot candidate — booted, runtime validation in progress

The user approved building and installing a separate test kernel on 2026-11-08. The user rebooted into #12-rxsafe1; both primary and P40 secondary now run this candidate. Existing NVIDIA stacks were reused without modification.

## What is queued

- Candidate image: `/boot/vmlinuz-7.0.0-mk2-shiba-mk1-rxsafe1`.
- Matching initramfs: `/boot/initrd.img-7.0.0-mk2-shiba-mk1-rxsafe1`, byte-identical to the current initramfs.
- Kernel release remains `7.0.0-mk2-shiba-mk1` to reuse existing modules; `uname -v` will identify the candidate as `#12-rxsafe1`.
- GRUB selects the candidate for ONE boot. The persistent default remains the original #9 kernel; that image and its initramfs remain untouched.
- Source branch `experiment/shiba-vsock-rxsafe`, commit `b435af1`, based on the actual running kernel's `54b0135`. The unbooted logging change `df0139d` is deliberately excluded.

Only `net/vmw_vsock/mk_transport.c` changes. All received packets enter the existing receive work item instead of calling generic VSOCK helpers from the hard IPI callback. Credit reservations, ring capacity, send pacing and CID selection are unchanged.

## Failure evidence

`live-rcu-stall-3.log` identifies the existing stall: CPU0's receive worker is interrupted at `vsock_find_connected_socket_net+0x67` while holding the socket-table lock. The IPI handler re-enters that helper and blocks at its lock call (`+0x2c` return). Disassembly places lock acquisition at `+0x27` and unlock at `+0x16a`, confirming same-CPU lock re-entrancy. The lock address matches both contexts. `check_backtrace.py` verifies these relationships. `kvm_wait` is the spin-wait implementation, not evidence of an NVIDIA or outer-PCI failure.

The earlier hardirq warning is preserved in the preceding postboot experiment. The proposed callback was first compile-checked as an external module; that `.ko` is research-only, NOT installed, and cannot hot-swap the built-in transport.

## Build and compatibility evidence

A complete isolated build produced bzImage and in-tree modules with GCC 13.4 and binutils 2.46. It used the current runtime configuration; reviewed differences reflect host-tool versions, unavailable unused Rust capabilities, pahole version with BTF disabled, and OpenSSL capability with RSA signing unchanged. The candidate preserves the kernel release and module vermagic.

All 124 preserved primary module files and the three secondary/bootstrap modules passed symbol-version checks: 11,401 primary imports and 697 secondary imports, with zero mismatches. The installed boot image/initramfs and driver-module hashes in all 130 baseline files remain unchanged. No driver installer, package update, module installation, initramfs rebuild or secondary reset was performed. Compiling in-tree modules was solely for ABI validation; they were NOT installed.

The candidate's compiled IPI handler queues receive work and has no direct socket lookup/receive calls. `boot-preparation/abi-report.json`, `installed.json`, `compiled-receive-handler.log` and `kernel.patch` retain the evidence. The source and build directories are separate from the known-good trees.

## After the user reboots

1. Confirm `uname -v` contains `rxsafe1`, check primary CUDA/PCI state, and verify the API.
2. Start the secondary with `boot-preparation/restore-secondary-rxsafe.sh`. It refuses an old primary or an existing instance and loads the SAME candidate image with the existing rootfs, 1792 MiB, four CPUs and P40. No auto-start or secondary restart has been executed yet.
3. Verify the P40 driver and basic RPC, then connection/credit-window stress, exact 512 KiB, 1 MiB and 16 MiB traffic and logs.
4. Test compact-cache correctness and fallback with partial primary slots before expanding GPU residency or benchmarking models.

Runtime correctness, latency and bandwidth of the patched kernel are still unvalidated. Workqueue latency regression is possible. Keep the original boot entry and upload pacing intact until tests establish a reliable improvement. No physical Proxmox host changes are involved.

## Runtime validation, 2026-11-08

Both candidate banners verified. Primary CUDA enumerates RTX 5070 Ti; secondary NVIDIA 580.173.02 enumerates P40. Conservative RPC5000 passed allocation, transfer, and remote computation through 16 MiB. Fast RPC5002 and compact worker6011 were restored from preserved binaries. All 24 compact-worker format/width cases were bit-exact against CPU reference.

Native ordered echo payloads through 16 MiB passed. Sequential connection churn passed 2,500 connections. Ten download trials per size yielded median 0.7115415 ms for 512 KiB (702.70 MiB/s), 1.4271885 ms for 1 MiB (700.68 MiB/s), and 26.0476835 ms for 16 MiB (614.26 MiB/s). These are synthetic socket results, not inference throughput; primary write pacing remains enabled.

IMPORTANT: five concurrent clients produced ECONNRESET and left the test listener rejecting further connections, though its process remained in vsock_accept and the secondary console remained responsive without new kernel warnings. Recreating only that test listener restored operation. The exact connection-accounting cause is unresolved; sequential testing does not establish concurrent reliability or prove all transport bugs fixed. No kernel/default-boot promotion follows from these results.

Full-model compact-cache correctness campaign is in progress in sibling `deepseek-p40-rxsafe-20261108`; worker-loss/lazy-slot fallback remains an outstanding gate.


### Full-model postboot gate completed

All six 32-token DeepSeek runs passed exact token agreement, without worker disable/timeouts. Code CPU/off 3.30, compact-shadow 4.83, compact-P40 4.17 tok/s; prose CPU/off 3.41, compact-shadow 3.68, compact-P40 2.69 tok/s. Shadow comparisons:424 code and414 prose, all exact. These single sequential trials have storage/cache confounding and do NOT establish a P40 performance win. Keep production settings unchanged. Peak cgroup memory35.8GiB; API restored with health ok and no active model. Both kernels remained responsive. Post-test primary kernel search and secondary dmesg found no new ring-full, RCU stalls or Xids. Outstanding: concurrent-listener reset and live worker-loss/lazy-slot fallback. Persistent GRUB default remains old #9, candidate was one-shot only.
