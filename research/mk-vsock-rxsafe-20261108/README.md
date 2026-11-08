# RX-safe boot candidate — installed, awaiting reboot

The user approved building and installing a separate test kernel on 2026-11-08. The live kernel has NOT changed and no reboot has been performed.

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
