# Historical initial worker checkpoint

**This snapshot predates the recovery reboot. See [CURRENT.md](CURRENT.md) for
the working, qualified IQ2 offload path, corrected oracle, results and commands.**

Experimental correctness path, 2026-11-03. Not installed in the API.
No NVIDIA package, kernel, boot entry or production write-pacing changes.
The old filesystem bridge remains disabled.

The primary owns model files, PLE disk reads, the RAM expert arena, MTP and
the RTX coordinator. A secondary worker owns a small static set of packed
Qwen experts on the P40. Only one-time expert loads, activation/routing
requests and expert output rows cross AF_VSOCK. The protocol explicitly
selects Multikernel transport type 1. Both directions retain 32 KiB chunks
and 100 us progress pacing. Partial transfers are handled; malformed frames
and unsupported shapes fail explicitly. Request bodies are capped at 2 MiB,
resident expert payloads at 64 MiB, and windows at 8 rows/80 entries.
Nothing exports arbitrary file paths or tries to load the full GGUF in the
secondary's RAM. A connection owns its cache, freed on disconnect.

## Evidence

* Worker v1 passed 146 requests on the real P40. Six checkpoint-native experts
  from layers 0, 1 and 8 cover IQ2_S, IQ2_XXS and IQ1_M gate/up with Q2_0 down.
  Their hashes are recorded; source offsets come from the native-experts v3 index.
  Total resident packed payload was 7,987,200 bytes. M=1/2/4/8 output relative
  L1 error against independently dequantized CPU float reference was
  0.01253–0.01392, cosine similarity above 0.99990. All ten repeats per case
  matched exactly. This tolerance allows the kernel's activation quantization;
  it is not bitwise equivalence to float arithmetic or full-model validation.
* M=1 end-to-end expert-request medians were 0.789–0.804 ms; M=8 medians
  5.756–5.849 ms. These include paced output of ten expert rows per input row,
  not just CUDA compute. They are not model tok/s. Three short writes occurred
  and were handled. The cache-clear check returned resident payload to zero.
* The familiar secondary VSOCK softirq warning recurred. The saved primary
  dmesg contains no ring-full, Xid or soft-lockup report in this boot up to
  capture. This does not establish long-run transport reliability.
* The primary adapter is in the separate Strata worktree/branch
  `Strata-mk-worker`, `experiment/shiba-mk-expert-worker`. It changes only the
  host expert-dispatch adapter. `build_primary.py` relinks against fingerprinted
  original RTX objects/libraries, replacing only the adapter object. With the
  feature disabled it exactly matched the original engine on a two-token
  Paris response and a 32-token bicycle response (34 committed tokens total).
  Both test arms used a 2K context, original MTP/PLE settings and the existing
  diagnostic GPU-expert bypass. Production context was not edited.
* Initial reference run misparsed chat control markers. Its logs/results are
  retained as `invalid-chat-markers-*` and excluded from correctness claims.

## Current blocker and scope

After stopping and reloading the secondary to stage the updated worker, the
P40 failed NVIDIA initialization: `RmInitAdapter failed (0x23:0x65:1587)`.
Reloading the same modules did not recover it and emitted a secondary SLUB
remaining-object warning. No PCI reset/reset_method sysfs file is exposed for
the P40 in either kernel. The primary also logged IRQ 11 being disabled after
an unhandled interrupt; saved logs name USB/HDA handlers. These may involve
virtualized PCI/reinitialization behavior, but their root cause is unproven.
The secondary is stopped. No shared PCI bus reset or Proxmox change was tried.
A VM reboot is the next recovery step; it has not been performed by the agent.

Worker v2's sparse routing operation and the live coordinator shadow/execute
gates remain UNVALIDATED. The failed v2 validation JSON has no `passed` flag.
Do not expose this adapter through the API or claim a full-model speedup.
DeepSeek remains a separate task: its current expert source fails SM61
compilation on WMMA code. This Qwen worker is not a DeepSeek Pascal port.

## Resume after VM reboot

1. Keep the current primary kernel and both NVIDIA installations. Unload the
   API model for the isolated run; do not change its catalog/configuration.
2. Use the prepared minimal image and `start-secondary.sh`. Only the pool is
   grown in small chunks before boot; the running instance is never hot-added.
   The initial 3 GiB pool and contiguous 3 GiB growth failed during preparation;
   1 GiB/1.5 GiB pool increments succeeded. Image allocation also needed a fresh
   contiguous chunk after the secondary reload. Treat errors as failures, not
   instructions to grow memory without bound. The original RPC root is intact.
3. From the secondary console launch:

   ```sh
   /opt/mk-worker/runtime/ld-linux-x86-64.so.2 --library-path /opt/mk-worker/runtime:/opt/mk-worker:/opt/rpc/lib:/lib /opt/mk-worker/worker > /tmp/mk-expert-worker.log 2>&1 &
   ```

4. Run `validate.py` using `research/dsv41-venv/bin/python`, writing a NEW result
   filename. Require all full/sparse CPU-reference and routing/reordering checks.
   The worker supplied in the image is v2. Use the staged `vsock_file_pull`
   helper for future bounded binary updates; avoid another secondary reload
   just to update a worker executable. Do not run console transfers concurrently
   with bulk data transfers.
5. Run `run_primary_gate.py --mode shadow --tokens 32` in a 40G maximum / 38G high
   / zero-swap systemd user service. A maximum of 32 selected profile pairs are
   loaded lazily; CPU results remain authoritative. Reject excessive error.
6. Only after shadow passes, run `--mode execute`. This initial correctness
   mode still computes CPU reference rows and then replaces selected rows with
   P40 results. It deliberately offers no performance promise. Require exact
   committed-token agreement with the reference before removing duplicate CPU
   work, growing residency or benchmarking speed.

The adapter is opt-in via `STRATA_MK_EXPERT_ENDPOINT=1:6005`,
`STRATA_MK_EXPERT_PAIRS=/path/profile-first32.txt` and mode `shadow` or `execute`.
Without the endpoint it does not create a socket. Original GPU cache bypass
remains enabled in the harness. Errors fail generation instead of silently
using stale rows. The production API continues to use the original executable.
