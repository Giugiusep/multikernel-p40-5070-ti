

## 2026-11-13 — reset root cause, worker-loss recovery, cache ABBA

Live #12-rxsafe1 primary/secondary retained, no source/kernel/NVIDIA/pacing/production changes. The previous concurrent-reset interpretation is corrected: a connect-close loop can queue more than five pending children despite five threads. Exact running-vmlinux kprobe at mk_vsock_rx_work+0x177 observed LISTEN=10, sk_ack_backlog=9, max=8, shutdown=0; the sk_acceptq_is_full branch sends OP_RST and connect reports ECONNRESET. Both listeners subsequently answered without recreation. Five acknowledged clients passed2500 exchanges with zero errors. Listener locking/unmatched-control concerns remain review hypotheses, not established causes of this failure. Temporary probe removed. Evidence: research/mk-vsock-listener-20261113.

First manual worker-loss attempt missed the injection window and is explicitly NOT a recovery pass. Automatic token8 loss passed exact64 tokens but had zero deferred reads. Late automatic loss after >=48 committed tokens passed all64 against independent CPU-only reference:100362240 down bytes deferred,36495360 lazily refilled after worker loss; Broken pipe triggered CPU fallback with zero failed expert reads. Control is single-use and validates experimental worker executable before SIGTERM; no driver reset. Evidence: deepseek-p40-fallback-auto-20261113 and deepseek-p40-fallback-late-20261113/recovery-gate.json.

512/1024/1024/512MiB P40 down-cache ABBA, unchanged64-token code prompt/CPU32GB/six experts/DSpark/pacing:

|cache MiB|experts|generation tok/s|decode RPC ms/interval|P40 GPU ms/interval|disk wait ms/token|exact|
|---:|---:|---:|---:|---:|---:|:---:|
|512|176|4.83|4.259|2.016|100.3|yes|
|1024|351|3.19|17.224|4.993|160.8|yes|
|1024|351|4.82|13.472|4.990|102.5|yes|
|512|176|4.47|6.538|2.029|99.1|yes|

Decode counts are end-minus-prefill checkpoints divided by63 generated intervals; GPU time is included in RPC, not additive. Disk/CPU counters overlap and cannot form an additive critical-path accounting.512MiB mean4.65 vs1024MiB4.005 tok/s, only two trials each with large storage variation: no reliable cache-expansion win, do not promote. More residency doubled decode calls5.92->13.35/token while disk expert reads remained~391–392MB/token; CPU gate/up and storage remain substantial. True elimination of those costs requires independently qualified full-expert offload or better residency/IO strategy, not simply a larger down-only cache. No accepted optimization/source changes. API restored to its pretest qwen3.8-flash-next-strata active profile. Both original RPC services retained, P40 worker restored and idle.
