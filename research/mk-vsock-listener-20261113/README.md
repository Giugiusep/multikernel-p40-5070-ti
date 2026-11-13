# Concurrent-listener reset investigation

Live primary and secondary remain #12-rxsafe1. No kernel or NVIDIA changes.

The previous stress test was not bounded to five pending connections: its fifth
thread connected and immediately closed 500 times without waiting for accept.
This can fill the single-threaded server's listen(8) queue.

A temporary secondary kprobe at mk_vsock_rx_work+0x177 captured the request
condition using offsets verified in the exact running vmlinux disassembly:
state at 0x12, sk_ack_backlog at 0x268, sk_max_ack_backlog at 0x26c,
shutdown at 0x24f. Two requests showed state=10 (LISTEN), ack=9, max=8,
shutdown=0. sk_acceptq_is_full therefore sends OP_RST, reported by connect as
ECONNRESET. Later requests showed ack=0; both original and monitored listeners
responded to fresh requests without recreation. Permanent listener failure is
not reproduced, correcting the earlier interpretation.

The acknowledged variant replaces immediate-close churn with zero-length echo
requests whose reply confirms server acceptance. Five concurrent clients then
completed 2,500 exchanges, zero errors, with byte equality checked. This is a
bounded-concurrency pass, not proof of arbitrary-load reliability.

There are source-review concerns about missing listener locking and dispatch of
unmatched control packets to bound sockets. They remain separate hypotheses;
the captured reset does not justify attributing this test failure to them.

Temporary monitoring is test-only. Production RPC ports, pacing and model
configuration remain unchanged.
