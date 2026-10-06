#!/usr/bin/env bash
set -euo pipefail

tracefs=/sys/kernel/tracing
case "${1:-}" in
    start)
        printf '0\n' > "$tracefs/tracing_on"
        printf 'nop\n' > "$tracefs/current_tracer"
        : > "$tracefs/trace"
        printf '4096\n' > "$tracefs/buffer_size_kb"
        : > "$tracefs/set_ftrace_pid"
        printf 'mk_transport_stream_dequeue\nmk_vsock_rx_pkt\n' \
            > "$tracefs/set_graph_function"
        printf 'function_graph\n' > "$tracefs/current_tracer"
        printf '1\n' > "$tracefs/tracing_on"
        ;;
    stop)
        printf '0\n' > "$tracefs/tracing_on"
        cat "$tracefs/trace" > "${2:?trace output path required}"
        printf 'nop\n' > "$tracefs/current_tracer"
        : > "$tracefs/set_graph_function"
        : > "$tracefs/set_ftrace_pid"
        ;;
    *)
        echo "usage: $0 start | stop OUTPUT" >&2
        exit 2
        ;;
esac
