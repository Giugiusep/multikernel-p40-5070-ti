#!/bin/sh
set -eu
root=/opt/mk-worker
export MK_WORKER_CPU_COMPAT=1 MK_WORKER_CPU_EXACT_IQ2=1 MK_WORKER_CPU_POOL=1 MK_WORKER_PORT=6006
exec "$root/runtime/ld-linux-x86-64.so.2" --library-path "$root/runtime:$root:/opt/rpc/lib:/lib" "$root/worker-qualified"
