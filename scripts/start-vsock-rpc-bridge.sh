#!/usr/bin/env bash
set -euo pipefail

# Multikernel's custom option is handled by the AF_VSOCK level in this kernel.
# Linux's SOL_VSOCK (287) is not the level used by this implementation.
log=/home/shiba/multikernel-experiment/logs/primary-vsock-bridge.log
exec socat -d -d \
  TCP-LISTEN:50053,bind=127.0.0.1,reuseaddr,fork \
  VSOCK-CONNECT:1:5000,setsockopt-int=40:9:1 \
  >>"$log" 2>&1
