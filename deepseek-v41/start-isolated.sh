#!/usr/bin/env bash
set -euo pipefail
exec /home/shiba/multikernel-experiment/research/dsv41-venv/bin/python \
 /home/shiba/multikernel-experiment/research/Strata-dsv41-experiment/serve/server.py \
 --engine strata --config /home/shiba/multikernel-experiment/deepseek-v41/strata-deepseek-iq2-xs.json \
 --host 127.0.0.1 --port 19085 --no-api-key --offline
