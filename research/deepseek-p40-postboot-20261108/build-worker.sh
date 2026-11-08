#!/bin/bash
set -euo pipefail
base=/home/shiba/multikernel-experiment/research
/usr/local/cuda-12.8/bin/nvcc -ccbin /usr/bin/g++-13 -O2 -std=c++17 -arch=sm_61 -I"$base/Strata-dsv41-experiment/src/dsv41" -I"$base/Strata-dsv41-experiment/include" -I"$base/strata-p40-20261102/cuda-compat-include" "$base/deepseek-p40-postboot-20261108/worker-q8k-trace.cu" "$base/engine-review-20261107/ds4_iq-sm61.o" "$base/engine-review-20261107/ds4_tq.o" -lpthread -o "$base/deepseek-p40-postboot-20261108/worker-q8k-trace"
