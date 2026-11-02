#!/bin/bash
# Capability diagnostic: unmodified upstream source currently fails on WMMA for SM61.
set -euo pipefail
root=/home/shiba/multikernel-experiment/research/strata-p40-20261102
src=/home/shiba/multikernel-experiment/research/Strata-dsv41-experiment
/usr/local/cuda-12.8/bin/nvcc -ccbin /usr/bin/g++-13 -O2 -std=c++17 -arch=sm_61 -I"$root/cuda-compat-include" -I"$src/include" -I"$src/third_party/ggml" -c "$src/src/dsv41/ds4_iq.cu" -o "$root/ds4_iq-sm61.o"
