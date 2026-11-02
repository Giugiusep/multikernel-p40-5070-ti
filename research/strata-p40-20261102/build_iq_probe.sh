#!/bin/bash
set -euo pipefail
root=/home/shiba/multikernel-experiment/research/strata-p40-20261102
src=/home/shiba/multikernel-experiment/research/Strata-dsv41-experiment
/usr/local/cuda-12.8/bin/nvcc -ccbin /usr/bin/g++-13 -O2 --use_fast_math -std=c++17 -arch=sm_61 -DSTRATA_EXPERIMENTAL_SM60=1 -DSTRATA_VERIFY_MAX_T=8 -I"$root/cuda-compat-include" -I"$src/include" -I"$src/third_party/ggml" "$src/src/kernels/cuda/native_mmvq.cu" "$src/src/kernels/cuda/iq_kernels.cu" "$src/src/kernels/cuda/verify_kernels.cu" "$src/src/kernels/iq_parity.cpp" -o "$root/iq_parity"
