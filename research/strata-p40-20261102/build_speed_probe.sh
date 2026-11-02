#!/bin/bash
set -euo pipefail
root=/home/shiba/multikernel-experiment/research/strata-p40-20261102
src=/home/shiba/multikernel-experiment/research/Strata-dsv41-experiment
mkdir -p "$root/speed-objects"
flags=(-ccbin /usr/bin/g++-13 -O2 --use_fast_math -std=c++17 -arch=sm_61 -DSTRATA_EXPERIMENTAL_SM60=1 -DSTRATA_VERIFY_MAX_T=8 -I"$root/cuda-compat-include" -I"$src/include" -I"$src/third_party/ggml")
for unit in native_mmvq iq_kernels verify_kernels; do
    object="$root/speed-objects/$unit.o"
    if [[ ! -f $object || $src/src/kernels/cuda/$unit.cu -nt $object ]]; then
        /usr/local/cuda-12.8/bin/nvcc "${flags[@]}" -c "$src/src/kernels/cuda/$unit.cu" -o "$object"
    fi
done
/usr/local/cuda-12.8/bin/nvcc "${flags[@]}" "$root/p40_iq_speed.cpp" "$root/speed-objects/"*.o -o "$root/p40_iq_speed"
