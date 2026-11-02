#!/bin/bash
set -euo pipefail
root=/home/shiba/multikernel-experiment/research/strata-p40-20261102
src=/home/shiba/multikernel-experiment/research/Strata-dsv41-experiment
bash "$root/build_speed_probe.sh"
for program in p40_expert_speed p40_expert_diag; do
    /usr/local/cuda-12.8/bin/nvcc -ccbin /usr/bin/g++-13 -O2 --use_fast_math -std=c++17 -arch=sm_61 -DSTRATA_EXPERIMENTAL_SM60=1 -DSTRATA_VERIFY_MAX_T=8 -I"$root/cuda-compat-include" -I"$src/include" -I"$src/third_party/ggml" "$root/$program.cpp" "$root/speed-objects/"*.o -o "$root/$program"
done
