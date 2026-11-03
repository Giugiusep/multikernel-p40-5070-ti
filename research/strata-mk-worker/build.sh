#!/bin/bash
set -euo pipefail
base=/home/shiba/multikernel-experiment/research
old=$base/strata-p40-20261102
src=$base/Strata-dsv41-experiment
here=$base/strata-mk-worker
/usr/local/cuda-12.8/bin/nvcc -ccbin /usr/bin/g++-13 -O2 --use_fast_math -std=c++17 -arch=sm_61 -DSTRATA_EXPERIMENTAL_SM60=1 -DSTRATA_VERIFY_MAX_T=8 -I"$old/cuda-compat-include" -I"$src/include" -I"$src/third_party/ggml" "$here/worker.cpp" "$old/speed-objects/"*.o -o "$here/worker"
