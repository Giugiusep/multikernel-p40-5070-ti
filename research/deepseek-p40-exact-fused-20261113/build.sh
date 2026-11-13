#!/bin/bash
set -euo pipefail
base=/home/shiba/multikernel-experiment/research
here=$base/deepseek-p40-exact-fused-20261113
src=$base/Strata-dsv41-experiment
/usr/local/cuda-12.8/bin/nvcc -ccbin /usr/bin/g++-13 -O2 -std=c++17 -arch=sm_61 -DSTRATA_EXPERIMENTAL_SM60=1 -DSTRATA_VERIFY_MAX_T=8 -I"$base/strata-p40-20261102/cuda-compat-include" -I"$src/include" -I"$src/third_party/ggml" -I"$src/src/dsv41" "$here/worker-fused.cu" "$base/deepseek-p40-compact-20261108/exact-down.o" "$base/engine-review-20261107/ds4_iq-sm61.o" "$base/engine-review-20261107/ds4_tq.o" -lpthread -o "$here/worker-fused"
rm -f "$base/deepseek-decode-build/strata-dsv41"
/usr/bin/cmake --build "$base/deepseek-decode-build" --target strata-dsv41 -j1
cp "$base/deepseek-decode-build/strata-dsv41" "$here/strata-dsv41-fused"
cp "$here/strata-dsv41-baseline" "$base/deepseek-decode-build/strata-dsv41"
