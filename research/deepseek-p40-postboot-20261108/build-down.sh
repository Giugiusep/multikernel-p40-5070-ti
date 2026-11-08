#!/bin/bash
set -euo pipefail
base=/home/shiba/multikernel-experiment/research
here=$base/deepseek-p40-postboot-20261108
src=$base/Strata-dsv41-experiment
flags=(-ccbin /usr/bin/g++-13 -O2 -std=c++17 -arch=sm_61 -DSTRATA_EXPERIMENTAL_SM60=1 -DSTRATA_VERIFY_MAX_T=8 -I"$base/strata-p40-20261102/cuda-compat-include" -I"$src/include" -I"$src/third_party/ggml")
/usr/local/cuda-12.8/bin/nvcc "${flags[@]}" -c "$here/exact-down.cu" -o "$here/exact-down.o"
/usr/local/cuda-12.8/bin/nvcc "${flags[@]}" -I"$src/src/dsv41" "$here/worker-down.cu" "$here/exact-down.o" "$base/engine-review-20261107/ds4_iq-sm61.o" "$base/engine-review-20261107/ds4_tq.o" -lpthread -o "$here/worker-down"
/usr/bin/cmake --build "$base/deepseek-decode-build" --target strata-dsv41 -j 1
