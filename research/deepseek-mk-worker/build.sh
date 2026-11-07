#!/bin/bash
set -euo pipefail
base=/home/shiba/multikernel-experiment/research
out=$base/deepseek-mk-worker
src=$base/Strata-dsv41-experiment
probe=$base/engine-review-20261107
/usr/local/cuda-12.8/bin/nvcc -ccbin /usr/bin/g++-13 -O2 -std=c++17 -arch=sm_61 -I"$src/src/dsv41" -I"$src/include" -I"$base/strata-p40-20261102/cuda-compat-include" "$out/worker.cu" "$probe/ds4_iq-sm61.o" "$probe/ds4_tq.o" -lpthread -o "$out/worker"
/usr/bin/g++-13 -O2 -std=c++17 -I"$base/Strata/third_party/llama.cpp/ggml/include" "$out/decode.cpp" "$base/Strata/build/ggml/src/libggml-base.a" -lpthread -ldl -o "$out/decode"
