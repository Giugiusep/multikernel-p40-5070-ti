#!/bin/bash
set -euo pipefail
here=/home/shiba/multikernel-experiment/research/strata-mk-worker
ref=/home/shiba/multikernel-experiment/research/Strata
/usr/bin/g++-13 -O2 -std=c++20 -I"$ref/include" -I"$ref/third_party/llama.cpp/ggml/include" -I"$ref/third_party/llama.cpp/ggml/src" -c "$here/cpu_math.cpp" -o "$here/cpu_math_oracle.o"
libs=("$here/cpu_math_oracle.o" "$ref/build/libstrata_kernels_cpu.a" "$ref/build/ggml/src/libggml-cpu.a" "$ref/build/ggml/src/libggml-base.a" -fopenmp -lpthread -lm)
/usr/bin/g++-13 -O2 "$here/cpu_reference.cpp" "${libs[@]}" -o "$here/cpu_reference"
cp "$here/cpu_reference" "$here/cpu_reference_pool"
/usr/bin/g++-13 -O2 -march=native -ffp-contract=off "$here/emulate_gu.cpp" "${libs[@]}" -o "$here/emulate_gu"
