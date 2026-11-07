#!/bin/bash
set -euo pipefail
base=/home/shiba/multikernel-experiment/research
src=$base/Strata-dsv41-experiment
out=$base/engine-review-20261107
inc=(-I"$src/include" -I"$src/third_party/ggml" -I"$src/src/kernels/cpu" -I"$base/Strata/third_party/llama.cpp/ggml/include" -I"$base/Strata/third_party/llama.cpp/ggml/src" -I"$base/strata-p40-20261102/cuda-compat-include" -I/usr/local/cuda-12.8/include)
python3 "$out/prepare_candidate.py"
/usr/local/cuda-12.8/bin/nvcc -ccbin /usr/bin/g++-13 -O2 -std=c++17 -arch=sm_61 -DSTRATA_DS4_SCALAR_ONLY=1 "${inc[@]}" -I"$src/src/dsv41" -c "$out/ds4_iq-sm61.cu" -o "$out/ds4_iq-sm61.o"
for name in ds4_iq_cpu ds4_gg_cpu ds4_tq_cpu; do
 /usr/bin/g++-13 -O2 -std=c++17 -DSTRATA_DSV41_TQ_CPU=1 "${inc[@]}" -c "$src/src/dsv41/$name.cpp" -o "$out/$name.o"
done
/usr/local/cuda-12.8/bin/nvcc -ccbin /usr/bin/g++-13 -O2 -std=c++17 -arch=sm_61 "${inc[@]}" -c "$src/src/dsv41/ds4_tq.cu" -o "$out/ds4_tq.o"
/usr/bin/g++-13 -O2 -std=c++17 -DSTRATA_DSV41_TQ_CPU=1 "${inc[@]}" -c "$src/tests/dsv41/iq_expert_test.cpp" -o "$out/iq_expert_test.o"
for name in q2_avx1 q2_avx2 hadamard expert_layout iq_avx2 iq_avx512 tq_cpu tq_avx2 tq_avx512; do
 arch=(-march=native); if [[ "$name" == q2_avx1 ]]; then arch=(-msse4.2 -mavx -mno-avx2 -mno-fma -mno-f16c); fi
 if [[ ! -f "$out/$name.o" ]]; then /usr/bin/g++-13 -O2 -ffunction-sections -fdata-sections -std=c++17 "${arch[@]}" "${inc[@]}" -c "$src/src/kernels/cpu/$name.cpp" -o "$out/$name.o"; fi
done
/usr/local/cuda-12.8/bin/nvcc -ccbin /usr/bin/g++-13 -O2 -std=c++17 -arch=sm_61 "$out/iq_expert_test.o" "$out/ds4_iq-sm61.o" "$out/ds4_iq_cpu.o" "$out/ds4_gg_cpu.o" "$out/ds4_tq_cpu.o" "$out/ds4_tq.o" "$out/q2_avx1.o" "$out/q2_avx2.o" "$out/hadamard.o" "$out/expert_layout.o" "$out/iq_avx2.o" "$out/iq_avx512.o" "$out/tq_cpu.o" "$out/tq_avx2.o" "$out/tq_avx512.o" "$base/Strata/build/ggml/src/libggml-cpu.a" "$base/Strata/build/ggml/src/libggml-base.a" -Xlinker --gc-sections -lgomp -lpthread -lm -o "$out/ds4-iq-sm61-test"
