#!/bin/bash
set -euo pipefail
base=/home/shiba/multikernel-experiment/research
here=$base/strata-mk-worker
old=$base/strata-p40-20261102
src=$base/Strata-dsv41-experiment
reference=$base/Strata
flags=(-ccbin /usr/bin/g++-13 -O2 -std=c++17 -arch=sm_61 -DSTRATA_EXPERIMENTAL_SM60=1 -DSTRATA_VERIFY_MAX_T=8 -I"$old/cuda-compat-include" -I"$src/include" -I"$src/third_party/ggml")
cat > "$here/compat_iq.cu" <<'EOF'
#include "/home/shiba/multikernel-experiment/research/Strata-dsv41-experiment/src/kernels/cuda/iq_kernels.cu"
#include "compat-kernels.inc"
#include "grouped-kernels.inc"
EOF
# No fast-math flag in this correctness experiment.
if [[ ! -f $here/compat_iq.o || $here/compat-kernels.inc -nt $here/compat_iq.o || $here/grouped-kernels.inc -nt $here/compat_iq.o || $src/src/kernels/cuda/iq_kernels.cu -nt $here/compat_iq.o ]]; then
    /usr/local/cuda-12.8/bin/nvcc "${flags[@]}" -c "$here/compat_iq.cu" -o "$here/compat_iq.o"
fi
/usr/bin/g++-13 -O2 -std=c++20 -I"$reference/include" -I"$reference/third_party/llama.cpp/ggml/include" -I"$reference/third_party/llama.cpp/ggml/src" -c "$here/cpu_math.cpp" -o "$here/cpu_math.o"
/usr/local/cuda-12.8/bin/nvcc "${flags[@]}" -DMK_ENABLE_CPU_COMPAT=1 "$here/worker.cpp" "$here/compat_iq.o" "$here/cpu_math.o" "$old/speed-objects/native_mmvq.o" "$old/speed-objects/verify_kernels.o" "$reference/build/libstrata_kernels_cpu.a" "$reference/build/ggml/src/libggml-cpu.a" "$reference/build/ggml/src/libggml-base.a" -lgomp -lpthread -lm -o "$here/worker-compat"
cp "$here/worker-compat" "$here/worker-qualified"
