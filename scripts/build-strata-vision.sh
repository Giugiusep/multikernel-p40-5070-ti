#!/bin/bash
set -euo pipefail
root=/home/shiba/multikernel-experiment/research/Strata
/usr/bin/cmake -S "$root/tools/vision" -B "$root/build-vision-cpu" \
  -DLLAMA_DIR="$root/third_party/llama.cpp" -DSTRATA_VISION_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
/usr/bin/cmake --build "$root/build-vision-cpu" --target strata-vision -j8
