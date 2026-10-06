#!/usr/bin/env bash
set -euo pipefail

root=/home/shiba
llama="$root/llama.cpp/build-main-cuda-rpc-128/bin/llama-cli"
model="$root/.cache/huggingface/hub/models--unsloth--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF/snapshots/ff84a7685a5cf4fb8f1c4959129999d808ff3ba5/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q8_0.gguf"
logs="$root/multikernel-experiment/logs"

for run in 1 2; do
    output="$logs/nemotron-q8-48-52-server-write-fast-repeat${run}-20260920"
    /usr/bin/time -f 'wall_seconds=%e exit=%x' -o "${output}-time.txt" \
        timeout 600 env \
        LD_LIBRARY_PATH="$root/llama.cpp/build-main-cuda-rpc-128/bin" \
        GGML_MK_VSOCK_NO_READ_SLEEP=1 GGML_RPC_NO_RDMA=1 \
        "$llama" -m "$model" \
        --rpc mkvsock:1:5002 --device CUDA0,RPC0 --tensor-split 48,52 \
        -ngl 99 -c 256 -b 128 -ub 64 -n 16 --seed 42 --temp 0 -st \
        -p 'Explain briefly why PCIe transfers are slower than on-device memory.' \
        > "${output}.log" 2>&1
    rg '\[ Prompt:' "${output}.log"
    cat "${output}-time.txt"
done
