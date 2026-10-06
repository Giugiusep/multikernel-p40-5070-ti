#!/usr/bin/env bash
set -euo pipefail

base=/home/shiba/multikernel-experiment
root="$base/nemotron-p40-root"
llama=/home/shiba/llama.cpp/build-p40/bin
driver="$base/NVIDIA-580.173.02-extracted"
release=$(cat "$base/artifacts/kernel-release.txt")
model=$(readlink -f /home/shiba/.cache/huggingface/hub/models--unsloth--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF/snapshots/ff84a7685a5cf4fb8f1c4959129999d808ff3ba5/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q8_0.gguf)
model_name=NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q8_0.gguf

mkdir -p "$root/bin" "$root/sbin" "$root/lib64" "$root/lib" \
    "$root/lib/modules/$release" "$root/opt/llama/lib" "$root/models" \
    "$root/dev/pts" "$root/dev/shm" "$root/proc" "$root/sys" \
    "$root/run" "$root/tmp"
cp -L /usr/bin/busybox "$root/bin/busybox"
cp "$base/scripts/llama-p40-entrypoint" "$root/sbin/entrypoint"
cp "$driver/nvidia-modprobe" "$root/sbin/nvidia-modprobe"
cp "$driver/nvidia-smi" "$root/bin/nvidia-smi"
cp "$driver/kernel/nvidia.ko" "$root/lib/modules/$release/nvidia.ko"
cp "$driver/kernel/nvidia-uvm.ko" "$root/lib/modules/$release/nvidia-uvm.ko"
cp "$driver/libnvidia-ml.so.580.173.02" "$root/lib/libnvidia-ml.so.580.173.02"
cp "$driver/libcuda.so.580.173.02" "$root/lib/libcuda.so.580.173.02"
ln -sfn libnvidia-ml.so.580.173.02 "$root/lib/libnvidia-ml.so.1"
ln -sfn libcuda.so.580.173.02 "$root/lib/libcuda.so.1"
ln -sfn libcuda.so.1 "$root/lib/libcuda.so"

cp -aL "$llama"/llama-cli "$root/opt/llama/llama-cli"
find "$llama" -maxdepth 1 -type f -name '*.so*' -exec cp -aL {} "$root/opt/llama/lib/" \;
for link in "$llama"/*.so "$llama"/*.so.0; do
    [[ -L "$link" ]] || continue
    ln -sfn "$(basename "$(readlink -f "$link")")" "$root/opt/llama/lib/$(basename "$link")"
done

cp -L /usr/lib/x86_64-linux-gnu/libcudart.so.12 "$root/lib/libcudart.so.12"
cp -L /usr/lib/x86_64-linux-gnu/libcublas.so.12 "$root/lib/libcublas.so.12"
cp -L /usr/lib/x86_64-linux-gnu/libcublasLt.so.12 "$root/lib/libcublasLt.so.12"

while IFS= read -r dep; do
    case "$(basename "$dep")" in
        libcuda.so.1|libcudart.so.12|libcublas.so.12|libcublasLt.so.12|libggml*|libllama*|libmtmd*) continue ;;
    esac
    cp -L "$dep" "$root/lib/$(basename "$dep")"
done < <(LD_LIBRARY_PATH="$llama" ldd "$llama/llama-cli" | awk '/=> \// {print $3} /^[[:space:]]*\// {print $1}' | sort -u)
cp -L /lib/x86_64-linux-gnu/ld-linux-x86-64.so.2 "$root/lib64/ld-linux-x86-64.so.2"

for applet in sh setsid cttyhack uname cat mount poweroff sync ls grep tail; do
    ln -sfn busybox "$root/bin/$applet"
done

if [[ ! -e "$root/models/$model_name" ]]; then
    ln "$model" "$root/models/$model_name"
fi
[[ $(stat -c '%d:%i' "$model") == $(stat -c '%d:%i' "$root/models/$model_name") ]]

chmod 0755 "$root/bin/busybox" "$root/bin/nvidia-smi" \
    "$root/sbin/entrypoint" "$root/sbin/nvidia-modprobe" \
    "$root/opt/llama/llama-cli"
sha256sum "$model" "$root/opt/llama/llama-cli" \
    > "$base/logs/llama-p40-rootfs-sha256.txt"
du -sb "$root" | tee "$base/logs/llama-p40-rootfs-size.txt"

PYTHONPATH="$base/kerf/src" "$base/venv/bin/python" - "$root" <<'PY'
import sys
from kerf.daxfs.mkdaxfs import DaxfsBuilder

builder = DaxfsBuilder(sys.argv[1])
builder.scan()
builder.build_tree()
builder.calculate_offsets()
print(f"DAXFS_REQUIRED_BYTES={builder.calculate_total_size()}")
PY
