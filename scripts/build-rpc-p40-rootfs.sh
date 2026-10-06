#!/usr/bin/env bash
set -euo pipefail

base=/home/shiba/multikernel-experiment
root="$base/rpc-p40-root"
rpc=/home/shiba/llama.cpp/build-p40-rpc/bin
driver="$base/NVIDIA-580.173.02-extracted"
release=$(cat "$base/artifacts/kernel-release.txt")

mkdir -p "$root/bin" "$root/sbin" "$root/lib64" "$root/lib" \
    "$root/lib/modules/$release" "$root/opt/rpc/lib" \
    "$root/dev/pts" "$root/dev/shm" "$root/proc" "$root/sys" \
    "$root/run" "$root/tmp"
cp -L /usr/bin/busybox "$root/bin/busybox"
cp "$base/scripts/rpc-p40-entrypoint" "$root/sbin/entrypoint"
cp "$driver/nvidia-modprobe" "$root/sbin/nvidia-modprobe"
cp "$driver/nvidia-smi" "$root/bin/nvidia-smi"
cp "$driver/kernel/nvidia.ko" "$root/lib/modules/$release/nvidia.ko"
cp "$driver/kernel/nvidia-uvm.ko" "$root/lib/modules/$release/nvidia-uvm.ko"
cp "$driver/libnvidia-ml.so.580.173.02" "$root/lib/libnvidia-ml.so.580.173.02"
cp "$driver/libcuda.so.580.173.02" "$root/lib/libcuda.so.580.173.02"
ln -sfn libnvidia-ml.so.580.173.02 "$root/lib/libnvidia-ml.so.1"
ln -sfn libcuda.so.580.173.02 "$root/lib/libcuda.so.1"
ln -sfn libcuda.so.1 "$root/lib/libcuda.so"

cp -aL "$rpc/ggml-rpc-server" "$root/opt/rpc/ggml-rpc-server"
find "$rpc" -maxdepth 1 -type f -name '*.so*' -exec cp -aL {} "$root/opt/rpc/lib/" \;
for link in "$rpc"/*.so "$rpc"/*.so.0; do
    [[ -L "$link" ]] || continue
    ln -sfn "$(basename "$(readlink -f "$link")")" "$root/opt/rpc/lib/$(basename "$link")"
done

cp -L /usr/lib/x86_64-linux-gnu/libcudart.so.12 "$root/lib/libcudart.so.12"
cp -L /usr/lib/x86_64-linux-gnu/libcublas.so.12 "$root/lib/libcublas.so.12"
cp -L /usr/lib/x86_64-linux-gnu/libcublasLt.so.12 "$root/lib/libcublasLt.so.12"
for program in "$rpc/ggml-rpc-server"; do
    while IFS= read -r dep; do
        case "$(basename "$dep")" in
            libcuda.so.1|libcudart.so.12|libcublas.so.12|libcublasLt.so.12|libggml*) continue ;;
        esac
        cp -L "$dep" "$root/lib/$(basename "$dep")"
    done < <(LD_LIBRARY_PATH="$rpc" ldd "$program" | awk '/=> \// {print $3} /^[[:space:]]*\// {print $1}' | sort -u)
done
cp -L /lib/x86_64-linux-gnu/ld-linux-x86-64.so.2 "$root/lib64/ld-linux-x86-64.so.2"

for applet in sh setsid cttyhack uname cat mount poweroff sync ls grep tail sleep kill; do
    ln -sfn busybox "$root/bin/$applet"
done
chmod 0755 "$root/bin/busybox" "$root/bin/nvidia-smi" \
    "$root/sbin/entrypoint" "$root/sbin/nvidia-modprobe" \
    "$root/opt/rpc/ggml-rpc-server"

sha256sum "$root/opt/rpc/ggml-rpc-server" \
    > "$base/logs/rpc-p40-rootfs-sha256.txt"
du -sb "$root" | tee "$base/logs/rpc-p40-rootfs-size.txt"
PYTHONPATH="$base/kerf/src" "$base/venv/bin/python" - "$root" <<'PY'
import sys
from kerf.daxfs.mkdaxfs import DaxfsBuilder

builder = DaxfsBuilder(sys.argv[1])
builder.scan()
builder.build_tree()
builder.calculate_offsets()
print(f"DAXFS_REQUIRED_BYTES={builder.calculate_total_size()}")
PY
