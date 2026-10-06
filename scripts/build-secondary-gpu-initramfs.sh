#!/usr/bin/env bash
set -euo pipefail

base=/home/shiba/multikernel-experiment
build="$base/build"
driver="$base/NVIDIA-580.173.02-extracted"
release=$(cat "$base/artifacts/kernel-release.txt")
manifest="$base/artifacts/secondary-gpu.cpio-list"
image="$base/artifacts/secondary-p40-580-initramfs.cpio.gz"
cuda_test="$base/artifacts/cuda-secondary-smoke"

/usr/local/cuda-12.8/bin/nvcc -std=c++17 -O2 -arch=sm_60 \
    -ccbin=/usr/bin/g++-12 --cudart=static \
    "$base/tests/cuda-secondary-smoke.cu" -o "$cuda_test"

mkdir -p "$base/artifacts" "$base/logs"
cat > "$manifest" <<EOF
dir /bin 0755 0 0
dir /sbin 0755 0 0
dir /lib 0755 0 0
dir /lib64 0755 0 0
dir /lib/x86_64-linux-gnu 0755 0 0
dir /lib/modules 0755 0 0
dir /lib/modules/$release 0755 0 0
dir /dev 0755 0 0
dir /proc 0755 0 0
dir /sys 0755 0 0
dir /run 0755 0 0
dir /tmp 1777 0 0
nod /dev/console 0600 0 0 c 5 1
nod /dev/null 0666 0 0 c 1 3
file /bin/busybox /usr/bin/busybox 0755 0 0
file /init $base/scripts/secondary-gpu-init 0755 0 0
file /sbin/load-p40-driver $base/scripts/load-p40-driver 0755 0 0
file /sbin/nvidia-modprobe $driver/nvidia-modprobe 0755 0 0
file /bin/nvidia-smi $driver/nvidia-smi 0755 0 0
file /bin/cuda-secondary-smoke $cuda_test 0755 0 0
file /lib/modules/$release/nvidia.ko $driver/kernel/nvidia.ko 0644 0 0
file /lib/modules/$release/nvidia-uvm.ko $driver/kernel/nvidia-uvm.ko 0644 0 0
file /lib/libnvidia-ml.so.580.173.02 $driver/libnvidia-ml.so.580.173.02 0755 0 0
slink /lib/libnvidia-ml.so.1 libnvidia-ml.so.580.173.02 0777 0 0
file /lib/libcuda.so.580.173.02 $driver/libcuda.so.580.173.02 0755 0 0
slink /lib/libcuda.so.1 libcuda.so.580.173.02 0777 0 0
slink /lib/libcuda.so libcuda.so.1 0777 0 0
file /lib64/ld-linux-x86-64.so.2 /lib/x86_64-linux-gnu/ld-linux-x86-64.so.2 0755 0 0
file /lib/x86_64-linux-gnu/libc.so.6 /lib/x86_64-linux-gnu/libc.so.6 0755 0 0
file /lib/x86_64-linux-gnu/libpthread.so.0 /lib/x86_64-linux-gnu/libpthread.so.0 0755 0 0
file /lib/x86_64-linux-gnu/libm.so.6 /lib/x86_64-linux-gnu/libm.so.6 0755 0 0
file /lib/x86_64-linux-gnu/libdl.so.2 /lib/x86_64-linux-gnu/libdl.so.2 0755 0 0
file /lib/x86_64-linux-gnu/librt.so.1 /lib/x86_64-linux-gnu/librt.so.1 0755 0 0
EOF
while IFS= read -r applet; do
    [[ "$applet" == busybox ]] && continue
    printf 'slink /bin/%s busybox 0777 0 0\n' "$applet" >> "$manifest"
done < <(/usr/bin/busybox --list)

"$build/usr/gen_init_cpio" "$manifest" | gzip -n > "$image"
gzip -t "$image"
gzip -dc "$image" | cpio -it > "$base/logs/secondary-gpu-initramfs-contents.txt"
ldd "$cuda_test" > "$base/logs/cuda-secondary-smoke-ldd.txt"
sha256sum "$driver/kernel/nvidia.ko" "$driver/kernel/nvidia-uvm.ko" \
    "$cuda_test" "$image" > "$base/logs/secondary-gpu-artifact-sha256.txt"
printf 'Built %s (%s bytes)\n' "$image" "$(stat -c %s "$image")"
