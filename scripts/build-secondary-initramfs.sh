#!/usr/bin/env bash
set -euo pipefail
base=/home/shiba/multikernel-experiment
build="$base/build"
mkdir -p "$base/artifacts"
manifest="$base/artifacts/secondary.cpio-list"
cat > "$manifest" <<EOF
dir /bin 0755 0 0
dir /sbin 0755 0 0
dir /dev 0755 0 0
dir /proc 0755 0 0
dir /sys 0755 0 0
dir /run 0755 0 0
dir /tmp 1777 0 0
nod /dev/console 0600 0 0 c 5 1
nod /dev/null 0666 0 0 c 1 3
file /bin/busybox /usr/bin/busybox 0755 0 0
file /init $base/scripts/secondary-init 0755 0 0
EOF
while IFS= read -r applet; do
    [[ "$applet" == busybox ]] && continue
    printf 'slink /bin/%s busybox 0777 0 0\n' "$applet" >> "$manifest"
done < <(/usr/bin/busybox --list)
"$build/usr/gen_init_cpio" "$manifest" | gzip -n > "$base/artifacts/secondary-cpu-initramfs.cpio.gz"
gzip -t "$base/artifacts/secondary-cpu-initramfs.cpio.gz"
gzip -dc "$base/artifacts/secondary-cpu-initramfs.cpio.gz" | cpio -it > "$base/logs/secondary-initramfs-contents.txt"
