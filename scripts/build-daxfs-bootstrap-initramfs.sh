#!/usr/bin/env bash
set -euo pipefail

base=/home/shiba/multikernel-experiment
release=$(cat "$base/artifacts/kernel-release.txt")
manifest="$base/artifacts/secondary-daxfs-bootstrap.cpio-list"
image="$base/artifacts/secondary-daxfs-bootstrap-initramfs.cpio.gz"
module="$base/daxfs-v0.1.0/daxfs/daxfs.ko"

cat > "$manifest" <<EOF
dir /bin 0755 0 0
dir /sbin 0755 0 0
dir /lib 0755 0 0
dir /lib/modules 0755 0 0
dir /lib/modules/$release 0755 0 0
dir /dev 0755 0 0
dir /proc 0755 0 0
dir /sys 0755 0 0
dir /newroot 0755 0 0
nod /dev/console 0600 0 0 c 5 1
nod /dev/null 0666 0 0 c 1 3
file /bin/busybox /usr/bin/busybox 0755 0 0
file /init $base/scripts/secondary-daxfs-bootstrap-init 0755 0 0
file /lib/modules/$release/daxfs.ko $module 0644 0 0
EOF
while IFS= read -r applet; do
    [[ "$applet" == busybox ]] && continue
    printf 'slink /bin/%s busybox 0777 0 0\n' "$applet" >> "$manifest"
done < <(/usr/bin/busybox --list)

"$base/build/usr/gen_init_cpio" "$manifest" | gzip -n > "$image"
gzip -t "$image"
sha256sum "$module" "$image" > "$base/logs/daxfs-bootstrap-sha256.txt"
printf 'Built %s (%s bytes)\n' "$image" "$(stat -c %s "$image")"
