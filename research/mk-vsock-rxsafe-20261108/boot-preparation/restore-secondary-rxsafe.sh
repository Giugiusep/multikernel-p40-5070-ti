#!/usr/bin/env bash
set -euo pipefail
base=/home/shiba/multikernel-experiment
kernel=/boot/vmlinuz-7.0.0-mk2-shiba-mk1-rxsafe1
[[ $EUID -eq 0 && $(hostname) == shiba2th && $(uname -r) == 7.0.0-mk2-shiba-mk1 ]] || exit 1
[[ $(uname -v) == *rxsafe1* ]] || { echo 'RX-safe primary is not running; refuse mixed-kernel test' >&2; exit 1; }
[[ ! -e /sys/fs/multikernel/instances/rpc-p40 ]] || { echo 'Existing secondary must be inspected; no reset attempted' >&2; exit 1; }
test ! -e /sys/bus/pci/devices/0000:00:11.0/driver
test -s "$kernel"
modprobe lazy_cma
if [[ ! -d /sys/module/daxfs ]]; then insmod "$base/daxfs-v0.1.0/daxfs/daxfs.ko"; fi
kerf=$base/venv/bin/kerf
"$kerf" init --cpus=8-11 --memory=3GB --devices=0000:00:11.0
"$kerf" create rpc-p40 --cpus=8-11 --memory=1792MB --devices=0000:00:11.0
"$kerf" load rpc-p40 --kernel="$kernel" --initrd="$base/artifacts/secondary-daxfs-bootstrap-initramfs.cpio.gz" --rootfs-dir="$base/rpc-p40-root" --entrypoint=/sbin/entrypoint --cmdline='rdinit=/init loglevel=7' --console=mktty0
"$kerf" exec rpc-p40
"$kerf" show rpc-p40
