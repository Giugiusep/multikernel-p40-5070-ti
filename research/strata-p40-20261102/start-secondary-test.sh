#!/bin/bash
set -euo pipefail
[[ $EUID -eq 0 ]] || { echo 'Run with sudo'; exit 1; }
base=/home/shiba/multikernel-experiment
testroot=$base/research/strata-p40-20261102
[[ $(uname -r) == 7.0.0-mk2-shiba-mk1 ]]
[[ -x $testroot/rootfs/opt/strata/mmvq_multi_parity ]]
[[ ! -e /sys/bus/pci/devices/0000:00:11.0/driver ]]
modprobe lazy_cma
if ! grep -q '^daxfs ' /proc/modules; then
    insmod "$base/daxfs-v0.1.0/daxfs/daxfs.ko"
fi
kerf=$base/venv/bin/kerf
"$kerf" init --cpus=8-11 --memory=6GB --devices=0000:00:11.0
"$kerf" create rpc-p40 --cpus=8-11 --memory=1792MB --devices=0000:00:11.0
"$kerf" load rpc-p40 --kernel=/boot/vmlinuz-7.0.0-mk2-shiba-mk1 --initrd="$base/artifacts/secondary-daxfs-bootstrap-initramfs.cpio.gz" --rootfs-dir="$testroot/rootfs" --entrypoint=/sbin/entrypoint --cmdline='rdinit=/init loglevel=7' --console=mktty0
"$kerf" exec rpc-p40
"$kerf" show rpc-p40
