#!/usr/bin/env bash
set -euo pipefail

base=/home/shiba/multikernel-experiment
kerf="$base/venv/bin/kerf"
expected=$(cat "$base/artifacts/kernel-release.txt")
instance=p40-driver
device=0000:00:11.0

[[ $(id -u) == 0 ]] || { echo 'Run through sudo.' >&2; exit 1; }
[[ $(hostname) == shiba2th && $(uname -r) == "$expected" ]] || {
    echo "Expected shiba2th running $expected; refusing this environment." >&2
    exit 1
}

case "${1:-inventory}" in
    inventory)
        uname -a
        cat /sys/devices/system/cpu/online
        lspci -Dnnk -s "$device"
        nvidia-smi -L
        "$kerf" show
        ;;
    start)
        [[ $(cat /sys/devices/system/cpu/online) == 0-11 ]] || {
            echo 'CPU topology changed; inspect it before allocation.' >&2
            exit 1
        }
        test ! -e "/sys/bus/pci/devices/$device/driver" || {
            echo "$device is bound on the primary; refusing handoff." >&2
            exit 1
        }
        modprobe lazy_cma
        # This VM's Lazy CMA pool can still obtain 2 GiB after normal boot,
        # while a 6 GiB contiguous request fails after userspace has fragmented
        # guest memory.  The smoke workload uses less than 128 MiB of heap.
        "$kerf" init --cpus=10-11 --memory=2GB --devices="$device"
        "$kerf" create "$instance" --cpus=10-11 --memory=1536MB --devices="$device"
        "$kerf" load "$instance" --kernel="/boot/vmlinuz-$expected" \
            --initrd="$base/artifacts/secondary-p40-580-initramfs.cpio.gz" \
            --cmdline='rdinit=/init loglevel=7' --console=mktty0
        "$kerf" exec "$instance"
        "$kerf" show "$instance"
        echo "Attach using: $kerf console $instance"
        ;;
    stop)
        # A secondary can already be in "loaded" after poweroff -f from its
        # console.  Kerf correctly rejects kill in that state; continue with
        # unload and resource return.
        "$kerf" kill "$instance" || true
        "$kerf" unload "$instance"
        "$kerf" delete "$instance"
        "$kerf" init --cpus=none --memory=none --devices=none
        "$kerf" show
        cat /sys/devices/system/cpu/online
        [[ $(cat /sys/devices/system/cpu/online) == 0-11 ]] || {
            echo 'CPU return verification failed.' >&2
            exit 1
        }
        echo 'P40, CPU, and memory return completed.'
        ;;
    *)
        echo 'Usage: postboot-p40-driver-test.sh inventory|start|stop' >&2
        exit 2
        ;;
esac
