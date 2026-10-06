#!/usr/bin/env bash
set -euo pipefail
base=/home/shiba/multikernel-experiment
kerf="$base/venv/bin/kerf"
expected=$(cat "$base/artifacts/kernel-release.txt")
[[ $(id -u) == 0 ]] || { echo 'Run through sudo.' >&2; exit 1; }
[[ $(hostname) == shiba2th && $(uname -r) == "$expected" ]] || {
    echo "Expected shiba2th running $expected; refusing this environment." >&2
    exit 1
}
case "${1:-inventory}" in
    inventory)
        uname -a
        lspci -Dnnk -d 10de:
        nvidia-smi -L
        cat /sys/devices/system/cpu/online
        head /proc/meminfo
        modinfo lazy_cma
        ;;
    start)
        # First run only; never allocate PCI devices in this CPU/RAM test.
        [[ $(cat /sys/devices/system/cpu/online) == 0-11 ]] || {
            echo 'CPU topology changed; inspect it before allocation.' >&2; exit 1;
        }
        modprobe lazy_cma
        "$kerf" init --cpus=10-11 --memory=2GB --devices=none
        "$kerf" create cpu-test --cpus=10-11 --memory=1GB
        "$kerf" load cpu-test --kernel="/boot/vmlinuz-$expected" \
            --initrd="$base/artifacts/secondary-cpu-initramfs.cpio.gz" \
            --cmdline='rdinit=/init loglevel=7' --console=mktty0
        "$kerf" exec cpu-test
        "$kerf" show cpu-test
        echo "Attach using: $kerf console cpu-test"
        ;;
    stop)
        "$kerf" kill cpu-test
        "$kerf" unload cpu-test
        "$kerf" delete cpu-test
        "$kerf" init --cpus=none --memory=none --devices=none
        "$kerf" show
        cat /sys/devices/system/cpu/online
        [[ $(cat /sys/devices/system/cpu/online) == 0-11 ]] || {
            echo 'CPU return verification failed.' >&2; exit 1;
        }
        echo 'CPU return verified. Inspect pool state and memory accounting in the output.'
        ;;
    *) echo 'Usage: postboot-cpu-test.sh inventory|start|stop' >&2; exit 2 ;;
esac
