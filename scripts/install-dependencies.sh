#!/usr/bin/env bash
set -euo pipefail

# Run on the disposable guest only. Does not install a kernel or reboot.
if [[ $(id -u) != 0 ]]; then
    echo 'Run this script through sudo in the VM terminal.' >&2
    exit 1
fi
if [[ $(hostname) != shiba2th ]] || [[ $(systemd-detect-virt) != kvm ]]; then
    echo 'Expected the shiba2th KVM guest; refusing a different machine.' >&2
    exit 1
fi
export DEBIAN_FRONTEND=noninteractive
# Prevent needrestart from restarting running services during dependency setup.
export NEEDRESTART_MODE=l
apt-get update
apt-get install -y --no-install-recommends \
    build-essential bc bison flex libssl-dev libelf-dev libncurses-dev \
    dwarves cpio rsync kmod initramfs-tools device-tree-compiler \
    libfdt-dev python3-libfdt python3-venv python3-dev python3-pip \
    python3-pytest python3-click python3-yaml python3-pyudev \
    swig pkg-config busybox-static zstd xz-utils git
echo 'Multikernel build/test dependencies installed. No reboot requested.'
