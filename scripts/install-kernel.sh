#!/usr/bin/env bash
set -euo pipefail
export LOCALVERSION=
base=/home/shiba/multikernel-experiment
src="$base/source"
build="$base/build"
release=$(cat "$base/artifacts/kernel-release.txt")
[[ $(id -u) == 0 && $(hostname) == shiba2th ]] || exit 1
[[ "$release" == 7.0.0-mk2-shiba-mk1 ]] || {
    echo "Unexpected release: $release" >&2; exit 1;
}
[[ $(cat "$build/include/config/kernel.release") == "$release" ]] || exit 1
test -s "$build/arch/x86/boot/bzImage"
test -s "$build/Module.symvers"
test -s "$base/lazy_cma/lazy_cma.ko"
[[ ! "$build/vmlinux" -nt "$build/arch/x86/boot/bzImage" ]] || {
    echo "Refusing to install stale bzImage; rebuild with: make O=$build bzImage modules" >&2
    exit 1
}
make -C "$src" O="$build" modules_install
install -m 0644 "$build/arch/x86/boot/bzImage" "/boot/vmlinuz-$release"
install -m 0644 "$build/System.map" "/boot/System.map-$release"
install -m 0644 "$build/.config" "/boot/config-$release"
install -d "/lib/modules/$release/extra"
install -m 0644 "$base/lazy_cma/lazy_cma.ko" "/lib/modules/$release/extra/lazy_cma.ko"
depmod -a "$release"
nvsrc="$base/nvidia-open-595.84"
install -d "/lib/modules/$release/updates/dkms"
for module in nvidia nvidia-modeset nvidia-drm nvidia-uvm nvidia-peermem; do
    test -s "$nvsrc/$module.ko"
    install -m 0644 "$nvsrc/$module.ko" "/lib/modules/$release/updates/dkms/$module.ko"
done
depmod -a "$release"
update-initramfs -c -k "$release"
echo "Installed $release and NVIDIA-open 595.84. Boot selection is not yet changed."
