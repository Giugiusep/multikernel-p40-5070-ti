#!/usr/bin/env bash
set -euo pipefail
base=/home/shiba/multikernel-experiment
src=$base/research/linux-rxsafe-source
out=$base/research/linux-rxsafe-build
prep=$base/research/mk-vsock-rxsafe-20261108/boot-preparation
export LOCALVERSION=
export KBUILD_BUILD_VERSION=12-rxsafe1
export KBUILD_BUILD_USER=shiba
export KBUILD_BUILD_HOST=shiba2th
export KBUILD_BUILD_TIMESTAMP="$(date -u '+%a %b %e %T UTC %Y')"
mkdir -p "$out"
test ! -e "$out/.config"
cp /boot/config-7.0.0-mk2-shiba-mk1 "$out/.config"
printf '%s\n' "$KBUILD_BUILD_TIMESTAMP" > "$prep/build-timestamp.txt"
make -C "$src" O="$out" CC=/usr/bin/gcc-13 HOSTCC=/usr/bin/gcc-13 LD=/usr/bin/ld.bfd olddefconfig
release=$(make -s -C "$src" O="$out" CC=/usr/bin/gcc-13 HOSTCC=/usr/bin/gcc-13 LD=/usr/bin/ld.bfd kernelrelease)
test "$release" = 7.0.0-mk2-shiba-mk1
make -C "$src" O="$out" CC=/usr/bin/gcc-13 HOSTCC=/usr/bin/gcc-13 LD=/usr/bin/ld.bfd -j2 bzImage modules
"$src/scripts/diffconfig" /boot/config-7.0.0-mk2-shiba-mk1 "$out/.config" > "$prep/config-diff.txt"
