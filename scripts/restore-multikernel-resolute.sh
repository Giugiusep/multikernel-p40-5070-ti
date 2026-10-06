#!/bin/bash
set -euo pipefail
[[ $EUID -eq 0 ]] || { echo 'Run with sudo'; exit 1; }
base=/home/shiba/multikernel-experiment
release=7.0.0-mk2-shiba-mk1
source_dir="$base/nvidia-open-595.91.07"
[[ $(modinfo -F version "$source_dir/nvidia.ko") == 595.91.07 ]]
[[ $(modinfo -F vermagic "$source_dir/nvidia.ko") == "$release "* ]]
[[ $(readlink -f /usr/lib/x86_64-linux-gnu/libcuda.so.1) == *595.91.07 ]]
backup=/home/shiba/multikernel-upgrade-recovery-20261006
mkdir -p "$backup/pre-resolute-module-replacement"
cp -a /lib/modules/"$release"/updates/dkms "$backup/pre-resolute-module-replacement/"
cp -a /etc/default/grub.d/99-multikernel-experiment.cfg "$backup/stock-boot-before-reactivation.cfg"
for module in nvidia nvidia-modeset nvidia-uvm nvidia-drm nvidia-peermem; do
  install -m 0644 "$source_dir/$module.ko" /lib/modules/"$release"/updates/dkms/
done
depmod -a "$release"
update-initramfs -u -k "$release"
cat > /etc/default/grub.d/99-multikernel-experiment.cfg <<'GRUB'
GRUB_TIMEOUT_STYLE=menu
GRUB_TIMEOUT=15
GRUB_DEFAULT="gnulinux-advanced-a30d6b3f-539b-47bf-98e2-0f1a87f7ce24>gnulinux-7.0.0-mk2-shiba-mk1-advanced-a30d6b3f-539b-47bf-98e2-0f1a87f7ce24"
GRUB
update-grub
grub-script-check /boot/grub/grub.cfg
modinfo -k "$release" -F version nvidia
modinfo -k "$release" -F vermagic nvidia
echo 'Custom boot prepared; no reboot performed.'
