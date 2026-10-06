#!/bin/bash
set -euo pipefail
[[ $EUID -eq 0 ]] || { echo 'Run with sudo'; exit 1; }
backup=/home/shiba/multikernel-upgrade-recovery-20261006
mkdir -p "$backup"
if [[ ! -f "$backup/boot-and-custom-modules.tar.gz" ]]; then
  tar -czf "$backup/boot-and-custom-modules.tar.gz" /boot /lib/modules/7.0.0-mk2-shiba-mk1 /etc/default/grub /etc/default/grub.d /etc/apt/sources.list.d /etc/apt/preferences.d
  sha256sum "$backup/boot-and-custom-modules.tar.gz" > "$backup/boot-and-custom-modules.sha256"
fi
cp -n /etc/default/grub.d/99-multikernel-experiment.cfg "$backup/99-multikernel-experiment.cfg"
cat > /etc/default/grub.d/99-multikernel-experiment.cfg <<'GRUB'
GRUB_TIMEOUT_STYLE=menu
GRUB_TIMEOUT=15
GRUB_DEFAULT="gnulinux-advanced-a30d6b3f-539b-47bf-98e2-0f1a87f7ce24>gnulinux-6.8.0-142-generic-advanced-a30d6b3f-539b-47bf-98e2-0f1a87f7ce24"
GRUB
update-grub
grub-script-check /boot/grub/grub.cfg
printf '%s\n' 'Stock boot prepared. No reboot or package upgrade performed.'
