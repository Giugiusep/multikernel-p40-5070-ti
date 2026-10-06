#!/bin/bash
set -euo pipefail
[[ $EUID -eq 0 ]] || exit 1
# The release upgrade added a 1 GiB crash-kernel reservation on this VM.
# Keep the pre-upgrade memory layout for Lazy CMA / secondary allocation.
if [[ -f /etc/default/grub.d/kdump-tools.cfg ]]; then
  cp -a /etc/default/grub.d/kdump-tools.cfg /home/shiba/multikernel-upgrade-recovery-20261006/kdump-tools.cfg
  mv /etc/default/grub.d/kdump-tools.cfg /etc/default/grub.d/kdump-tools.cfg.disabled
fi
update-grub
grub-script-check /boot/grub/grub.cfg
