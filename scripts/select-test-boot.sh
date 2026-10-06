#!/usr/bin/env bash
set -euo pipefail
base=/home/shiba/multikernel-experiment
release=$(cat "$base/artifacts/kernel-release.txt")
[[ $(id -u) == 0 && $(hostname) == shiba2th ]] || exit 1
test -s "/boot/vmlinuz-$release"
test -s "/boot/initrd.img-$release"
test -s /boot/vmlinuz-6.8.0-139-generic
python3 - "$base" <<'PY'
import pathlib, re, sys
base = pathlib.Path(sys.argv[1])
old = (base / 'logs/grub.cfg.before').read_text()
submenu = re.search(r"submenu .*'([^']*gnulinux-advanced-[^']*)'", old).group(1)
entry = re.search(r"menuentry .*'(gnulinux-6\.8\.0-139-generic-advanced-[^']*)'", old).group(1)
config = pathlib.Path('/etc/default/grub.d/99-multikernel-experiment.cfg')
config.parent.mkdir(exist_ok=True)
config.write_text(f'GRUB_TIMEOUT_STYLE=menu\nGRUB_TIMEOUT=10\nGRUB_DEFAULT="{submenu}>{entry}"\n')
PY
update-grub
grub-script-check /boot/grub/grub.cfg
python3 - "$base" "$release" <<'PY'
import pathlib, re, subprocess, sys
base = pathlib.Path(sys.argv[1])
release = sys.argv[2]
text = pathlib.Path('/boot/grub/grub.cfg').read_text()
submenu = re.search(r"submenu .*'([^']*gnulinux-advanced-[^']*)'", text).group(1)
entry = re.search(r"menuentry .*'(gnulinux-" + re.escape(release) + r"-advanced-[^']*)'", text).group(1)
selection = submenu + '>' + entry
subprocess.run(['grub-reboot', selection], check=True)
(base / 'artifacts/next-boot-entry.txt').write_text(selection + '\n')
print('One-time next boot:', selection)
PY
grub-editenv /boot/grub/grubenv list
echo 'STOP: no reboot has been executed. Persistent default remains Ubuntu 6.8.0-139-generic.'
