#!/usr/bin/env bash
set -euo pipefail
export LOCALVERSION=
base=/home/shiba/multikernel-experiment
src="$base/source"
build="$base/build"
mkdir -p "$build"
cp /boot/config-6.8.0-139-generic "$build/.config"
make -C "$src" O="$build" olddefconfig
make -C "$src" O="$build" LSMOD="$base/logs/modules.before" localmodconfig </dev/null
"$src/scripts/config" --file "$build/.config" \
    --set-str LOCALVERSION '-shiba-mk1' --disable LOCALVERSION_AUTO \
    --enable MULTIKERNEL --enable KEXEC --enable KEXEC_FILE \
    --enable KEXEC_HANDOVER --enable MKTTY \
    --enable VSOCKETS --enable MULTIKERNEL_VSOCKETS \
    --enable OF --enable OF_FLATTREE --enable OF_EARLY_FLATTREE \
    --enable SMP --enable HOTPLUG_CPU --enable NUMA \
    --enable MEMORY_HOTPLUG --enable MEMORY_HOTREMOVE \
    --enable CONTIG_ALLOC --enable COMPACTION --enable MIGRATION \
    --enable CMA --enable DMA_CMA \
    --enable BLK_DEV_INITRD --enable RD_GZIP \
    --enable DEVTMPFS --enable DEVTMPFS_MOUNT \
    --enable PROC_FS --enable SYSFS --enable TMPFS \
    --enable SERIAL_8250 --enable SERIAL_8250_CONSOLE \
    --enable BLK_DEV_DM --enable EXT4_FS \
    --enable VIRTIO_PCI --enable VIRTIO_BLK --enable VIRTIO_NET \
    --enable SCSI --enable BLK_DEV_SD --enable SCSI_VIRTIO \
    --module DRM_NOUVEAU \
    --disable DEBUG_INFO --disable DEBUG_INFO_BTF \
    --disable DEBUG_INFO_DWARF_TOOLCHAIN_DEFAULT \
    --disable DEBUG_INFO_DWARF4 --disable DEBUG_INFO_DWARF5 --enable DEBUG_INFO_NONE \
    --disable MODULE_SIG --disable MODULE_SIG_ALL --disable MODULE_SIG_FORCE \
    --disable KEXEC_SIG --disable KEXEC_SIG_FORCE \
    --set-str SYSTEM_TRUSTED_KEYS '' --set-str SYSTEM_REVOCATION_KEYS ''
make -C "$src" O="$build" olddefconfig
make -C "$src" O="$build" prepare
make -s -C "$src" O="$build" kernelrelease > "$base/artifacts/kernel-release.txt"
cp "$build/.config" "$base/artifacts/primary-secondary.config"
python3 - "$build/.config" <<'PY'
from pathlib import Path
import sys
text = Path(sys.argv[1]).read_text().splitlines()
required = ['MULTIKERNEL', 'MKTTY', 'KEXEC_FILE', 'HOTPLUG_CPU',
            'MEMORY_HOTPLUG', 'MEMORY_HOTREMOVE', 'CONTIG_ALLOC',
            'BLK_DEV_INITRD', 'RD_GZIP', 'DEVTMPFS', 'EXT4_FS', 'BLK_DEV_DM']
missing = [name for name in required if f'CONFIG_{name}=y' not in text]
assert not missing, f'Missing kernel configuration: {missing}'
print('Required first-boot and secondary configuration checks passed.')
PY
