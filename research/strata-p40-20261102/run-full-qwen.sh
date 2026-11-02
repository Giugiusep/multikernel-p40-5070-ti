#!/bin/sh
# Disabled after a VM soft lockup; explicit engineering review is required before a retry.
[ "${MODEL_BRIDGE_RETEST_AFTER_HANG:-0}" = 1 ] || { echo "Full-model bridge test disabled after 2026-11-02 hang"; exit 78; }
export LD_LIBRARY_PATH=/opt/strata:/lib
/bin/busybox timeout -k 30 1800 /opt/strata/strata --pack /mnt/model/pack --native /mnt/model/weights/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf --ple-gguf /mnt/model/weights/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf --expert-profile /mnt/model/profile.bin --expert-cache auto --mmap-experts --pool-workers 3 --max-context 2048 --prefill 1 --spec 2 --adapt-every 0 --kv int8 --vram-reserve-mib 1024 --greedy --serve
status=$?
echo FULL_QWEN_EXIT=$status
exit "$status"
