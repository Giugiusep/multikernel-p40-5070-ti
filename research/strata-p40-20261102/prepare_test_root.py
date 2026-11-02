from pathlib import Path
import shutil
root=Path(__file__).resolve().parent
base=Path('/home/shiba/multikernel-experiment')
if not (root/'rootfs').exists():
    shutil.copytree(base/'rpc-p40-root',root/'rootfs',symlinks=True)
dest=root/'rootfs/opt/strata';dest.mkdir(exist_ok=True)
for name in ['strata','libstrata_kq.so','libcublas.so.12','libcublasLt.so.12','libcudart.so.12','engine-component.json','BUILD.json']:
    shutil.copy2(root/'engine'/name,dest/name)
shutil.copy2(root/'mmvq_multi_parity',dest/'mmvq_multi_parity')
f=root/'rootfs/sbin/entrypoint';text=(base/'rpc-p40-root/sbin/entrypoint').read_text();needle='nvidia-smi || exec sh\n';assert needle in text
text=text.replace(needle,needle+"""echo STRATA_SM61_PROBE_START
LD_LIBRARY_PATH=/opt/strata:/lib /opt/strata/strata --version
if /bin/busybox timeout 180 /opt/strata/mmvq_multi_parity; then
    echo STRATA_SM61_PROBE_PASS
else
    echo STRATA_SM61_PROBE_FAILED
fi
""");f.write_text(text)
print('Separate test rootfs staged; original untouched.')
