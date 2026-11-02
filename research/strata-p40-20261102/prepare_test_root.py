from pathlib import Path
import shutil
root=Path(__file__).resolve().parent
base=Path('/home/shiba/multikernel-experiment')
if not (root/'rootfs').exists():
    shutil.copytree(base/'rpc-p40-root',root/'rootfs',symlinks=True)
dest=root/'rootfs/opt/strata';dest.mkdir(exist_ok=True)
for name in ['strata','libstrata_kq.so','libcublas.so.12','libcublasLt.so.12','libcudart.so.12','engine-component.json','BUILD.json']:
    shutil.copy2(root/'engine'/name,dest/name)
for name in ['mmvq_multi_parity', 'iq_parity', 'vsock_file_pull']:
    shutil.copy2(root/name,dest/name)
shutil.copytree(root/'iq-fixture',dest/'iq-fixture',dirs_exist_ok=True)
runtime=dest/'probe-runtime';runtime.mkdir(exist_ok=True)
for name in ['ld-linux-x86-64.so.2','libc.so.6','libm.so.6','libstdc++.so.6','libgcc_s.so.1']:
    shutil.copy2((Path('/usr/lib/x86_64-linux-gnu')/name).resolve(),runtime/name)
f=root/'rootfs/sbin/entrypoint';text=(base/'rpc-p40-root/sbin/entrypoint').read_text();needle='nvidia-smi || exec sh\n';assert needle in text
text=text.replace(needle,needle+"""echo STRATA_SM61_PROBE_START
LD_LIBRARY_PATH=/opt/strata:/lib /opt/strata/strata --version
if /bin/busybox timeout 180 /opt/strata/probe-runtime/ld-linux-x86-64.so.2 --library-path /opt/strata/probe-runtime:/opt/rpc/lib:/lib /opt/strata/mmvq_multi_parity > /tmp/mmvq-parity.log 2>&1; then
    echo STRATA_SM61_PROBE_PASS
    if /opt/strata/probe-runtime/ld-linux-x86-64.so.2 --library-path /opt/strata/probe-runtime:/opt/rpc/lib:/lib /opt/strata/iq_parity /opt/strata/iq-fixture > /tmp/iq-parity.log 2>&1; then echo STRATA_SM61_IQ_PASS; else echo STRATA_SM61_IQ_FAILED; fi
else
    echo STRATA_SM61_PROBE_FAILED
fi
""");f.write_text(text)
print('Separate test rootfs staged; original untouched.')
