"""Prepare an isolated secondary image; never mutate the original RPC filesystem."""
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parents[1]
dest = ROOT / "rootfs"
shutil.copytree(BASE / "rpc-p40-root", dest, symlinks=True, dirs_exist_ok=True)
worker = dest / "opt/mk-worker"
worker.mkdir(exist_ok=True)
shutil.copy2(ROOT / "worker", worker / "worker")
shutil.copy2(ROOT.parent / "strata-p40-20261102/vsock_file_pull", worker / "vsock_file_pull")
runtime = worker / "runtime"
runtime.mkdir(exist_ok=True)
for name in ("ld-linux-x86-64.so.2", "libc.so.6", "libm.so.6", "libstdc++.so.6", "libgcc_s.so.1"):
    shutil.copy2((Path("/usr/lib/x86_64-linux-gnu") / name).resolve(), runtime / name)
shutil.copy2("/usr/local/cuda-12.8/targets/x86_64-linux/lib/libcudart.so.12", worker / "libcudart.so.12")
print("Prepared isolated image:", dest)
