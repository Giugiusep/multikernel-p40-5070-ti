"""Prepare an isolated secondary image; never mutate the original RPC filesystem."""
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parents[1]
dest = ROOT / "rootfs"
if not dest.exists():
    shutil.copytree(BASE / "rpc-p40-root", dest, symlinks=True)
elif not (dest / "sbin/entrypoint").is_file() or not (dest / "lib/modules/7.0.0-mk2-shiba-mk1/nvidia.ko").is_file():
    raise SystemExit("Existing staged image is incomplete; inspect it before replacing any files")
worker = dest / "opt/mk-worker"
worker.mkdir(exist_ok=True)
shutil.copy2(ROOT / "worker", worker / "worker")
if (ROOT / "worker-qualified").exists():
    shutil.copy2(ROOT / "worker-qualified", worker / "worker-qualified")
    shutil.copy2(ROOT / "run_worker_qualified.sh", worker / "run_worker_qualified.sh")
shutil.copy2(ROOT.parent / "strata-p40-20261102/vsock_file_pull", worker / "vsock_file_pull")
runtime = worker / "runtime"
runtime.mkdir(exist_ok=True)
for name in ("ld-linux-x86-64.so.2", "libc.so.6", "libm.so.6", "libstdc++.so.6", "libgcc_s.so.1"):
    shutil.copy2((Path("/usr/lib/x86_64-linux-gnu") / name).resolve(), runtime / name)
shutil.copy2("/usr/local/cuda-12.8/targets/x86_64-linux/lib/libcudart.so.12", worker / "libcudart.so.12")
print("Prepared isolated image:", dest)
