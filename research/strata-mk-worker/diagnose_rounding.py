"""Reproduce the isolated IQ2_XXS discrepancy with a CPU emulation of CUDA order."""
import json
import os
from pathlib import Path
import struct
import subprocess
import numpy as np
from validate import Client, read_expert

ROOT = Path(__file__).resolve().parent
out = ROOT / "quantization-diagnostic"
out.mkdir(exist_ok=True)
rng = np.random.default_rng(20261103)
for n in (1, 2, 4, 8, 1, 2):
    rng.standard_normal((n, 2560))
x = (rng.standard_normal((4, 2560)) * .02).astype("<f4")
gu, blob, matrices = read_expert(1, 0)
(out / "weights.bin").write_bytes(blob)
(out / "x.f32").write_bytes(x.tobytes())
for name, matrix in zip(("gate", "up"), matrices):
    (out / (name + ".f32")).write_bytes(matrix.astype("<f4").tobytes())
env = os.environ.copy()
env["MK_CPU_REF_CAPTURE_PREFIX"] = str(out / "reference")
subprocess.run([str(ROOT / "cpu_reference"), str(gu), "4", str(out / "weights.bin"), str(out / "x.f32"), str(out / "cpu-M4.f32")], env=env, check=True)
subprocess.run([str(ROOT / "emulate_gu"), str(out)], check=True)
c = Client(port=6006)
try:
    c.request(1, struct.pack("<7I", 1, 0, gu, 42, 2560, 640, len(blob)) + blob)
    payload = struct.pack("<4I", 1, 4, 4, 0) + b"".join(struct.pack("<ii", 0, t) for t in range(4)) + x.tobytes()
    raw = c.request(5, payload)
    (out / "gpu-compat-M4.f32").write_bytes(raw)
finally:
    c.sock.close()
got = np.frombuffer(raw, dtype="<f4").reshape(4, 2560)
cpu = np.fromfile(out / "cpu-M4.f32", dtype="<f4").reshape(4, 2560)
emulated = np.fromfile(out / "emulated.out", dtype="<f4").reshape(4, 2560)
ch = np.fromfile(out / "reference.hq", dtype=np.uint8).reshape(4, 1024)[:, :680]
eh = np.fromfile(out / "emulated.hq", dtype=np.uint8).reshape(4, 680)
report = {"layer": 1, "expert": 0, "M": 4,
          "GPU_equals_host_emulated_CUDA_GU_plus_CPU_down": bool(np.array_equal(got, emulated)),
          "GPU_vs_emulation_max_abs": float(np.max(np.abs(got - emulated))),
          "CPU_vs_emulated_hq_changed_byte_indices": [np.flatnonzero(a != b).tolist() for a, b in zip(ch, eh)],
          "GPU_vs_CPU_per_row_relative_L1": (np.abs(got - cpu).sum(axis=1) / np.abs(cpu).sum(axis=1)).tolist()}
(ROOT / "rounding-diagnostic.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
assert report["GPU_equals_host_emulated_CUDA_GU_plus_CPU_down"]
