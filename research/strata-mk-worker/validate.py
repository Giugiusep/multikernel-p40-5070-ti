"""Bounded real-checkpoint Qwen expert validation across Multikernel VSOCK.

The primary reads at most six expert blobs; no remote model filesystem.
CPU float dequantization is independent of the CUDA grouped expert kernel.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import statistics
import struct
import sys
import time
import subprocess
import tempfile

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np

BASE = Path("/home/shiba/multikernel-experiment")
sys.path.insert(0, "/home/shiba/llama.cpp/gguf-py")
sys.path.insert(0, str(BASE / "research/Strata-dsv41-experiment/tools"))
from gguf import GGMLQuantizationType
from gguf.quants import dequantize
from gguf_writer import dequantize_q2_0

MAGIC, VERSION = 0x5058454D, 1
HEADER = struct.Struct("<6I")
H, FF = 2560, 640
ROOT = Path(__file__).resolve().parent
PACK = BASE / "research/Strata/packs/iq2_xs/native_experts.txt"
SHARD = Path("/home/shiba/models/qwen3.8-flash-next-gsq-rco/IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf")


class Client:
    def __init__(self, cid=1, port=6005):
        self.sock = socket.socket(socket.AF_VSOCK, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.AF_VSOCK, 9, 1)
        self.sock.settimeout(10)
        self.sock.connect((cid, port))
        self.sequence = 0
        self.sent = self.received = 0
        self.short_writes = self.enospc = 0

    def send(self, data):
        view = memoryview(data)
        while view:
            part = view[:32768]
            n = self.sock.send(part)
            if n <= 0:
                raise RuntimeError("closed while sending")
            self.short_writes += int(n < len(part))
            self.sent += n
            view = view[n:]
            if view:
                time.sleep(0.0001)

    def recv(self, size):
        buf = bytearray(size)
        view = memoryview(buf)
        while view:
            n = self.sock.recv_into(view)
            if not n:
                raise RuntimeError("closed while receiving")
            self.received += n
            view = view[n:]
        return bytes(buf)

    def request(self, op, payload=b"", expect_error=None):
        self.sequence += 1
        self.send(HEADER.pack(MAGIC, VERSION, op, 0, self.sequence, len(payload)))
        self.send(payload)
        magic, ver, got_op, status, seq, length = HEADER.unpack(self.recv(24))
        assert (magic, ver, got_op, seq) == (MAGIC, VERSION, op, self.sequence)
        assert length <= 2 * 1024 * 1024
        result = self.recv(length)
        if expect_error:
            assert status == 1 and expect_error.encode() in result, result
        elif status:
            raise RuntimeError(result.decode())
        return result


def read_expert(layer, expert):
    fields = next(line.split() for line in PACK.read_text().splitlines()
                  if line and not line.startswith("#") and int(line.split()[0]) == layer)
    _, gu, down, _, blob_bytes, gate_off, up_off, down_off = map(int, fields[:8])
    block = {16: 66, 22: 82, 29: 56}[gu]
    gu_bytes = FF * (H // 256) * block
    down_bytes = H * (FF // 64) * 18
    assert down == 42 and blob_bytes == 2 * gu_bytes + down_bytes
    path = SHARD if len(fields) == 8 else SHARD.parent / fields[8]
    chunks = []
    with path.open("rb", buffering=0) as f:
        for off, size in ((gate_off, gu_bytes), (up_off, gu_bytes), (down_off, down_bytes)):
            f.seek(off + expert * size)
            raw = f.read(size)
            assert len(raw) == size
            chunks.append(raw)
    blob = b"".join(chunks)
    matrices = [dequantize(np.frombuffer(c, dtype=np.uint8), GGMLQuantizationType(gu)).reshape(FF, H)
                for c in chunks[:2]]
    matrices.append(dequantize_q2_0(chunks[2]).reshape(H, FF))
    assert all(np.isfinite(m).all() for m in matrices)
    return gu, blob, matrices


def reference(x, matrices):
    gate, up, down = matrices
    a = x @ gate.T
    b = x @ up.T
    sigmoid = 1 / (1 + np.exp(-np.clip(a, -80, 80)))
    return ((a * sigmoid) * b) @ down.T


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cid", type=int, default=1)
    parser.add_argument("--port", type=int, default=6005)
    parser.add_argument("--output", type=Path, default=ROOT / "real-expert-results.json")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--cpu-reference", type=Path, help="Compare to deployed ggml CPU quantized arithmetic instead of float oracle")
    parser.add_argument("--diagnostic", action="store_true", help="Collect all CPU-comparison cases; still fail overall if the strict tolerance is exceeded")
    parser.add_argument("--layers", default="0,1,8")
    parser.add_argument("--require-exact", action="store_true")
    args = parser.parse_args()
    layers = tuple(int(x) for x in args.layers.split(","))
    weights = {(l, e): read_expert(l, e) for l in layers for e in (0, 511)}
    manifest = [{"layer": l, "expert": e, "gu_type": v[0], "down_type": 42,
                 "bytes": len(v[1]), "sha256": hashlib.sha256(v[1]).hexdigest()}
                for (l, e), v in weights.items()]
    result = {"checkpoint": str(SHARD), "weights": manifest,
              "scope": "real Qwen expert MLP only; not full model or DeepSeek", "runs": [],
              "require_exact": args.require_exact,
              "cpu_reference_executable": str(args.cpu_reference) if args.cpu_reference else None,
              "cpu_reference_contract": ("actual Strata ActQ pool" if os.environ.get("MK_CPU_POOL_ORACLE") or "cpu_reference_pool" in str(args.cpu_reference) else "generic ggml native") if args.cpu_reference else "independent float dequantization"}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if args.prepare_only:
        print(json.dumps({"experts": len(weights), "bytes": sum(len(v[1]) for v in weights.values()), "cpu_dequantization": "finite"}))
        return
    client = Client(args.cid, args.port)
    try:
        result["before"] = json.loads(client.request(3))
        assert result["before"]["device"] == "Tesla P40" and result["before"]["sm"] == 61
        payloads = []
        for (layer, expert), (gu, blob, _) in weights.items():
            payload = struct.pack("<7I", layer, expert, gu, 42, H, FF, len(blob)) + blob
            client.request(1, payload)
            payloads.append(payload)
        client.request(1, payloads[0], expect_error="duplicate")
        client.request(2, struct.pack("<4I", 0, 9, 10, 0), expect_error="dimensions")
        client.request(2, struct.pack("<4I", 0, 1, 1, 0) + struct.pack("<i", 123) + bytes(H * 4), expect_error="not resident")
        client.request(99, expect_error="unknown")
        rng = np.random.default_rng(20261103)
        for layer in layers:
            for rows in (1, 2, 4, 8):
                k = 10
                x = (rng.standard_normal((rows, H)) * 0.02).astype("<f4")
                ids = np.array([(0, 511)[i % 2] for i in range(rows * k)], dtype="<i4")
                payload = struct.pack("<4I", layer, rows, k, 0) + ids.tobytes() + x.tobytes()
                expected = np.stack([reference(x[i // k], weights[(layer, int(e))][2]) for i, e in enumerate(ids)])
                if args.cpu_reference:
                    with tempfile.TemporaryDirectory(prefix="cpu-reference-", dir=ROOT) as temp:
                        d = Path(temp)
                        (d / "x").write_bytes(x.tobytes())
                        cpu_outputs = {}
                        for expert in (0, 511):
                            gu, blob, _ = weights[(layer, expert)]
                            (d / "weights").write_bytes(blob)
                            subprocess.run([str(args.cpu_reference), str(gu), str(rows), str(d / "weights"), str(d / "x"), str(d / "out")], check=True, timeout=30)
                            cpu_outputs[expert] = np.fromfile(d / "out", dtype="<f4").reshape(rows, H)
                        expected = np.stack([cpu_outputs[int(e)][i // k] for i, e in enumerate(ids)])
                raw = client.request(2, payload)
                got = np.frombuffer(raw, dtype="<f4").reshape(rows * k, H)
                assert np.isfinite(got).all()
                rel = float(np.abs(got - expected).sum() / (np.abs(expected).sum() + 1e-30))
                cosine = float(np.sum(got.astype("f8") * expected) / (np.linalg.norm(got.astype("f8")) * np.linalg.norm(expected.astype("f8"))))
                tolerance = 0.0001 if args.cpu_reference else 0.03
                per_entry_rel = np.abs(got - expected).sum(axis=1) / (np.abs(expected).sum(axis=1) + 1e-30)
                strict_ok = rel < tolerance and cosine > 0.999 and np.max(per_entry_rel) < tolerance
                if args.require_exact:
                    strict_ok = strict_ok and raw == expected.astype("<f4").tobytes()
                assert strict_ok or args.diagnostic, (layer, rows, "CPU-reference tolerance", rel, per_entry_rel.tolist())
                assert rel < 0.03 and cosine > 0.999, (layer, rows, "large diagnostic discrepancy", rel, cosine)
                positions = list(range(rows * k - 1, -1, -3))
                sparse = struct.pack("<4I", layer, rows, len(positions), 0)
                sparse += b"".join(struct.pack("<ii", int(ids[i]), i // k) for i in positions)
                sparse += x.tobytes()
                sparse_got = np.frombuffer(client.request(5, sparse), dtype="<f4").reshape(len(positions), H)
                assert np.array_equal(sparse_got, got[positions]), "sparse routing/reordering mismatch"
                timings = []
                for _ in range(10):
                    start = time.monotonic_ns()
                    again = client.request(2, payload)
                    timings.append((time.monotonic_ns() - start) / 1e6)
                    assert again == raw, "repeated output changed"
                record = {"layer": layer, "gu_type": weights[(layer, 0)][0], "M": rows,
                          "entries": rows * k, "relative_L1": rel, "cosine": cosine,
                          "max_entry_relative_L1": float(np.max(per_entry_rel)),
                          "median_roundtrip_ms": statistics.median(timings),
                          "max_roundtrip_ms": max(timings), "exact_repeated_outputs": 10}
                record["sparse_routing_exact"] = True
                record["cpu_quantized_reference"] = bool(args.cpu_reference)
                record["within_strict_tolerance"] = bool(strict_ok)
                record["reference_bytes_equal"] = raw == expected.astype("<f4").tobytes()
                result["runs"].append(record)
                args.output.write_text(json.dumps(result, indent=2) + "\n")
                print(json.dumps(record), flush=True)
        result["loaded"] = json.loads(client.request(3))
        client.request(4)
        result["cleared"] = json.loads(client.request(3))
        assert result["cleared"]["resident_bytes"] == 0 and result["cleared"]["experts"] == 0
        result["transport"] = {"requests": client.sequence, "sent": client.sent,
                               "received": client.received, "short_writes": client.short_writes}
        result["passed"] = all(r["within_strict_tolerance"] for r in result["runs"])
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        assert result["passed"], "strict CPU-reference tolerance exceeded; diagnostic results retained"
    finally:
        client.sock.close()


if __name__ == "__main__":
    main()
