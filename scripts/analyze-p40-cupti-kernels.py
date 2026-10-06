#!/usr/bin/env python3
"""Summarize repeated P40 decode groups from the existing CUPTI activity CSV."""

import csv
import statistics
import sys
from collections import defaultdict


def family(name: str, output: bool) -> str:
    if name.startswith("_Z13mul_mat_vec_q"):
        return "Q8 output projection" if output else "Q8 matvec, other"
    if "quantize_q8_1" in name:
        return "Q8 activation quantization"
    if "flash_attn" in name:
        return "flash attention"
    if "rms_norm" in name:
        return "RMS normalization"
    if "topk_moe" in name or "get_rows_float" in name or "mul_mat_vec_f" in name:
        return "MoE routing / FP32 matvec"
    if "ssm_" in name:
        return "SSM scan / convolution"
    if "rope" in name.lower() or "rotary" in name.lower():
        return "RoPE"
    return "elementwise / other kernels"


rows = list(csv.reader(open(sys.argv[1], newline="")))
kernels = [(int(r[1]), int(r[2]), r[5]) for r in rows if r[0] == "kernel"]
copies = [(int(r[1]), int(r[2]), int(r[5]), int(r[6])) for r in rows if r[0] == "memcpy"]

groups = []
current = []
for kernel in kernels:
    if current and kernel[0] - current[-1][1] > 1_000_000:
        groups.append(current)
        current = []
    current.append(kernel)
if current:
    groups.append(current)

decode = groups[-14:]
if len(decode) != 14 or any(len(group) != 400 for group in decode):
    raise SystemExit("Expected 14 final repeated decode groups of 400 kernels")

totals = defaultdict(lambda: [0, 0.0])
spans = []
kernel_times = []
copy_times = []
gap_times = []
copy_bytes = []
for group in decode:
    start, end = group[0][0], group[-1][1]
    spans.append((end - start) / 1e6)
    longest = max(range(len(group)), key=lambda i: group[i][1] - group[i][0])
    if longest != len(group) - 1 or not group[longest][2].startswith("_Z13mul_mat_vec_q"):
        raise SystemExit("Longest kernel is not final Q8 output projection")
    for i, (a, b, name) in enumerate(group):
        key = family(name, i == longest)
        totals[key][0] += 1
        totals[key][1] += (b - a) / 1e6
    inside = [c for c in copies if c[0] >= start and c[1] <= end]
    if any(c[3] != 8 for c in inside):
        raise SystemExit("Expected device-to-device copies only")
    c_ms = sum((b - a) / 1e6 for a, b, _, _ in inside)
    c_bytes = sum(size for _, _, size, _ in inside)
    totals["device-to-device copies"][0] += len(inside)
    totals["device-to-device copies"][1] += c_ms
    k_ms = sum((b - a) / 1e6 for a, b, _ in group)
    kernel_times.append(k_ms)
    copy_times.append(c_ms)
    copy_bytes.append(c_bytes)
    gap_times.append(spans[-1] - k_ms - c_ms)

active = sum(v[1] for v in totals.values()) / len(decode)
print(f"decode groups={len(decode)}, kernels/token=400")
print("family | calls/token | ms/token | ms/call | % of active CUDA time")
for name, (count, ms) in sorted(totals.items(), key=lambda item: item[1][1], reverse=True):
    print(f"{name} | {count / 14:.1f} | {ms / 14:.4f} | {ms / count:.4f} | {100 * ms / 14 / active:.2f}")
print(f"RoPE | 0.0 | 0.0000 | n/a | 0.00 (no separate launch in trace)")
for name, values in [
    ("P40 CUDA span ms", spans),
    ("kernel execution ms", kernel_times),
    ("D2D copy ms", copy_times),
    ("unoccupied gap ms", gap_times),
    ("D2D bytes", copy_bytes),
]:
    print(f"{name}: mean={statistics.mean(values):.4f}, min={min(values):.4f}, max={max(values):.4f}")
