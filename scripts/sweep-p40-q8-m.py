#!/usr/bin/env python3
"""Correctness-gated remote P40 Q8_0 matrix-width sweep.

These are isolated GGML matvec proxies, not full-model verification rounds.
"""

import csv
import pathlib
import re
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
PROBE = ROOT / "tests/p40_q8_matvec_probe"
SHAPES = [
    ("output", 2688, 131072),
    ("ssm_in", 2688, 10304),
    ("ssm_out", 4096, 2688),
    ("moe_up_expert", 2688, 1856),
    ("moe_down_expert", 1856, 2688),
    ("moe_up_shared", 2688, 3712),
    ("moe_down_shared", 3712, 2688),
    ("attention_q", 2688, 4096),
    ("attention_kv", 2688, 256),
]
MOE_SHAPES = [
    ("moe_up_selected6", 2688, 1856),
    ("moe_down_selected6", 1856, 2688),
]


def parse(output: str) -> dict[str, str]:
    fields = dict(re.findall(r"(\w+)=([^\s]+)", output))
    required = {"weights_bytes", "upload_s", "max_error", "correct", "median", "p95", "mean", "n"}
    if not required <= fields.keys() or fields["correct"] != "yes":
        raise ValueError(output)
    return fields


def main() -> int:
    samples = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    disjoint = len(sys.argv) > 2 and sys.argv[2] == "moe_disjoint"
    if len(sys.argv) > 2 and sys.argv[2] in ("moe", "moe_disjoint"):
        shapes = MOE_SHAPES
        probe = ROOT / "tests/p40_q8_moe_probe"
        path = ROOT / ("logs/p40-q8-moe-disjoint-m-sweep-20260927.csv" if disjoint
                       else "logs/p40-q8-moe-m-sweep-20260927.csv")
    elif len(sys.argv) > 2 and sys.argv[2] == "attention_kv":
        shapes = [SHAPES[-1]]
        probe = PROBE
        path = ROOT / "logs/p40-q8-attention-kv-m-sweep-20260927.csv"
    else:
        shapes = SHAPES[:-1]
        probe = PROBE
        path = ROOT / "logs/p40-q8-m-sweep-20260927.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "shape", "input_width", "output_rows", "M", "weights_bytes",
            "upload_s", "median_ms", "p95_ms", "mean_ms", "ms_per_M",
            "effective_weight_GiB_s", "max_error", "correct", "samples",
        ])
        writer.writeheader()
        for name, width, rows in shapes:
            for columns in range(1, 9):
                args = ([str(probe), str(columns), str(width), str(rows), str(samples),
                         "disjoint" if disjoint else "same"]
                        if probe != PROBE else
                        [str(probe), str(columns), "q8", str(samples), str(width), str(rows)])
                result = subprocess.run(
                    args,
                    text=True, capture_output=True, check=True,
                )
                fields = parse(result.stdout)
                median = float(fields["median"])
                row = {
                    "shape": name,
                    "input_width": width,
                    "output_rows": rows,
                    "M": columns,
                    "weights_bytes": int(fields["weights_bytes"]),
                    "upload_s": fields["upload_s"],
                    "median_ms": median,
                    "p95_ms": fields["p95"],
                    "mean_ms": fields["mean"],
                    "ms_per_M": round(median / columns, 6),
                    "effective_weight_GiB_s": round(int(fields["weights_bytes"]) / (1024 ** 3) / (median / 1000), 3),
                    "max_error": fields["max_error"],
                    "correct": fields["correct"],
                    "samples": fields["n"],
                }
                writer.writerow(row)
                stream.flush()
                print(f"{name:18} M={columns} median={median:.3f} ms correct=yes", flush=True)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
