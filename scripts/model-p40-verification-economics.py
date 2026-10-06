#!/usr/bin/env python3
"""Conditional full-round economics from measured Q8 width ratios.

The 48/52 CUPTI Q8 mix calibrates the ratio, which is applied to the 50/50
72 tok/s baseline. All other 50/50 work is held at its measured M=1 cost.
This is an optimistic scenario, not a full-model or Qwen3.8 MTP prediction.
"""

import csv
import math
import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"
RATES = (72, 88, 100, 120, 150)
# 50/50 primary trace: RTX 2.430, dependent receive 9.245,
# inter-graph gaps 0.972 ms. The CLI 72 tok/s mean adds 1.242 ms/token
# outside those 12.647 ms repeated intervals. The Q8 calibration comes
# from the older 48/52 secondary CUPTI trace, so split mix is approximate.
RTX_MS = 2.430
DEPENDENT_RECEIVE_MS = 9.245
GAPS_MS = 0.972
BASE_TOKEN_MS = 1000 / 72
CLI_OUTSIDE_GRAPH_MS = BASE_TOKEN_MS - (RTX_MS + DEPENDENT_RECEIVE_MS + GAPS_MS)
Q8_BASE_MS = 1.397 + 6.1321
DEPENDENT_NON_Q8_MS = DEPENDENT_RECEIVE_MS - Q8_BASE_MS
SHAPES = {
    "output": (2688, 131072, 1),
    "ssm_in": (2688, 10304, 11),
    "ssm_out": (4096, 2688, 11),
    "moe_up_selected6": (2688, 1856, 12 * 6),
    "moe_down_selected6": (1856, 2688, 12 * 6),
    "moe_up_shared": (2688, 3712, 12),
    "moe_down_shared": (3712, 2688, 12),
    "attention_q": (2688, 4096, 3),
    "attention_kv": (2688, 256, 6),
    "attention_out": (4096, 2688, 3),
}


def load(path: Path) -> dict[tuple[str, int], float]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert all(row["correct"] == "yes" and int(row["samples"]) >= 50 for row in rows)
    return {(row["shape"], int(row["M"])): float(row["median_ms"]) for row in rows}


def required_committed(rate: int, ms: float) -> int:
    """Smallest integer c with 1000*c/ms > rate."""
    return math.floor(rate * ms / 1000) + 1


def average_accepted_threshold(rate: int, ms: float) -> float:
    """Average accepted drafts/round must be strictly above this value."""
    return rate * ms / 1000 - 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--routing", choices=("same", "disjoint"), default="same")
    args = parser.parse_args()
    # Deterministic boundary checks for the strict greater-than convention.
    assert required_committed(100, 10.0) == 2
    assert required_committed(99, 10.0) == 1
    assert required_committed(72, 1000 / 72) == 2
    assert average_accepted_threshold(100, 10.0) == 0.0
    assert abs(average_accepted_threshold(120, 1000 / 72) - 2 / 3) < 1e-12
    assert abs(CLI_OUTSIDE_GRAPH_MS - 1.242) < 0.01
    assert abs(DEPENDENT_NON_Q8_MS - 1.716) < 0.01

    times = load(LOGS / "p40-q8-m-sweep-20260927.csv")
    times.update(load(LOGS / "p40-q8-attention-kv-m-sweep-20260927.csv"))
    moe_file = ("p40-q8-moe-disjoint-m-sweep-20260927.csv" if args.routing == "disjoint"
                else "p40-q8-moe-m-sweep-20260927.csv")
    times.update(load(LOGS / moe_file))
    for m in range(1, 9):
        # Identical Q8 matrix dimensions; only the graph role differs.
        times["attention_out", m] = times["ssm_out", m]
    for name in SHAPES:
        for m in range(1, 9):
            assert (name, m) in times, (name, m)
    weight_bytes = {name: width * rows * 34 // 32 * count
                    for name, (width, rows, count) in SHAPES.items()}
    assert sum(weight_bytes.values()) == 1_919_049_216
    other_bytes = sum(value for name, value in weight_bytes.items() if name != "output")

    suffix = "-disjoint" if args.routing == "disjoint" else ""
    out = LOGS / f"p40-q8-verification-economics{suffix}-20260927.csv"
    with out.open("w", newline="") as stream:
        names = ["M", "q8_proxy_ms", "round_scenario_ms", "max_tok_s"]
        names += [f"accepted_avg_gt_{rate}" for rate in RATES]
        names += [f"integer_committed_single_round_to_beat_{rate}" for rate in RATES]
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        for m in range(1, 9):
            output_ratio = times["output", m] / times["output", 1]
            other_ratio = sum(
                value * times[name, m] / times[name, 1]
                for name, value in weight_bytes.items() if name != "output"
            ) / other_bytes
            q8_ms = 1.397 * output_ratio + 6.1321 * other_ratio
            # Constant non-Q8, RTX and host/RPC terms are an optimistic
            # scenario. Their actual M>1 cost has not been measured.
            round_ms = (RTX_MS + q8_ms + DEPENDENT_NON_Q8_MS +
                        GAPS_MS + CLI_OUTSIDE_GRAPH_MS)
            assert round_ms > 0
            row = {
                "M": m,
                "q8_proxy_ms": f"{q8_ms:.4f}",
                "round_scenario_ms": f"{round_ms:.4f}",
                "max_tok_s": f"{1000 * m / round_ms:.2f}",
            }
            row.update({f"accepted_avg_gt_{rate}": f"{average_accepted_threshold(rate, round_ms):.4f}"
                        for rate in RATES})
            row.update({f"integer_committed_single_round_to_beat_{rate}": required_committed(rate, round_ms)
                        for rate in RATES})
            writer.writerow(row)
            print(row)
    print(out)


if __name__ == "__main__":
    main()
