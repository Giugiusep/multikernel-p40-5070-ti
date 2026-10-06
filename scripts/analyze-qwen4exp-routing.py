#!/usr/bin/env python3
"""Pair Qwen4exp expert selections with an uninstrumented MTP trace."""

import argparse
import csv
import importlib.util
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path


PROFILE_PATH = Path(__file__).with_name("profile-qwen4exp-mtp.py")
spec = importlib.util.spec_from_file_location("qwen_profile", PROFILE_PATH)
assert spec and spec.loader
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)

TOPK = re.compile(r"ROUTE_TOPK t_us=(\d+) layer=(\d+) row=(\d+) experts=([0-9,]+)")


def tokens(path: Path) -> list[int]:
    return [int(value) for value in profile.COMMIT.findall(path.read_text(errors="replace"))]


def route_events(path: Path) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int, tuple[int, ...]]]]:
    intervals = []
    selections = []
    for line in path.read_text(errors="replace").splitlines():
        if match := profile.DECODE.search(line):
            intervals.append((int(match[1]), int(match[2]), int(match[3])))
        if match := TOPK.search(line):
            selections.append((int(match[1]), int(match[2]), int(match[3]),
                               tuple(int(value) for value in match[4].split(","))))
    return intervals, selections


def pearson(x: list[float], y: list[float]) -> float:
    assert len(x) == len(y) and len(x) >= 2
    mx, my = statistics.mean(x), statistics.mean(y)
    numerator = sum((a - mx) * (b - my) for a, b in zip(x, y))
    denominator = math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))
    return numerator / denominator if denominator else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path, help="unmodified MTP trace")
    parser.add_argument("routing", type=Path, help="same prompt/depth with ROUTE_TOPK enabled")
    parser.add_argument("--layers", type=int, default=48)
    parser.add_argument("--experts-per-row", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    assert tokens(args.baseline) == tokens(args.routing), "instrumented output differs from baseline"
    base = profile.parse(args.baseline)
    routed = profile.parse(args.routing)
    assert len(base["groups"]) == len(routed["groups"]), "different logical round counts"
    intervals, selections = route_events(args.routing)
    assert selections, "no ROUTE_TOPK events"

    rows = []
    for round_index, (control, trace) in enumerate(zip(base["groups"], routed["groups"])):
        control_first = control["passes"][0]
        trace_first = trace["passes"][0]
        assert control_first["proposed"] == trace_first["proposed"], (round_index, control_first, trace_first)
        start_us = trace["start_us"]
        end_us = trace_first["end_us"]
        candidates = [interval for interval in intervals if start_us <= interval[0] and interval[1] <= end_us]
        logical_width = sum(interval[2] for interval in candidates)
        assert logical_width == trace_first["proposed"] + 1, (round_index, logical_width, trace_first)
        by_layer = defaultdict(list)
        for timestamp, layer, local_row, experts in selections:
            if any(decode_start <= timestamp <= decode_end for decode_start, decode_end, _ in candidates):
                assert local_row == 0, "this analyzer requires -ub 1"
                assert 0 <= layer < args.layers
                assert len(experts) == args.experts_per_row
                by_layer[layer].append(experts)
        assert set(by_layer) == set(range(args.layers)), (round_index, sorted(by_layer))
        assert all(len(by_layer[layer]) == logical_width for layer in by_layer), (
            round_index, {layer: len(values) for layer, values in by_layer.items()})
        unique = sum(len({expert for ids in by_layer[layer] for expert in ids}) for layer in range(args.layers))
        total = args.layers * args.experts_per_row * logical_width
        adjacent_overlap = [len(set(left) & set(right)) / args.experts_per_row
                            for layer in range(args.layers)
                            for left, right in zip(by_layer[layer], by_layer[layer][1:])]
        rows.append({
            "round": round_index,
            "width": logical_width,
            "first_pass_accepted": control_first["accepted"],
            "committed": control["useful_commits"],
            "selected_expert_slots": total,
            "unique_layer_experts": unique,
            "reuse_factor": total / unique,
            "adjacent_row_overlap": statistics.mean(adjacent_overlap) if adjacent_overlap else float("nan"),
            "baseline_first_verify_ms": control_first["decode_ms"],
            "baseline_logical_round_ms": control["round_ms"],
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for width in sorted({row["width"] for row in rows}):
        subset = [row for row in rows if row["width"] == width]
        print(f"M={width} rounds={len(subset)} median_unique={statistics.median(row['unique_layer_experts'] for row in subset):.1f} median_reuse={statistics.median(row['reuse_factor'] for row in subset):.3f}")
        if len(subset) >= 3:
            correlation = pearson([row["unique_layer_experts"] for row in subset],
                                  [row["baseline_first_verify_ms"] for row in subset])
            print(f"M={width} diversity-vs-verify-time Pearson r={correlation:.3f} (exploratory)")
    print(args.output)


if __name__ == "__main__":
    main()
