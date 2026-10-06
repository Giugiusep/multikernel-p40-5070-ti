#!/usr/bin/env python3
"""Summarize committed Qwen4exp MTP rounds from LLAMA_MTP_TRACE logs."""

import argparse
import re
import statistics
from collections import defaultdict
from pathlib import Path


COMMIT = re.compile(r"MTP_COMMIT slot=\d+ ordinal=\d+ token=(\d+)")
DRAFT = re.compile(r"MTP_DRAFT start_us=(\d+) end_us=(\d+)")
DECODE = re.compile(r"MTP_DECODE start_us=(\d+) end_us=(\d+) tokens=(\d+) output=(\d+)")
ROUND = re.compile(r"MTP_ROUND slot=\d+ proposed=(\d+) accepted=(\d+) rejection=(-?\d+) committed=(\d+)(?: restore=(\d+) replay=(\d+))? end_us=(\d+)")
RATE = re.compile(r"\[ Prompt: ([0-9.]+) t/s \| Generation: ([0-9.]+) t/s \]")


def parse(path: Path) -> dict:
    source = path.read_text(errors="replace")
    rate = RATE.search(source)
    if rate is None:
        raise ValueError(f"missing timing footer: {path}")
    groups = []
    group = {"passes": [], "commits": [], "start_us": None}
    initial = 0
    draft_start = None
    draft_ms = 0.0
    decode_ms = 0.0
    decode_start = None

    def finish() -> None:
        nonlocal group
        if group["passes"]:
            groups.append(group)
        group = {"passes": [], "commits": [], "start_us": None}

    for line in source.splitlines():
        if match := DRAFT.search(line):
            if group["commits"]:
                finish()
            draft_start = int(match[1])
            draft_ms = (int(match[2]) - draft_start) / 1000
            if group["start_us"] is None:
                group["start_us"] = draft_start
        if match := DECODE.search(line):
            start, end = int(match[1]), int(match[2])
            if group["start_us"] is not None:
                decode_ms += (end - start) / 1000
                if decode_start is None:
                    decode_start = start
        if match := ROUND.search(line):
            proposed, accepted, rejection, committed = map(int, match.group(1, 2, 3, 4))
            assert 0 <= accepted <= proposed and committed == accepted + 1
            assert rejection == (-1 if accepted == proposed else accepted)
            if group["start_us"] is None:
                group["start_us"] = decode_start
            group["passes"].append({
                "proposed": proposed,
                "accepted": accepted,
                "committed": committed,
                "restore": int(match[5]) if match[5] else 0,
                "replay": int(match[6]) if match[6] else 0,
                "draft_ms": draft_ms,
                "decode_ms": decode_ms,
                "end_us": int(match[7]),
            })
            draft_start = None
            draft_ms = 0.0
            decode_ms = 0.0
            decode_start = None
        if match := COMMIT.search(line):
            if group["passes"]:
                group["commits"].append(int(match[1]))
            else:
                initial += 1
    finish()

    accepted_by_pos = defaultdict(lambda: [0, 0])
    verify_by_width = defaultdict(list)
    draft_by_width = defaultdict(list)
    for index, item in enumerate(groups):
        first = item["passes"][0]
        last = item["passes"][-1]
        observed = len(item["commits"])
        if index < len(groups) - 1:
            assert observed == last["committed"], (path, index, observed, last)
        else:
            assert 1 <= observed <= last["committed"] + 1, (path, index, observed, last)
        assert item["start_us"] is not None
        item["round_ms"] = (last["end_us"] - item["start_us"]) / 1000
        item["useful_commits"] = min(observed, last["committed"])
        for pos in range(first["proposed"]):
            accepted_by_pos[pos][1] += 1
            accepted_by_pos[pos][0] += int(pos < first["accepted"])
        width = first["proposed"] + 1
        verify_by_width[width].append(first["decode_ms"])
        draft_by_width[width].append(first["draft_ms"])

    tokens = len(COMMIT.findall(source))
    assert tokens == initial + sum(len(item["commits"]) for item in groups)
    return {
        "tokens": tokens,
        "prompt_tps": float(rate[1]),
        "generation_tps": float(rate[2]),
        "initial": initial,
        "groups": groups,
        "accepted_by_pos": accepted_by_pos,
        "verify_by_width": verify_by_width,
        "draft_by_width": draft_by_width,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3, 4, 7])
    args = parser.parse_args()

    reference = parse(args.directory / f"qwen4exp-target-{args.tokens}.log")
    print("k,tokens,prompt_tps,generation_tps,logical_rounds,physical_verifications,proposed_per_round,accepted_per_round,committed_per_round,median_round_ms,median_first_draft_ms,median_first_verify_ms,acceptance_by_position")
    print(",".join(["0", str(reference["tokens"]), str(reference["prompt_tps"]), str(reference["generation_tps"])] + [""] * 9))
    reference_tokens = [int(value) for value in COMMIT.findall((args.directory / f"qwen4exp-target-{args.tokens}.log").read_text(errors="replace"))]
    for depth in args.depths:
        path = args.directory / f"qwen4exp-mtp-k{depth}-{args.tokens}.log"
        result = parse(path)
        actual = [int(value) for value in COMMIT.findall(path.read_text(errors="replace"))]
        assert actual == reference_tokens, f"token divergence at k={depth}"
        groups = result["groups"]
        n = len(groups)
        proposed = sum(item["passes"][0]["proposed"] for item in groups)
        useful = sum(item["useful_commits"] for item in groups)
        accepted = sum(item["passes"][0]["accepted"] for item in groups)
        assert accepted == sum(values[0] for values in result["accepted_by_pos"].values())
        physical = sum(len(item["passes"]) for item in groups)
        positions = ";".join(f"{result['accepted_by_pos'][pos][0]}/{result['accepted_by_pos'][pos][1]}" for pos in range(depth))
        first_drafts = [item["passes"][0]["draft_ms"] for item in groups]
        first_verifies = [item["passes"][0]["decode_ms"] for item in groups]
        print(f"{depth},{result['tokens']},{result['prompt_tps']},{result['generation_tps']},{n},{physical},{proposed/n:.4f},{accepted/n:.4f},{useful/n:.4f},{statistics.median(item['round_ms'] for item in groups):.4f},{statistics.median(first_drafts):.4f},{statistics.median(first_verifies):.4f},{positions}")
        print(f"# k={depth} width verify_ms: " + ", ".join(f"M={width} n={len(values)} median={statistics.median(values):.4f}" for width, values in sorted(result["verify_by_width"].items())))


if __name__ == "__main__":
    main()
