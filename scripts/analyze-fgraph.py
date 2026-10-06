#!/usr/bin/env python3
"""Summarize inclusive function times from a bounded tracefs function graph."""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path


DURATION = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s+us")


def parse(path: Path) -> dict[str, list[float]]:
    samples: dict[str, list[float]] = defaultdict(list)
    stacks: dict[str, list[str]] = defaultdict(list)
    for line in path.read_text().splitlines():
        if "|" not in line or ")" not in line.split("|", 1)[0]:
            continue
        left, right = line.split("|", 1)
        cpu = left.split(")", 1)[0].strip()
        call = right.strip()
        duration = DURATION.search(left)
        if call.endswith("() {"):
            stacks[cpu].append(call[:-4])
        elif call == "}" or call.startswith("} /*"):
            if not stacks[cpu] or duration is None:
                raise ValueError(f"unmatched return: {line}")
            samples[stacks[cpu].pop()].append(float(duration.group(1)))
        elif call.endswith("();") and duration:
            samples[call[:-3]].append(float(duration.group(1)))
    if any(stacks.values()):
        raise ValueError(f"truncated function graph: {stacks}")
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    args = parser.parse_args()
    samples = parse(args.trace)
    print("function\tcalls\ttotal_us\tmean_us\tmax_us")
    for name in (
        "mk_transport_stream_enqueue",
        "mk_send_pkt",
        "mk_send_message",
        "multikernel_send_ipi_data",
        "mk_arch_send_ipi",
        "mk_vsock_rx_pkt",
        "mk_transport_stream_dequeue",
        "skb_copy_datagram_iter",
        "simple_copy_to_iter",
        "sock_def_write_space",
        "sock_def_readable",
        "try_to_wake_up",
        "x2apic_send_IPI",
        "mk_send_credit_pkt.isra.0",
        "__alloc_skb",
        "sk_skb_reason_drop",
    ):
        values = samples[name]
        if values:
            print(
                f"{name}\t{len(values)}\t{sum(values):.3f}\t"
                f"{sum(values) / len(values):.3f}\t{max(values):.3f}"
            )


if __name__ == "__main__":
    main()
