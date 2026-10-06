#!/usr/bin/env python3
"""Summarize opt-in ggml RPC server phase traces."""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


TRACE_RE = re.compile(
    rb"rpc-server-trace cmd=(\S+) bytes=(\d+) recv_us=(\d+) "
    rb"handler_us=(\d+) send_us=(\d+)"
)


@dataclass(frozen=True)
class Record:
    command: str
    bytes: int
    recv_us: int
    handler_us: int
    send_us: int


def parse(path: Path) -> list[Record]:
    return [
        Record(
            match.group(1).decode(),
            int(match.group(2)),
            int(match.group(3)),
            int(match.group(4)),
            int(match.group(5)),
        )
        for match in TRACE_RE.finditer(path.read_bytes())
    ]


def report(label: str, records: list[Record]) -> None:
    grouped: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        grouped[record.command].append(record)

    print(
        f"[{label}] entries={len(records)} bytes={sum(r.bytes for r in records)} "
        f"recv_us={sum(r.recv_us for r in records)} "
        f"handler_us={sum(r.handler_us for r in records)} "
        f"send_us={sum(r.send_us for r in records)}"
    )
    print("command\tcount\tbytes\trecv_us\thandler_us\tsend_us")
    for command in sorted(grouped):
        values = grouped[command]
        print(
            f"{command}\t{len(values)}\t{sum(r.bytes for r in values)}\t"
            f"{sum(r.recv_us for r in values)}\t"
            f"{sum(r.handler_us for r in values)}\t"
            f"{sum(r.send_us for r in values)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    args = parser.parse_args()

    records = parse(args.trace)
    if not records:
        raise SystemExit("no rpc-server-trace records found")

    first_graph = next(
        index for index, record in enumerate(records)
        if record.command in {"graph_compute", "graph_recompute"}
    )
    last_graph_compute = max(
        index for index, record in enumerate(records)
        if record.command == "graph_compute"
    )

    report("all", records)
    report("model-load-and-setup", records[:first_graph])
    report("inference", records[first_graph:])
    steady = records[last_graph_compute + 1:]
    report("steady-after-final-graph-build", steady)

    outputs = sum(
        record.command == "get_tensor" and record.bytes > 0
        for record in steady
    )
    if outputs:
        print(
            f"[steady-per-output] outputs={outputs} "
            f"requests={len(steady) / outputs:.3f} "
            f"bytes={sum(r.bytes for r in steady) / outputs:.3f} "
            f"recv_us={sum(r.recv_us for r in steady) / outputs:.3f} "
            f"handler_us={sum(r.handler_us for r in steady) / outputs:.3f} "
            f"send_us={sum(r.send_us for r in steady) / outputs:.3f}"
        )

    first_steady_output = next(
        index for index in range(last_graph_compute + 1, len(records))
        if records[index].command == "get_tensor" and records[index].bytes > 0
    )
    repeated = records[first_steady_output + 1:]
    repeated_outputs = sum(
        record.command == "get_tensor" and record.bytes > 0
        for record in repeated
    )
    report("repeated-token-loop", repeated)
    if repeated_outputs:
        print(
            f"[repeated-per-output] outputs={repeated_outputs} "
            f"requests={len(repeated) / repeated_outputs:.3f} "
            f"bytes={sum(r.bytes for r in repeated) / repeated_outputs:.3f} "
            f"recv_us={sum(r.recv_us for r in repeated) / repeated_outputs:.3f} "
            f"handler_us={sum(r.handler_us for r in repeated) / repeated_outputs:.3f} "
            f"send_us={sum(r.send_us for r in repeated) / repeated_outputs:.3f}"
        )


if __name__ == "__main__":
    main()
