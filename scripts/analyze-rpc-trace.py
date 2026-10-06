#!/usr/bin/env python3
"""Summarize opt-in ggml RPC dispatcher traces."""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


TRACE_RE = re.compile(
    rb"rpc-trace cmd=(\S+) input=(\d+) output=(\d+) "
    rb"elapsed_us=(\d+) status=(\S+)"
)


@dataclass(frozen=True)
class Record:
    command: str
    input_bytes: int
    output_bytes: int
    elapsed_us: int
    status: str


def parse(path: Path) -> list[Record]:
    data = path.read_bytes()
    return [
        Record(
            match.group(1).decode(),
            int(match.group(2)),
            int(match.group(3)),
            int(match.group(4)),
            match.group(5).decode(),
        )
        for match in TRACE_RE.finditer(data)
    ]


def report(label: str, records: list[Record]) -> None:
    grouped: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        grouped[record.command].append(record)

    total_input = sum(record.input_bytes for record in records)
    total_output = sum(record.output_bytes for record in records)
    total_elapsed = sum(record.elapsed_us for record in records)
    transport_requests = sum(record.command != "synchronize" for record in records)

    print(
        f"[{label}] entries={len(records)} transport_requests={transport_requests} "
        f"input_bytes={total_input} output_bytes={total_output} "
        f"elapsed_us={total_elapsed}"
    )
    print("command\tcount\tinput_bytes\toutput_bytes\telapsed_us\tmean_us")
    for command in sorted(grouped):
        command_records = grouped[command]
        elapsed = sum(record.elapsed_us for record in command_records)
        print(
            f"{command}\t{len(command_records)}\t"
            f"{sum(record.input_bytes for record in command_records)}\t"
            f"{sum(record.output_bytes for record in command_records)}\t"
            f"{elapsed}\t{elapsed / len(command_records):.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--generated-tokens", type=int, default=0)
    args = parser.parse_args()

    records = parse(args.trace)
    if not records:
        raise SystemExit("no rpc-trace records found")
    if any(record.status != "ok" for record in records):
        raise SystemExit("trace contains failed RPC records")

    first_graph = next(
        index for index, record in enumerate(records)
        if record.command in {"graph_compute", "graph_recompute"}
    )
    last_graph_compute = max(
        index for index, record in enumerate(records)
        if record.command == "graph_compute"
    )
    cleanup = next(
        (
            index for index in range(last_graph_compute + 1, len(records))
            if records[index].command in {"get_device_memory", "free_buffer"}
        ),
        len(records),
    )

    report("all", records)
    report("model-load-and-setup", records[:first_graph])
    report("inference", records[first_graph:cleanup])
    steady = records[last_graph_compute + 1:cleanup]
    report("steady-after-final-graph-build", steady)

    token_outputs = sum(
        record.command == "get_tensor" and record.output_bytes > 0
        for record in steady
    )
    if token_outputs:
        transport_requests = sum(
            record.command != "synchronize" for record in steady
        )
        print(
            "[steady-per-output] "
            f"outputs={token_outputs} "
            f"dispatcher_entries={len(steady) / token_outputs:.3f} "
            f"transport_requests={transport_requests / token_outputs:.3f} "
            f"input_bytes={sum(r.input_bytes for r in steady) / token_outputs:.3f} "
            f"output_bytes={sum(r.output_bytes for r in steady) / token_outputs:.3f} "
            f"elapsed_us={sum(r.elapsed_us for r in steady) / token_outputs:.3f}"
        )

    first_steady_output = next(
        index for index in range(last_graph_compute + 1, cleanup)
        if records[index].command == "get_tensor" and records[index].output_bytes > 0
    )
    repeat_start = next(
        index for index in range(first_steady_output + 1, cleanup)
        if records[index].command == "set_tensor"
    )
    repeated = records[repeat_start:cleanup]
    repeated_outputs = sum(
        record.command == "get_tensor" and record.output_bytes > 0
        for record in repeated
    )
    report("repeated-token-loop", repeated)
    if repeated_outputs:
        print(
            f"[repeated-per-output] outputs={repeated_outputs} "
            f"dispatcher_entries={len(repeated) / repeated_outputs:.3f} "
            f"transport_requests="
            f"{sum(r.command != 'synchronize' for r in repeated) / repeated_outputs:.3f} "
            f"input_bytes={sum(r.input_bytes for r in repeated) / repeated_outputs:.3f} "
            f"output_bytes={sum(r.output_bytes for r in repeated) / repeated_outputs:.3f} "
            f"elapsed_us={sum(r.elapsed_us for r in repeated) / repeated_outputs:.3f}"
        )

    if args.generated_tokens:
        inference = records[first_graph:cleanup]
        print(
            "[inference-per-requested-token] "
            f"dispatcher_entries={len(inference) / args.generated_tokens:.3f} "
            f"transport_requests="
            f"{sum(r.command != 'synchronize' for r in inference) / args.generated_tokens:.3f} "
            f"input_bytes={sum(r.input_bytes for r in inference) / args.generated_tokens:.3f} "
            f"output_bytes={sum(r.output_bytes for r in inference) / args.generated_tokens:.3f} "
            f"elapsed_us={sum(r.elapsed_us for r in inference) / args.generated_tokens:.3f}"
        )


if __name__ == "__main__":
    main()
