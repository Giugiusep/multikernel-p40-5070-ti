#!/usr/bin/env python3
"""Follow up high-context limits after the first sequential matrix."""

import json

from probe_context import ROOT, probe


CASES = [
    ("ling-3-flash-iq2-s", 262144, None, None, None),
    ("nemotron-3.5-lightning-q8", 1048576, None, None, None),
    ("k2-horizon-mova-q4-k-m", 262144, "q8_0", None, None),
    ("mistral-small-4-iq2-xxs", 160000, None, None, 8),
]


def main() -> None:
    results = ROOT / "context-probe-results.json"
    data = json.loads(results.read_text()) if results.exists() else []
    for model_id, context, cache_type, batch_size, ngl in CASES:
        print(f"Probing {model_id} at {context}, KV {cache_type}, ngl {ngl}...", flush=True)
        try:
            row = probe(model_id, context, 240, cache_type, batch_size, ngl)
        except Exception as exc:
            row = {"model": model_id, "context": context, "error": repr(exc)}
        data.append(row)
        results.write_text(json.dumps(data, indent=2) + "\n")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
