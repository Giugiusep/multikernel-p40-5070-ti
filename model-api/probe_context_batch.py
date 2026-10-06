#!/usr/bin/env python3
"""Run sequential high-context startup probes; never keep two models loaded."""

import json
from pathlib import Path

from probe_context import ROOT, probe


CASES = [
    ("ling-3-flash-iq2-s", 160000),
    ("mimo-v2.6-q6-k", 262144),
    ("k2-horizon-mova-q4-k-m", 262144),
    ("nemotron-3.5-lightning-q8", 262144),
    ("mistral-small-4-iq2-xxs", 160000),
    ("qwen3.8-27b-q8", 262144),
    ("qwen3.8-27b-q4-xl", 262144),
    ("gemma-4-e4b-q2-xl", 131072),
    ("gemma-4-26b-iq3-xxs", 262144),
    ("glm-4.7-flash-q6-xl", 202752),
    ("muse-glimmer-30b-q8", 131072),
]


def main() -> None:
    results = ROOT / "context-probe-results.json"
    data = json.loads(results.read_text()) if results.exists() else []
    for model_id, context in CASES:
        print(f"Probing {model_id} at {context}...", flush=True)
        try:
            row = probe(model_id, context, 240)
        except Exception as exc:
            row = {"model": model_id, "context": context, "error": repr(exc)}
        data.append(row)
        results.write_text(json.dumps(data, indent=2) + "\n")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
