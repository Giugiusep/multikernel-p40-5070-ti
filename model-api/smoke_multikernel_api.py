#!/usr/bin/env python3
"""Check deterministic Nemotron output through the switchable API."""

import hashlib
import json
import time
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MODEL = "nemotron-3.5-lightning-q8-mk"
PROMPT = "Explain briefly why PCIe transfers are slower than on-device memory."
EXPECTED_SHA256 = "b57e083d9bca673789b51cf64802c8d05d459b20c61b6917505584565370f177"


def main() -> None:
    key = (ROOT / "api-key").read_text().strip()
    with urllib.request.urlopen("http://127.0.0.1:19080/health", timeout=10) as response:
        health = json.load(response)
    if health.get("active_model") != MODEL:
        raise SystemExit(f"unexpected active model: {health.get('active_model')!r}")
    model_request = urllib.request.Request(
        "http://127.0.0.1:19080/v1/models",
        headers={"Authorization": f"Bearer {key}"},
    )
    with urllib.request.urlopen(model_request, timeout=10) as response:
        models = json.load(response)["data"]
    selected = next((entry for entry in models if entry["id"] == MODEL), None)
    if selected is None or not selected["metadata"]["active"]:
        raise SystemExit("Multikernel alias missing or inactive in /v1/models")
    print(json.dumps({"active_model": MODEL, "context": selected["metadata"]["context_length"],
                      "models_available": len(models)}), flush=True)
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "temperature": 0,
        "max_tokens": 128,
    }).encode()
    for run in range(1, 4):
        request = urllib.request.Request(
            "http://127.0.0.1:19080/v1/chat/completions",
            data=body,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )
        start = time.monotonic()
        with urllib.request.urlopen(request, timeout=120) as response:
            result = json.load(response)
        message = result["choices"][0]["message"]
        output = (message.get("reasoning_content") or "") + (message.get("content") or "")
        digest = hashlib.sha256(output.encode()).hexdigest()
        timings = result.get("timings") or {}
        print(json.dumps({
            "run": run,
            "match": digest == EXPECTED_SHA256,
            "sha256": digest,
            "completion_tokens": result["usage"]["completion_tokens"],
            "decode_tok_s": round(timings.get("predicted_per_second", 0), 3),
            "wall_s": round(time.monotonic() - start, 3),
        }), flush=True)
        if digest != EXPECTED_SHA256:
            raise SystemExit("deterministic output differs from isolated 160K probe")


if __name__ == "__main__":
    main()
