#!/usr/bin/env python3
"""Exercise the switchable API one model at a time and save a small report."""

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent
URL = "http://127.0.0.1:19080"


def request(path: str, payload: dict, key: str) -> dict:
    req = urllib.request.Request(
        URL + path,
        data=json.dumps(payload).encode(),
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=720) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code}: {exc.read().decode()[:500]}") from exc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+")
    parser.add_argument("--output", type=Path, default=ROOT / "model-smoke-results.json")
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()
    key = (ROOT / "api-key").read_text().strip()
    results = json.loads(args.output.read_text()) if args.output.exists() else {}
    for model in args.models:
        start = time.monotonic()
        row: dict = {"model": model}
        row["max_tokens"] = args.max_tokens
        try:
            switched = request("/v1/switch", {"model": model}, key)
            row["load_seconds"] = round(time.monotonic() - start, 2)
            row["switch"] = switched
            start = time.monotonic()
            response = request("/v1/chat/completions", {
                "model": model,
                "messages": [{"role": "user", "content": "Reply with the numeral 4."}],
                "temperature": 0,
                "max_tokens": args.max_tokens,
            }, key)
            row["answer_seconds"] = round(time.monotonic() - start, 2)
            message = response["choices"][0]["message"]
            row["content"] = message.get("content")
            row["reasoning_excerpt"] = (message.get("reasoning_content") or "")[:180]
            row["usage"] = response.get("usage")
            row["finish_reason"] = response["choices"][0].get("finish_reason")
            row["passed"] = row["content"] == "4" and row["finish_reason"] == "stop"
        except (OSError, RuntimeError, KeyError, ValueError) as exc:
            row["error"] = str(exc)
        results[model] = row
        args.output.write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
