#!/usr/bin/env python3
"""Switch to the RTX-only IQ2_S profile and collect decode/prefill timings."""

import json
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent
API = "http://127.0.0.1:19080"
MODEL = "qwen3.8-27b-iq2-s-rtx"
NOTES = (
    "PCI Express connects the host CPU and peripherals using serialized lanes. "
    "Its bandwidth depends on link generation, lane count, topology, packet overhead, "
    "and whether transfers can use DMA. GPU memory is attached directly to the GPU, "
    "so kernels can read weights and activations without crossing the host interface. "
    "Memory throughput also depends on access patterns, cache behavior, and occupancy. "
)
PROMPT_TEMPLATE = (
    "Benchmark replicate {n}. Use these technical notes: " + NOTES * 18 +
    "Explain the difference between host-to-GPU PCIe transfers and GPU-local memory. "
    "Give 12 numbered points, be precise, and write a thorough answer."
)


def request(path: str, payload: dict, key: str, timeout: int = 600) -> dict:
    req = urllib.request.Request(
        API + path,
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code}: {exc.read().decode()[:500]}") from exc


def chat(key: str, prompt: str, max_tokens: int) -> dict:
    start = time.monotonic()
    result = request("/v1/chat/completions", {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
    }, key, timeout=240)
    elapsed = time.monotonic() - start
    return {
        "wall_s": round(elapsed, 3),
        "usage": result.get("usage", {}),
        "timings": result.get("timings", {}),
        "finish_reason": result["choices"][0].get("finish_reason"),
        "response_chars": len((result["choices"][0]["message"].get("reasoning_content") or "") +
                              (result["choices"][0]["message"].get("content") or "")),
    }


def main() -> None:
    key = (ROOT / "api-key").read_text().strip()
    started = time.monotonic()
    switched = request("/v1/switch", {"model": MODEL}, key, timeout=1800)
    load_s = round(time.monotonic() - started, 2)
    print(json.dumps({"switched": switched, "cold_load_s": load_s}), flush=True)

    warmup = chat(key, "Reply with READY and one short sentence.", 64)
    print(json.dumps({"warmup": warmup}), flush=True)

    measurements = []
    for index in range(1, 4):
        row = chat(key, PROMPT_TEMPLATE.format(n=index), 256)
        row["run"] = index
        measurements.append(row)
        print(json.dumps(row), flush=True)

    decode = [r.get("timings", {}).get("predicted_per_second") for r in measurements]
    prompt = [r.get("timings", {}).get("prompt_per_second") for r in measurements]
    decode = [float(v) for v in decode if v is not None]
    prompt = [float(v) for v in prompt if v is not None]
    report = {
        "model": MODEL,
        "cold_load_s": load_s,
        "warmup": warmup,
        "runs": measurements,
        "median_decode_tok_s": sorted(decode)[len(decode) // 2] if decode else None,
        "median_prompt_tok_s": sorted(prompt)[len(prompt) // 2] if prompt else None,
        "settings": {"RTX_only": True, "ngl": 99, "context": 180000,
                     "KV": "Q8_0", "batch": 512, "ubatch": 128, "flash_attention": True},
    }
    out = ROOT / "qwen3.8-27b-iq2-s-rtx-benchmark-180000.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"saved": str(out),
                      "median_decode_tok_s": report["median_decode_tok_s"],
                      "median_prompt_tok_s": report["median_prompt_tok_s"]}), flush=True)


if __name__ == "__main__":
    main()
