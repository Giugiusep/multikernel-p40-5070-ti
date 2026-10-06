#!/usr/bin/env python3
"""Probe isolated model startup, correctness, and decode speed on port 19082."""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from model_switch_api import ENTRIES, K2, LLAMA, STRATA_SERVER


ROOT = Path(__file__).resolve().parent
PORT = 19082
RPC_LLAMA = "/home/shiba/llama.cpp/build-main-cuda-rpc-128/bin/llama-server"
BENCH_PROMPT = "Continue this list of integers, separated by commas: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,"


def memory() -> dict[str, int]:
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, _, value = line.partition(":")
        if key in {"MemAvailable", "SwapFree"}:
            values[key] = int(value.strip().split()[0]) // 1024
    return values


def benchmark_once(prompt: str, max_tokens: int) -> dict:
    body = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
    }).encode()
    request = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                                     data=body, headers={"Content-Type": "application/json"})
    start = time.monotonic()
    with urllib.request.urlopen(request, timeout=360) as response:
        result = json.load(response)
    elapsed = round(time.monotonic() - start, 3)
    message = result["choices"][0]["message"]
    content = (message.get("reasoning_content") or "") + (message.get("content") or "")
    usage = result.get("usage", {})
    return {"wall_s": elapsed, "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "completion_tok_s_wall": round(usage.get("completion_tokens", 0) / elapsed, 3),
            "finish_reason": result["choices"][0].get("finish_reason"),
            "output_sha256": hashlib.sha256(content.encode()).hexdigest(),
            "output_excerpt": content[:180],
            "output_text": content,
            "timings": result.get("timings")}


def probe(model_id: str, context: int, timeout: int, cache_type: str | None = None,
          batch_size: int | None = None, ngl: int | None = None,
          benchmark: bool = False, rpc: bool = False,
          tensor_split: str = "45,55", coordinator: bool = False,
          bench_prompt: str = BENCH_PROMPT, bench_tokens: int = 128,
          extra_repeats: int = 0, flash_attn: str | None = None,
          backend_sampling: bool = False,
          allow_unsafe_context: bool = False) -> dict:
    if (model_id == "mistral-small-4-iq2-xxs" and rpc and context >= 131072
            and not allow_unsafe_context):
        raise ValueError("Mistral RPC at 131K froze this VM; use --allow-unsafe-context only for an intentional retry")
    entry = ENTRIES[model_id]
    env = os.environ.copy()
    if entry["runner"] == "strata":
        if cache_type or batch_size or ngl is not None:
            raise ValueError("Strata uses its own KV and batch controls")
        config = json.loads(Path(entry["path"]).read_text())
        args = config["args"]
        args[args.index("--max-context") + 1] = str(context)
        config_path = ROOT / "logs" / f"strata-context-{context}.json"
        config_path.write_text(json.dumps(config, indent=2) + "\n")
        env["STRATA_DIAG_BYPASS_EXPERT_HITS"] = "1"
        command = ["/usr/bin/python3", STRATA_SERVER, "--engine", "strata",
                   "--config", str(config_path), "--host", "127.0.0.1", "--port", str(PORT)]
        expected = config["model_name"]
        gpu_layers = None
    else:
        if rpc and entry["runner"] == "k2":
            raise ValueError("the K2 publisher fork has no Multikernel RPC transport")
        exe = RPC_LLAMA if rpc or coordinator else (K2 if entry["runner"] == "k2" else LLAMA)
        gpu_layers = (99 if rpc else entry["ngl"]) if ngl is None else ngl
        command = [exe, "-m", entry["path"], "--alias", model_id, "--host", "127.0.0.1",
                   "--port", str(PORT), "-c", str(context), "-ngl", str(gpu_layers),
                   "-np", "1", "--jinja", "--no-ui"]
        if rpc:
            command += ["--rpc", "mkvsock:1:5002", "--device", "CUDA0,RPC0",
                        "--split-mode", "layer", "--tensor-split", tensor_split,
                        "--fit", "off"]
            env["GGML_MK_VSOCK_NO_READ_SLEEP"] = "1"
            env["GGML_RPC_NO_RDMA"] = "1"
            env.pop("GGML_MK_VSOCK_NO_WRITE_SLEEP", None)
            env.pop("GGML_MK_VSOCK_NO_PROGRESS_SLEEP", None)
        if cache_type:
            command += ["--cache-type-k", cache_type, "--cache-type-v", cache_type]
        if batch_size:
            command += ["--batch-size", str(batch_size), "--ubatch-size", str(min(128, batch_size))]
        if flash_attn:
            command += ["--flash-attn", flash_attn]
        if backend_sampling:
            command += ["--backend-sampling", "--samplers", "temperature"]
        expected = model_id
    suffix = ('-' + cache_type if cache_type else '') + ('-b' + str(batch_size) if batch_size else '') + ('-ngl' + str(ngl) if ngl is not None else '') + ('-rpc' + tensor_split.replace(',', '_') if rpc else '') + ('-coordinator' if coordinator and not rpc else '')
    log_path = ROOT / "logs" / f"context-{model_id}-{context}{suffix}.log"
    before = memory()
    row = {"model": model_id, "context": context, "ngl": gpu_layers,
           "cache_type": cache_type or "f16", "batch_size": batch_size,
           "flash_attn": flash_attn,
           "backend_sampling": backend_sampling,
           "rpc": rpc, "coordinator": coordinator,
           "bench_prompt": bench_prompt if benchmark else None,
           "bench_tokens": bench_tokens if benchmark else None,
           "tensor_split": tensor_split if rpc else None,
           "before_mib": before, "log": str(log_path)}
    start = time.monotonic()
    with log_path.open("wb") as log:
        proc = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            deadline = start + timeout
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    row["error"] = f"server exited {proc.returncode}"
                    break
                now = memory()
                if now["MemAvailable"] < 3072 or now["SwapFree"] < 512:
                    row["error"] = "memory safety threshold reached"
                    break
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as response:
                        if response.status == 200:
                            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=2) as models:
                                ids = {m["id"] for m in json.load(models)["data"]}
                            if expected in ids:
                                row["ready"] = True
                                break
                except (OSError, urllib.error.HTTPError):
                    pass
                time.sleep(2)
            else:
                row["error"] = "startup timeout"
            row["startup_s"] = round(time.monotonic() - start, 2)
            row["loaded_mib"] = memory()
            if row.get("ready") and benchmark:
                try:
                    row["warmup"] = benchmark_once(bench_prompt, bench_tokens)
                    row["benchmark"] = benchmark_once(bench_prompt, bench_tokens)
                    row["repeat"] = benchmark_once(bench_prompt, bench_tokens)
                    row["extra_repeats"] = [benchmark_once(bench_prompt, bench_tokens)
                                            for _ in range(extra_repeats)]
                except (OSError, ValueError, KeyError) as exc:
                    row["benchmark_error"] = str(exc)
            if proc.poll() is None:
                status = Path(f"/proc/{proc.pid}/status")
                for line in status.read_text().splitlines():
                    if line.startswith(("VmRSS:", "VmHWM:")):
                        name, amount = line.split(":", 1)
                        row[name + "_mib"] = int(amount.strip().split()[0]) // 1024
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait(timeout=10)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("context", type=int)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--cache-type", choices=["f16", "q8_0"], default=None)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--ngl", type=int)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--rpc", action="store_true")
    parser.add_argument("--coordinator", action="store_true")
    parser.add_argument("--tensor-split", default="45,55")
    parser.add_argument("--bench-prompt", default=BENCH_PROMPT)
    parser.add_argument("--bench-tokens", type=int, default=128)
    parser.add_argument("--extra-repeats", type=int, default=0)
    parser.add_argument("--flash-attn", choices=["on", "off", "auto"])
    parser.add_argument("--backend-sampling", action="store_true")
    parser.add_argument("--allow-unsafe-context", action="store_true")
    args = parser.parse_args()
    row = probe(args.model, args.context, args.timeout, args.cache_type,
                args.batch_size, args.ngl, args.benchmark, args.rpc,
                args.tensor_split, args.coordinator, args.bench_prompt,
                args.bench_tokens, args.extra_repeats, args.flash_attn,
                args.backend_sampling, args.allow_unsafe_context)
    results = ROOT / "context-probe-results.json"
    data = json.loads(results.read_text()) if results.exists() else []
    data.append(row)
    results.write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
