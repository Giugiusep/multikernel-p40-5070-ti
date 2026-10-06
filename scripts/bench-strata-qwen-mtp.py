#!/usr/bin/env python3
"""Isolated RTX-only Strata Qwen3.8 fixed-width correctness/throughput sweep.

Run only after both IQ2_XS shards pass verify-qwen-iq2-xs.py and Strata's
setup.py --no-start has produced strata-iq2_xs.json.
"""

import argparse
import csv
import json
import os
import pathlib
import re
import subprocess
import sys
import time


EXPERIMENT = pathlib.Path(__file__).resolve().parents[1]
STRATA = EXPERIMENT / "research/Strata"
DEFAULT_CONFIG = STRATA / "strata-iq2_xs.json"
RATES = (0, 1, 2, 3, 4, 7)  # draft depths; verifier widths are k+1


def prompt_ids(shard: pathlib.Path, target: int) -> list[int]:
    sys.path.insert(0, str(STRATA / "tools"))
    from strata_tokenizer import Tokenizer

    tokenizer = Tokenizer.from_gguf(shard)
    prefix = tokenizer.encode("<|im_start|>user\nRead this source text and explain the main idea.\n", True)
    suffix = tokenizer.encode("\n<|im_end|>\n<|im_start|>assistant\n", True)
    paths = [STRATA / "README.md", *sorted((STRATA / "src").rglob("*.cpp"))]
    body: list[int] = []
    for path in paths:
        text = f"\nFile: {path.relative_to(STRATA)}\n" + path.read_text(errors="replace")
        body.extend(tokenizer.encode(text))
        if len(body) >= target:
            break
    if not body:
        raise RuntimeError("empty prompt source")
    need = target - len(prefix) - len(suffix)
    if need < 1:
        raise ValueError("context target too short")
    while len(body) < need:
        body.extend(body[:min(len(body), need - len(body))])
    result = prefix + body[:need] + suffix
    assert len(result) == target
    return result


def parse_metrics(output: str) -> dict:
    out = re.search(r"^output\s*:\s*([0-9 ]+)$", output, re.MULTILINE)
    decode = re.search(r"^decode\s+(\d+) tokens in ([\d.]+) ms\s+->\s+([\d.]+) tok/s", output, re.MULTILINE)
    prefill = re.search(r"^prefill\s+(\d+) tokens in ([\d.]+) ms\s+->\s+([\d.]+) tok/s", output, re.MULTILINE)
    spec = re.search(r"^speculation\s+(\d+) rounds of (\d+), drafts accepted (\d+) of (\d+) \(([^)]+)\), ([\d.]+) tokens per round", output, re.MULTILINE)
    windows = re.search(r"^window sizes\s+(.+)$", output, re.MULTILINE)
    if not out or not decode or not prefill or not spec or not windows:
        raise ValueError("Strata output lacks required token/timing/acceptance fields")
    ids = [int(value) for value in out.group(1).split()]
    histogram = {int(width): int(count) for width, count in re.findall(r"T(\d+):(\d+)", windows.group(1))}
    rounds = int(spec.group(1))
    if sum(histogram.values()) != rounds:
        raise ValueError("verification width histogram does not sum to rounds")
    return {
        "output_ids": ids,
        "decode_tokens": int(decode.group(1)),
        "decode_ms": float(decode.group(2)),
        "decode_tok_s": float(decode.group(3)),
        "prefill_tokens": int(prefill.group(1)),
        "prefill_ms": float(prefill.group(2)),
        "prefill_tok_s": float(prefill.group(3)),
        "rounds": rounds,
        "max_verify_width": int(spec.group(2)),
        "windows_by_width": json.dumps(histogram, sort_keys=True),
        "mean_verify_width": sum(width * count for width, count in histogram.items()) / rounds,
        "mean_round_ms": float(decode.group(2)) / rounds,
        "drafts_accepted": int(spec.group(3)),
        "drafts_offered": int(spec.group(4)),
        "committed_per_round": float(spec.group(6)),
    }


def measure_process(command: list[str], env: dict, logfile: pathlib.Path) -> tuple[int, int, int, int]:
    max_rss_kib = 0
    min_available_kib = 1 << 60
    max_vram_mib = 0
    with logfile.open("w") as output:
        process = subprocess.Popen(command, cwd=STRATA, env=env, stdout=output, stderr=subprocess.STDOUT)
        sample = 0
        while process.poll() is None:
            status = pathlib.Path(f"/proc/{process.pid}/status")
            try:
                found = re.search(r"^VmRSS:\s+(\d+) kB", status.read_text(), re.MULTILINE)
                if found:
                    max_rss_kib = max(max_rss_kib, int(found.group(1)))
            except FileNotFoundError:
                pass
            meminfo = pathlib.Path("/proc/meminfo").read_text()
            found = re.search(r"^MemAvailable:\s+(\d+) kB", meminfo, re.MULTILINE)
            if found:
                min_available_kib = min(min_available_kib, int(found.group(1)))
            if sample % 4 == 0:
                try:
                    gpu = subprocess.run(
                        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                        capture_output=True, text=True, timeout=2,
                    )
                    if gpu.returncode == 0:
                        max_vram_mib = max(max_vram_mib, int(gpu.stdout.splitlines()[0].strip()))
                except (OSError, ValueError, subprocess.TimeoutExpired, IndexError):
                    pass
            time.sleep(1)
            sample += 1
        rc = process.wait()
    return rc, max_rss_kib, min_available_kib if min_available_kib < (1 << 60) else 0, max_vram_mib


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=pathlib.Path, default=DEFAULT_CONFIG)
    parser.add_argument("--context", type=int, default=128, help="actual live prompt tokens")
    parser.add_argument("--max-new", type=int, default=128)
    parser.add_argument("--depths", default="0,1,2,3,4,7", help="comma-separated MTP draft depths")
    parser.add_argument("--expert-cache", default="1", help="positive slot count; 1 minimizes exposure to Strata's known incorrect GPU-hit path")
    parser.add_argument("--bypass-expert-hits", action="store_true",
                        help="diagnostic Strata build only: force all GPU cache hits to host misses")
    parser.add_argument("--label", default="", help="short filename label for an isolated parameter arm")
    args = parser.parse_args()
    if args.expert_cache == "0":
        raise ValueError("--expert-cache 0 with --expert-profile means use all profile slots in Strata; use 1 for a minimal cache")
    config = json.loads(args.config.read_text())
    opts = config["args"]
    shard = pathlib.Path(opts[opts.index("--native") + 1])
    if not shard.is_file():
        raise FileNotFoundError(shard)
    depths = [int(value) for value in args.depths.split(",")]
    if any(k not in RATES for k in depths):
        raise ValueError(f"allowed draft depths: {RATES}")
    token_path = EXPERIMENT / f"logs/qwen38-strata-prompt-{args.context}.tokens"
    token_path.write_text(" ".join(map(str, prompt_ids(shard, args.context))) + "\n")
    env = dict(os.environ)
    dirs = [d for d in config.get("lib_dirs") or [] if pathlib.Path(d).is_dir()]
    if dirs:
        env["LD_LIBRARY_PATH"] = os.pathsep.join(dirs + ([env["LD_LIBRARY_PATH"]] if env.get("LD_LIBRARY_PATH") else []))
    if args.bypass_expert_hits:
        env["STRATA_DIAG_BYPASS_EXPERT_HITS"] = "1"
    if args.label and not re.fullmatch(r"[a-zA-Z0-9_-]+", args.label):
        raise ValueError("--label must contain only letters, digits, hyphens or underscores")
    mode = ("-bypass" if args.bypass_expert_hits else "") + (f"-{args.label}" if args.label else "")
    csv_path = EXPERIMENT / f"logs/qwen38-strata-context-{args.context}-cache-{args.expert_cache}{mode}-mtp.csv"
    baseline: list[int] | None = None
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "live_context", "expert_cache", "draft_depth_k", "verify_width_M", "max_new", "prefill_tokens",
            "prefill_ms", "prefill_tok_s", "decode_tokens", "decode_ms", "decode_tok_s",
            "rounds", "max_verify_width", "windows_by_width", "mean_verify_width", "mean_round_ms",
            "drafts_accepted", "drafts_offered", "committed_per_round",
            "max_rss_kib", "min_available_kib", "max_vram_mib", "tokens_match_M1", "log",
            "first_token_mismatch",
        ])
        writer.writeheader()
        for k in depths:
            m = k + 1
            log = EXPERIMENT / f"logs/qwen38-strata-context-{args.context}-cache-{args.expert_cache}{mode}-k{k}.log"
            extra = ["--tokens-file", str(token_path), "--max-new", str(args.max_new),
                     "--max-context", str(max(4096, args.context + args.max_new + 32)),
                     "--suffix-draft", "0", "--spec-min-p", "0",
                     "--expert-cache", args.expert_cache]
            if k == 0:
                # Native IQ requires --spec>=2, but actual width can be one.
                # An empty --mtp disables the drafter; --mtp-max-t 1 forces T1.
                extra += ["--spec", "2", "--mtp-max-t", "1", "--mtp", ""]
            else:
                extra += ["--spec", str(m), "--mtp-max-t", str(m)]
            if args.context > 8192:
                extra += ["--kv", "int8"]
            if args.context >= 65536:
                extra += ["--kv-resident", "32768"]
            if args.bypass_expert_hits:
                extra += ["--pcie-frac", "0", "--adapt-every", "0"]
            command = [config["exe"], *opts, *extra]
            print(f"context={args.context} k={k} M={m} start", flush=True)
            rc, rss, available, vram = measure_process(command, env, log)
            raw = log.read_text(errors="replace")
            if rc != 0:
                raise RuntimeError(f"Strata failed rc={rc}: {log}\n{raw[-2000:]}")
            if args.bypass_expert_hits and "DIAGNOSTIC: all expert cache hits bypassed" not in raw:
                raise AssertionError(f"diagnostic bypass was not activated: {log}")
            metrics = parse_metrics(raw)
            if metrics["decode_tokens"] != args.max_new or len(metrics["output_ids"]) != args.max_new:
                raise AssertionError(f"unexpected generated count k={k}: {log}")
            hist = json.loads(metrics["windows_by_width"])
            if k == 0 and (int(hist.get("1", 0)) != metrics["rounds"] or metrics["drafts_offered"] != 0):
                raise AssertionError(f"M1 control actually used drafts or multirow windows: {log}")
            if k > 0 and any(int(width) not in (1, m) for width, count in hist.items() if count):
                raise AssertionError(f"unexpected verification width at k={k}: {log}")
            generated = metrics.pop("output_ids")
            if baseline is None:
                baseline = generated
                matches = True
                first_mismatch = -1
            else:
                matches = generated == baseline
                first_mismatch = next((i for i, (a, b) in enumerate(zip(baseline, generated)) if a != b), -1)
                if first_mismatch < 0 and not matches:
                    first_mismatch = min(len(baseline), len(generated))
            row = {"live_context": args.context, "expert_cache": args.expert_cache,
                   "draft_depth_k": k, "verify_width_M": m,
                   "max_new": args.max_new, "max_rss_kib": rss,
                   "min_available_kib": available, "max_vram_mib": vram,
                   "tokens_match_M1": matches, "first_token_mismatch": first_mismatch,
                   "log": str(log), **metrics}
            writer.writerow(row)
            stream.flush()
            print(f"context={args.context} k={k} M={m} {metrics['decode_tok_s']:.2f} tok/s "
                  f"{metrics['drafts_accepted']}/{metrics['drafts_offered']} drafts "
                  f"matches_M1={matches} first_mismatch={first_mismatch}", flush=True)
    print(csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
