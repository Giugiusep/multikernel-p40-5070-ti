"""Run unchanged RTX settings through a standalone, memory-bounded correctness gate.

Use --mode reference first, then disabled/shadow/execute. No API config is edited.
"""
import argparse
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
import hashlib


def resource_snapshot(pid):
    """Read cgroup throttling and guest scheduling counters around each request."""
    result = {}
    try:
        relative = next(line.split("::", 1)[1] for line in Path("/proc/self/cgroup").read_text().splitlines() if "::" in line)
        group = Path("/sys/fs/cgroup") / relative.lstrip("/")
        result["memory_events"] = {k: int(v) for k, v in (line.split() for line in (group / "memory.events").read_text().splitlines())}
        result["memory_current"] = int((group / "memory.current").read_text())
        result["cpu_stat"] = {k: int(v) for k, v in (line.split() for line in (group / "cpu.stat").read_text().splitlines())}
        result["memory_pressure"] = (group / "memory.pressure").read_text().strip()
    except (OSError, StopIteration, ValueError):
        pass
    try:
        result["guest_cpu_ticks"] = [int(x) for x in Path("/proc/stat").read_text().splitlines()[0].split()[1:]]
        stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        result["process_minor_faults"] = int(stat[7])
        result["process_major_faults"] = int(stat[9])
        result["process_cpu_ticks"] = int(stat[11]) + int(stat[12])
    except (OSError, ValueError):
        pass
    return result

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("reference", "disabled", "shadow", "execute", "offload"), required=True)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--endpoint", default="1:6005")
    parser.add_argument("--tag", default="")
    parser.add_argument("--reference", type=Path, default=ROOT / "reference-gate.json")
    parser.add_argument("--pairs", type=Path, default=ROOT / "profile-first32.txt")
    parser.add_argument("--prompts", type=Path, help="JSON list of fixed prompts for paired comparisons")
    args = parser.parse_args()
    config = json.loads((BASE / "model-api/strata-iq2-xs-safe.json").read_text())
    command = [config["exe"] if args.mode == "reference" else str(ROOT / "primary-build/strata"), *config["args"], "--serve"]
    # Same test context in all arms; production context remains unchanged.
    command[command.index("--max-context") + 1] = "2048"
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(config["lib_dirs"])
    env["STRATA_DIAG_BYPASS_EXPERT_HITS"] = "1"
    for key in ("STRATA_MK_EXPERT_ENDPOINT", "STRATA_MK_EXPERT_PAIRS", "STRATA_MK_EXPERT_MODE"):
        env.pop(key, None)
    if args.mode in ("shadow", "execute", "offload"):
        env["STRATA_MK_EXPERT_ENDPOINT"] = args.endpoint
        env["STRATA_MK_EXPERT_PAIRS"] = str(args.pairs)
        env["STRATA_MK_EXPERT_MODE"] = args.mode
    prefix = (args.tag + "-" if args.tag else "") + args.mode
    log = ROOT / (prefix + "-engine.log")
    result = {"mode": args.mode, "command": command, "context": 2048, "runs": [],
              "endpoint": args.endpoint, "pairs": str(args.pairs),
              "pairs_sha256": hashlib.sha256(args.pairs.read_bytes()).hexdigest(),
              "binary_sha256": hashlib.sha256(Path(command[0]).read_bytes()).hexdigest()}
    result_path = ROOT / (prefix + "-gate.json")
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    with log.open("w", buffering=1) as stderr:
        p = subprocess.Popen(command, cwd=config["cwd"], env=env, stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=stderr, text=True, bufsize=1, start_new_session=True)
        lines = queue.Queue()
        def read_lines():
            for line in p.stdout:
                lines.put(line.rstrip())
            lines.put(None)
        threading.Thread(target=read_lines, daemon=True).start()
        def get_line(deadline):
            left = deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError("engine deadline")
            line = lines.get(timeout=left)
            if line is None:
                raise RuntimeError(f"engine exited {p.poll()}; see {log}")
            if line.startswith("ERR"):
                raise RuntimeError(line)
            return line
        try:
            deadline = time.monotonic() + 900
            while not get_line(deadline).startswith("READY"):
                pass
            print("READY", args.mode, flush=True)
            sys.path.insert(0, str(Path(config["cwd"]) / "tools"))
            from strata_tokenizer import Tokenizer
            tpath = Path(config["tokenizer"])
            vocab = json.loads((tpath / "vocab.json").read_text())
            tokens = [None] * len(vocab)
            for token, index in vocab.items():
                tokens[index] = token
            tokenizer = Tokenizer(tokens, (tpath / "merges.txt").read_text().split("\n"),
                                  json.loads((tpath / "token_type.json").read_text()))
            prompts = ["What is the capital of France? Reply with only the city name.",
                       "Explain how a bicycle works in one detailed paragraph."]
            if args.prompts:
                prompts = json.loads(args.prompts.read_text())
                if not isinstance(prompts, list) or not prompts or not all(isinstance(p, str) and p for p in prompts):
                    raise ValueError("prompts must be a nonempty JSON list of strings")
            for prompt in prompts:
                text = "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
                ids = tokenizer.encode(text, parse_special=True)
                resources_before = resource_snapshot(p.pid)
                p.stdin.write(f"GEN {args.tokens} temperature=0 top_p=1 top_k=1 seed=42 " + ",".join(map(str, ids)) + "\n")
                p.stdin.flush()
                start = time.monotonic()
                output = []
                deadline = start + 300
                while True:
                    line = get_line(deadline)
                    if line.startswith("T "):
                        output.append(int(line[2:]))
                    elif line.startswith("DONE"):
                        record = {"prompt": prompt, "prompt_ids": ids, "tokens": output,
                                  "text": tokenizer.decode(output), "done": line,
                                  "request_wall_seconds": time.monotonic() - start,
                                  "resources_before": resources_before,
                                  "resources_after": resource_snapshot(p.pid)}
                        result["runs"].append(record)
                        result_path.write_text(json.dumps(result, indent=2) + "\n")
                        print(json.dumps(record), flush=True)
                        break
            if args.mode != "reference":
                ref = json.loads(args.reference.read_text())
                assert len(ref["runs"]) == len(result["runs"])
                for a, b in zip(ref["runs"], result["runs"]):
                    assert a["prompt_ids"] == b["prompt_ids"]
                    assert a["tokens"] == b["tokens"], "committed tokens differ from reference"
                result["exact_committed_agreement"] = True
            if args.mode in ("shadow", "execute", "offload"):
                assert "MK_EXPERT_ROUND" in log.read_text(), "no P40 expert work observed"
            result["passed"] = True
            result_path.write_text(json.dumps(result, indent=2) + "\n")
        finally:
            p.stdin.close()
            try:
                p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                p.terminate()
                try:
                    p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait()


if __name__ == "__main__":
    main()
