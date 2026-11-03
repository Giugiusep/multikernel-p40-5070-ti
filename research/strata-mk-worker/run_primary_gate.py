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

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("reference", "disabled", "shadow", "execute"), required=True)
    parser.add_argument("--tokens", type=int, default=32)
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
    if args.mode in ("shadow", "execute"):
        env["STRATA_MK_EXPERT_ENDPOINT"] = "1:6005"
        env["STRATA_MK_EXPERT_PAIRS"] = str(ROOT / "profile-first32.txt")
        env["STRATA_MK_EXPERT_MODE"] = args.mode
    log = ROOT / (args.mode + "-engine.log")
    result = {"mode": args.mode, "command": command, "context": 2048, "runs": []}
    result_path = ROOT / (args.mode + "-gate.json")
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
            for prompt in prompts:
                text = "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
                ids = tokenizer.encode(text, parse_special=True)
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
                                  "request_wall_seconds": time.monotonic() - start}
                        result["runs"].append(record)
                        result_path.write_text(json.dumps(result, indent=2) + "\n")
                        print(json.dumps(record), flush=True)
                        break
            if args.mode != "reference":
                ref = json.loads((ROOT / "reference-gate.json").read_text())
                assert len(ref["runs"]) == len(result["runs"])
                for a, b in zip(ref["runs"], result["runs"]):
                    assert a["prompt_ids"] == b["prompt_ids"]
                    assert a["tokens"] == b["tokens"], "committed tokens differ from reference"
                result["exact_committed_agreement"] = True
            if args.mode in ("shadow", "execute"):
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
