"""Paired model benchmark. systemd ExecStopPost restores the API on every exit."""
import os
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT.parents[1] / "model-api"))
from validate_tts_api import request


def save(state):
    (HERE / "paired-state.json").write_text(json.dumps(state, indent=2) + "\n")


def main():
    health = json.loads(request("/health"))
    state = {"restore_model": health.get("active_model"), "phase": "unloading API", "completed": []}
    save(state)
    request("/v1/unload", {})
    subprocess.run(["systemctl", "--user", "stop", "shiba-model-api.service"], check=True)
    state["phase"] = "API paused; benchmarking"
    save(state)
    py = ROOT.parent / "strata-venv-resolute/bin/python"
    cases = [(32,"control32"),(32,"prepack32"),(128,"prepack128"),(32,"control32b")]
    for count, tag in cases:
        state["running"] = tag
        state["started"] = time.time()
        save(state)
        command = [str(py), str(ROOT / "run_primary_gate.py"), "--mode", "reference" if tag=="fixedref" else "shadow" if "shadow" in tag else "offload", "--no-suffix-draft",
                   "--endpoint", "1:6007" if tag.startswith("control") else "1:6008", "--pairs", str(HERE / f"ranked-{count}.txt"),
                   "--prompts", str(HERE / "prompts.json"), "--tag", "perf-dp4a-20261105/" + tag,
                   "--tokens", "192", "--reference", str(HERE / "fixedref-reference-gate.json")]
        with (HERE / (tag + "-stdout.log")).open("w") as log:
            env = os.environ.copy()
            env["STRATA_MK_EXPERT_ASYNC"] = "1"
            env["STRATA_MK_EXPERT_EXACT_CHECK"] = "1"
            env["STRATA_MK_EXPERT_MAX_PAIRS"] = str(count)
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=240)
        state["completed"].append(tag)
        save(state)
    state["phase"] = "complete; API restoration pending"
    save(state)


if __name__ == "__main__":
    main()
