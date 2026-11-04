"""Paired model benchmark. systemd ExecStopPost restores the API on every exit."""
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
    cases = [("reference", "1:6006", "ref"),
             ("offload", "1:6006", "base1"),
             ("offload", "1:6007", "packed1"),
             ("offload", "1:6007", "packed2"),
             ("offload", "1:6006", "base2")]
    for mode, endpoint, tag in cases:
        state["running"] = tag
        state["started"] = time.time()
        save(state)
        command = [str(py), str(ROOT / "run_primary_gate.py"), "--mode", mode,
                   "--endpoint", endpoint, "--pairs", str(ROOT / "profile-first32-iq2.txt"),
                   "--prompts", str(HERE / "prompts.json"), "--tag", "perf-20261104/" + tag,
                   "--tokens", "192", "--reference", str(HERE / "ref-reference-gate.json")]
        with (HERE / (tag + "-stdout.log")).open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=240)
        state["completed"].append(tag)
        save(state)
    state["phase"] = "complete; API restoration pending"
    save(state)


if __name__ == "__main__":
    main()
