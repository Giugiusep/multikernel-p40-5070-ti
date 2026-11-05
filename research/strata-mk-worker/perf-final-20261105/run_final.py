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
    cases = [(1024,"shadow-budget3"),(1024,"baseline-rtx"),(1024,"group-all1"),(1024,"group-budget2"),(1024,"group-budget3"),(1024,"group-singleton3"),(1024,"baseline-rtxb")]
    for count, tag in cases:
        names=subprocess.check_output(["ps","-eo","comm="],text=True).splitlines()
        compilers=[n.strip() for n in names if n.strip() in {"nvcc","ptxas","cc1plus","cc1","cicc","cudafe++"}]
        if compilers:raise RuntimeError("Timed replay refused while compilers are running: "+str(compilers))
        state["compiler_guard"] = "no visible compilers"
        state["running"] = tag
        state["started"] = time.time()
        save(state)
        command = [str(py), str(ROOT / "run_primary_gate.py"), "--mode", "disabled" if tag.startswith("baseline-rtx") else "shadow" if "shadow" in tag else "offload", "--no-suffix-draft", "--experimental-exe", str(ROOT / "primary-final-build/strata"),
                   "--endpoint", "1:6007" if tag.startswith("control") else "1:6008", "--pairs", str(HERE / f"ranked-{count}.txt"),
                   "--prompts", str(HERE / "prompts-warm.json"), "--tag", "perf-final-20261105/" + tag,
                   "--tokens", "192", "--reference", str(HERE / "expected-warm-reference.json")]
        with (HERE / (tag + "-stdout.log")).open("w") as log:
            env = os.environ.copy()
            env["STRATA_MK_EXPERT_ASYNC"] = "1"
            env["STRATA_MK_EXPERT_PROFILE"] = "1"
            env["STRATA_MK_EXPERT_SINGLETON_WIDE"] = "1" if tag=="group-singleton3" else "0"
            env["STRATA_MK_EXPERT_Q8K_INPUT"] = "1"
            env["STRATA_MK_EXPERT_MAX_ENTRIES"] = "2" if tag=="group-budget2" else "3" if "budget3" in tag or tag=="group-singleton3" else "80"
            env["STRATA_MK_EXPERT_PER_LAYER"] = "1" if tag=="group-all1" else "3"
            env["STRATA_MK_EXPERT_EXACT_CHECK"] = "1"
            env["STRATA_MK_EXPERT_MAX_PAIRS"] = str(count)
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=240)
        state["completed"].append(tag)
        save(state)
    state["phase"] = "complete; API restoration pending"
    save(state)


if __name__ == "__main__":
    main()
