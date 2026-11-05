"""Restore original API after the benchmark service exits or times out."""
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "model-api"))
from validate_tts_api import request

subprocess.run(["systemctl", "--user", "start", "shiba-model-api.service"], check=True)
for attempt in range(30):
    try:
        health = json.loads(request("/health"))
        break
    except OSError:
        time.sleep(1)
else:
    raise RuntimeError("API did not become healthy after benchmark")
state_path = HERE / "paired-state.json"
state = json.loads(state_path.read_text()) if state_path.exists() else {}
model = state.get("restore_model")
if model:
    request("/v1/switch", {"model": model})
health = json.loads(request("/health"))
(HERE / "api-restoration.json").write_text(json.dumps(health, indent=2) + "\n")
