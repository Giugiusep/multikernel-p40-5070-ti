#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/shiba/multikernel-experiment
python3 -m venv "$ROOT/research/stt-venv"
"$ROOT/research/stt-venv/bin/pip" install -r "$ROOT/model-api/stt-python-requirements.txt"
