#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/shiba/multikernel-experiment
python3 -m venv "$ROOT/research/piper-venv"
"$ROOT/research/piper-venv/bin/pip" install -r "$ROOT/model-api/tts-python-requirements.txt"
