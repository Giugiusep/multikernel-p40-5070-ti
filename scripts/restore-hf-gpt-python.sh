#!/bin/bash
set -euo pipefail
base=/home/shiba/multikernel-experiment
venv="$base/research/hf-gpt-venv"
python3 -m venv "$venv"
"$venv/bin/pip" install torch==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128
"$venv/bin/pip" install -r "$base/model-api/hf-gpt-python-requirements.txt"
