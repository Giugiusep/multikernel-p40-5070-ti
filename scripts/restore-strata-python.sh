#!/bin/bash
set -euo pipefail
base=/home/shiba/multikernel-experiment
python3 -m venv "$base/research/strata-venv-resolute"
"$base/research/strata-venv-resolute/bin/pip" install -r "$base/model-api/strata-python-requirements.txt"
"$base/research/strata-venv-resolute/bin/python" -c 'import regex,jinja2; print("Strata Python imports OK")'
