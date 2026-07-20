#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-${HOME}/miniconda3/envs/haptic-cache/bin/python}"
"${PYTHON_BIN}" -m src.evaluate_ttc_strategies --config configs/default.yaml "$@"
