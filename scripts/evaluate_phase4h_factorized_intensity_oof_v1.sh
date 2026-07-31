#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-/home/cheng/miniconda3/envs/haptic-cache/bin/python}"
"$PYTHON_BIN" -u -m src.evaluate_phase4h_factorized_intensity_oof --section phase4h_factorized_intensity_oof_v1 "$@"
