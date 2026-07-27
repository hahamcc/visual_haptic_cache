#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-/home/cheng/miniconda3/envs/haptic-cache/bin/python}"
"$PYTHON_BIN" -u -m src.recalibrate_phase4i5a_far_threshold_oof \
  --section phase4i5a_far_safety_recalibration_oof_v1 "$@"
