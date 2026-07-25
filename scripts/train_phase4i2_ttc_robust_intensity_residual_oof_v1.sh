#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-/home/cheng/miniconda3/envs/haptic-cache/bin/python}"
"$PYTHON_BIN" -u -m src.train_phase4i2_ttc_robust_intensity_residual \
  --section phase4i2_ttc_robust_intensity_residual_oof_v1 "$@"
