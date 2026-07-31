#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/cheng/miniconda3/envs/haptic-cache/bin/python}"
"$PYTHON_BIN" -u -m src.build_phase4i7_augmented_ttc_features \
  --section phase4i7_augmented_ttc_oof_v1 "$@"
