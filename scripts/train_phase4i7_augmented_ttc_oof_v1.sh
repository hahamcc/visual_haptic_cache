#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/cheng/miniconda3/envs/haptic-cache/bin/python}"
"$PYTHON_BIN" -u -m src.train_phase4i7_augmented_ttc_oof \
  --section phase4i7_augmented_ttc_oof_v1 "$@"
