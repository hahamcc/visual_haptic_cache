#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-/home/cheng/miniconda3/envs/haptic-cache/bin/python}"
"$PYTHON_BIN" -u -m src.train_phase4i_factorized_residual_cascade \
  --section phase4i1_factorized_residual_diagnostics_oof_v1 "$@"
