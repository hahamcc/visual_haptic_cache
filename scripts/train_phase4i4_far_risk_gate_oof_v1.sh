#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-/home/cheng/miniconda3/envs/haptic-cache/bin/python}"
"$PYTHON_BIN" -u -m src.train_phase4i4_far_risk_gate_oof \
  --section phase4i4_far_risk_gate_oof_v1 "$@"
