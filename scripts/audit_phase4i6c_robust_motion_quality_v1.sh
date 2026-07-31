#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-/home/cheng/miniconda3/envs/haptic-cache/bin/python}"
"$PYTHON_BIN" -u -m src.audit_phase4i6c_robust_motion_quality \
  --section phase4i6c_robust_motion_quality_audit_v1 "$@"
