#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-/home/cheng/miniconda3/envs/haptic-cache/bin/python}"
"$PYTHON_BIN" -u -m src.audit_phase4i5b_temporal_far_false_negatives \
  --section phase4i5b_temporal_far_false_negative_audit_v1 "$@"
