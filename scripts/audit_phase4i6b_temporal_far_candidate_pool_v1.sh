#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-/home/cheng/miniconda3/envs/haptic-cache/bin/python}"
"$PYTHON_BIN" -u -m src.audit_phase4i6b_temporal_far_candidate_pool \
  --section phase4i6b_temporal_far_candidate_pool_audit_v1 "$@"
