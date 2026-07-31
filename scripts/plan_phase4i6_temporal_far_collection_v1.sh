#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-/home/cheng/miniconda3/envs/haptic-cache/bin/python}"
"$PYTHON_BIN" -u -m src.plan_phase4i6_temporal_far_collection \
  --section phase4i6_temporal_far_collection_plan_v1 "$@"
