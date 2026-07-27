#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-/home/cheng/miniconda3/envs/haptic-cache/bin/python}"
"$PYTHON_BIN" -u -m src.reserve_phase4i6_raw_candidate_pool \
  --section phase4i6_raw_candidate_partition_v1 "$@"
