#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-/home/cheng/miniconda3/envs/haptic-cache/bin/python}"
"$PYTHON_BIN" -u -m src.select_phase4i6e_visual_motion_pairs \
  --section phase4i6e_visual_motion_pair_selection_v1 "$@"
