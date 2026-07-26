#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-/home/cheng/miniconda3/envs/haptic-cache/bin/python}"
"$PYTHON_BIN" -u -m src.select_phase4i3_monotonic_ttc_scales_oof \
  --section phase4i3_monotonic_ttc_scale_oof_v1 "$@"
