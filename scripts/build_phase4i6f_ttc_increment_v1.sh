#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-/home/cheng/miniconda3/envs/haptic-cache/bin/python}"
"$PYTHON_BIN" -u -m src.build_phase4i6f_ttc_increment \
  --section phase4i6f_ttc_increment_v1 "$@"
