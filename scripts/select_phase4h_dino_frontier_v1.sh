#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/cheng/miniconda3/envs/haptic-cache/bin/python}"
cd "$(dirname "$0")/.."
"$PYTHON_BIN" -u -m src.select_phase4h_dino_frontier --section phase4h_dino_frontier_v1 "$@"
