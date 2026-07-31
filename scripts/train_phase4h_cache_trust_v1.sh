#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/cheng/miniconda3/envs/haptic-cache/bin/python}"
cd "$(dirname "$0")/.."
"$PYTHON_BIN" -u -m src.train_phase4h_cache_trust --section phase4h_cache_trust_v1 "$@"
