#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/cheng/miniconda3/envs/haptic-cache/bin/python}"

cd "$PROJECT_DIR"
"$PYTHON_BIN" -u -m src.build_phase4d_oof_cache_confidence --section phase4d_oof_cache_confidence_v4 "$@"
