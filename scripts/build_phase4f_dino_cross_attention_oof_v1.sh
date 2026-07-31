#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/cheng/miniconda3/envs/haptic-cache/bin/python}"
cd "$PROJECT_DIR"
"$PYTHON_BIN" -u -m src.build_phase4f_dino_cross_attention_cache --section phase4f_dino_cross_attention_oof_v1 "$@"
