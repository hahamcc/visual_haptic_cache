#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/cheng/miniconda3/envs/haptic-cache/bin/python}"
cd "$(dirname "$0")/.."
"$PYTHON_BIN" -u -m src.build_phase4f_dino_patch_similarity_cache --section phase4f_dino_patch_similarity_oof_v1 "$@"
