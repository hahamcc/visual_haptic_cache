#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/cheng/miniconda3/envs/haptic-cache/bin/python}"

cd "$PROJECT_DIR"
"$PYTHON_BIN" -u -m src.train_pairwise_proposal_ranker --section proposal_ranker_pairwise_phase35_v3_linear_hg "$@"
