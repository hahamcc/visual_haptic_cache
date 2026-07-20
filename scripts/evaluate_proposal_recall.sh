#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-${HOME}/miniconda3/envs/haptic-cache/bin/python}"
"${PYTHON_BIN}" -m src.evaluate_proposal_recall --config configs/default.yaml "$@"
