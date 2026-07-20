#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-${HOME}/miniconda3/envs/haptic-cache/bin/python}"
"${PYTHON_BIN}" -m src.evaluate_dual_gate --config configs/default.yaml --section dual_gate_masked_16 "$@"
