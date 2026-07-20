#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-${HOME}/miniconda3/envs/haptic-cache/bin/python}"
"${PYTHON_BIN}" -m src.audit_failure_records --config configs/default.yaml "$@"
