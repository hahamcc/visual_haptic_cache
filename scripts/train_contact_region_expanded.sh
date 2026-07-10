#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  CONDA_PYTHON="${HOME}/miniconda3/envs/haptic-cache/bin/python"
  if [[ -x "${CONDA_PYTHON}" ]]; then
    PYTHON_BIN="${CONDA_PYTHON}"
  fi
fi

"${PYTHON_BIN}" -m src.train_contact_region --config configs/default.yaml --section contact_region_expanded "$@"
