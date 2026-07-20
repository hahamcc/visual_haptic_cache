#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-${HOME}/miniconda3/envs/haptic-cache/bin/python}"
"${PYTHON_BIN}" -m src.train_contact_region \
  --config configs/default.yaml \
  --section contact_region_masked_16 "$@"
