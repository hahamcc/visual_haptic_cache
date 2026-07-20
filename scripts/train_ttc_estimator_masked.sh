#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-${HOME}/miniconda3/envs/haptic-cache/bin/python}"
for WINDOW in 8 16 32; do
  "${PYTHON_BIN}" -m src.train_ttc_estimator \
    --config configs/default.yaml \
    --section "ttc_estimator_masked_${WINDOW}" "$@"
done
