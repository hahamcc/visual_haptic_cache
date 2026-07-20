#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-${HOME}/miniconda3/envs/haptic-cache/bin/python}"
"${PYTHON_BIN}" -m src.build_expanded_region_dataset --config configs/default.yaml --section expanded_region_dataset_masked "$@"
