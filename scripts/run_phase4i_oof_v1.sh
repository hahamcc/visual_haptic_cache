#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Phase4H.2 reuses its compatible checkpoints and completed query metrics while
# exporting the new per-candidate factorized inputs required by Phase4I.
bash scripts/evaluate_phase4h_factorized_intensity_oof_v1.sh
bash scripts/train_phase4i_factorized_residual_cascade_oof_v1.sh
