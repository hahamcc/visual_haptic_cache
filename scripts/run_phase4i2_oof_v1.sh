#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Refresh the frozen factor table so every candidate has deployable
# predicted-TTC, trajectory-quality, and crop-padding fields.
bash scripts/evaluate_phase4h_factorized_intensity_oof_v1.sh
bash scripts/train_phase4i2_ttc_robust_intensity_residual_oof_v1.sh
