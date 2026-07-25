#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Re-export the frozen factor candidate table with online-safe progress fields.
bash scripts/evaluate_phase4h_factorized_intensity_oof_v1.sh
bash scripts/train_phase4i1_factorized_residual_diagnostics_oof_v1.sh
