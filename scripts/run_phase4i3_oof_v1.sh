#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PHASE4I2_QUERY="outputs/cache/phase4i2_ttc_robust_intensity_residual_oof_v1_queries.csv"

if [[ ! -s "$PHASE4I2_QUERY" ]]; then
  echo "Phase4I.2 outputs are incomplete; rebuilding them first."
  bash scripts/run_phase4i2_oof_v1.sh
fi

bash scripts/select_phase4i3_monotonic_ttc_scales_oof_v1.sh
