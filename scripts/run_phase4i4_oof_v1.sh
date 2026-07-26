#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PHASE4I3_QUERY="outputs/cache/phase4i3_monotonic_ttc_scale_oof_v1_queries.csv"

if [[ ! -s "$PHASE4I3_QUERY" ]]; then
  echo "Phase4I.3 outputs are missing; building them first."
  bash scripts/run_phase4i3_oof_v1.sh
fi

bash scripts/train_phase4i4_far_risk_gate_oof_v1.sh
