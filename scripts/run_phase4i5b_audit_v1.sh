#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PHASE4I5A_QUERY="outputs/cache/phase4i5a_far_safety_recalibration_oof_v1_queries.csv"
if [[ ! -s "$PHASE4I5A_QUERY" ]]; then
  echo "Phase4I.5a outputs are missing; building them first."
  bash scripts/run_phase4i5a_oof_v1.sh
fi

bash scripts/audit_phase4i5b_temporal_far_false_negatives_v1.sh
