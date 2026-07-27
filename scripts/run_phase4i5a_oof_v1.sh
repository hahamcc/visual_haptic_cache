#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PHASE4I5_QUERY="outputs/cache/phase4i5_temporal_progress_oof_v1_queries.csv"
if [[ ! -s "$PHASE4I5_QUERY" ]]; then
  echo "Phase4I.5 outputs are missing; building them first."
  bash scripts/run_phase4i5_oof_v1.sh
fi

bash scripts/recalibrate_phase4i5a_far_threshold_oof_v1.sh
