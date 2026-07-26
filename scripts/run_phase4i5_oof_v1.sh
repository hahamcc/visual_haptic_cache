#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PHASE4I3_QUERY="outputs/cache/phase4i3_monotonic_ttc_scale_oof_v1_queries.csv"
if [[ ! -s "$PHASE4I3_QUERY" ]]; then
  echo "Phase4I.3 outputs are missing; building them first."
  bash scripts/run_phase4i3_oof_v1.sh
fi

bash scripts/build_phase4i5_temporal_dino_features_v1.sh
bash scripts/train_phase4i5_temporal_progress_oof_v1.sh
