#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

bash scripts/reserve_phase4i6d_raw_candidate_pool_b_v1.sh

SAMPLES="data/processed/phase4i6_temporal_far/candidate_pool_b/region_samples_auto.csv"
TRACKS="data/processed/phase4i6_temporal_far/candidate_pool_b/sensor_tracks_auto.csv"
if [[ ! -s "$SAMPLES" || ! -s "$TRACKS" ]]; then
  bash scripts/build_phase4i6d_raw_candidate_pool_b_v1.sh
else
  echo "Phase4I.6D candidate pool B already exists; skipping the expensive build."
fi

bash scripts/audit_phase4i6d_candidate_pool_b_v1.sh
bash scripts/audit_phase4i6d_robust_motion_quality_b_v1.sh
