#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

SAMPLES="data/processed/phase4i6_temporal_far/candidate_pool/region_samples_auto.csv"
TRACKS="data/processed/phase4i6_temporal_far/candidate_pool/sensor_tracks_auto.csv"
if [[ ! -s "$SAMPLES" || ! -s "$TRACKS" ]]; then
  echo "Phase4I.6A candidate pool is missing; building it first."
  bash scripts/run_phase4i6a_candidate_pool_v1.sh
fi

bash scripts/audit_phase4i6b_temporal_far_candidate_pool_v1.sh
