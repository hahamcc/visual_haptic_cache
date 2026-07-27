#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

COMPLETED_PLAN="data/processed/phase4i6_temporal_far/collection_manifest.csv"
NEW_SAMPLES="data/processed/phase4i6_temporal_far/region_samples_auto.csv"
NEW_TRACKS="data/processed/phase4i6_temporal_far/sensor_tracks_auto.csv"

for path in "$COMPLETED_PLAN" "$NEW_SAMPLES" "$NEW_TRACKS"; do
  if [[ ! -s "$path" ]]; then
    echo "Phase4I.6 input is missing: $path"
    echo "Fill the collection manifest and build the new samples/tracks first."
    exit 2
  fi
done

bash scripts/validate_phase4i6_temporal_far_data_v1.sh
