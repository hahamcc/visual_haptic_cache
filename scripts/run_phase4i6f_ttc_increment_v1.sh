#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PAIR_CSV="outputs/metrics/phase4i6e_visual_motion_pairs_v2.csv"
if [[ ! -s "$PAIR_CSV" ]]; then
  echo "Phase4I.6F needs the completed Phase4I.6E V2 pair proposal."
  exit 1
fi

bash scripts/build_phase4i6f_ttc_increment_v1.sh
