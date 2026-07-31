#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

A="outputs/metrics/phase4i6c_robust_motion_quality_records_v1.csv"
B="outputs/metrics/phase4i6d_robust_motion_quality_records_b_v1.csv"
if [[ ! -s "$A" || ! -s "$B" ]]; then
  echo "Phase4I.6E V2 needs completed robust audits from pools A and B."
  exit 1
fi

bash scripts/select_phase4i6e_visual_motion_pairs_v2.sh
