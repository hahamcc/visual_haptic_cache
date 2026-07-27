#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PHASE4I6B="outputs/metrics/phase4i6b_temporal_far_candidate_record_audit_v1.csv"
if [[ ! -s "$PHASE4I6B" ]]; then
  echo "Phase4I.6B output is missing; building it first."
  bash scripts/run_phase4i6b_candidate_audit_v1.sh
fi

bash scripts/audit_phase4i6c_robust_motion_quality_v1.sh
