#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

AUDIT_QUERY="outputs/metrics/phase4i5b_temporal_far_false_negative_queries_v1.csv"
AUDIT_METRICS="outputs/metrics/phase4i5b_temporal_far_false_negative_audit_v1.json"
if [[ ! -s "$AUDIT_QUERY" || ! -s "$AUDIT_METRICS" ]]; then
  echo "Phase4I.5b outputs are missing; building the read-only audit first."
  bash scripts/run_phase4i5b_audit_v1.sh
fi

bash scripts/plan_phase4i6_temporal_far_collection_v1.sh
