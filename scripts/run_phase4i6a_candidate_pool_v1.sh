#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

bash scripts/reserve_phase4i6_raw_candidate_pool_v1.sh
bash scripts/build_phase4i6_raw_candidate_pool_v1.sh
