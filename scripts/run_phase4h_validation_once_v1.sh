#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
bash scripts/build_phase4h_validation_features_v1.sh "$@"
bash scripts/evaluate_phase4h_validation_v1.sh "$@"
