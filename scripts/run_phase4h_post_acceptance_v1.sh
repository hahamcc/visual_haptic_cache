#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
# This module re-checks phase4h_validation_v1.json and refuses to run unless accepted=true.
bash scripts/train_phase4h_cache_trust_v1.sh "$@"
bash scripts/freeze_phase4h_recipe_v1.sh "$@"
