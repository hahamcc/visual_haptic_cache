#!/usr/bin/env bash
set -euo pipefail

mkdir -p outputs/logs
LOG_PATH="outputs/logs/phase4i7_augmented_ttc_oof_v1.log"

{
  echo "[$(date -Is)] Phase4I.7 feature build started"
  bash scripts/build_phase4i7_augmented_ttc_features_v1.sh
  echo "[$(date -Is)] Phase4I.7 control/augmented OOF training started"
  bash scripts/train_phase4i7_augmented_ttc_oof_v1.sh
  echo "[$(date -Is)] Phase4I.7 completed"
} 2>&1 | tee "$LOG_PATH"
