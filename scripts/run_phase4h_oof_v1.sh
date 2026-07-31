#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
bash scripts/audit_phase4h_protocol_v1.sh "$@"
bash scripts/build_phase4h_dino_ablation_oof_v1.sh "$@"
bash scripts/select_phase4h_dino_frontier_v1.sh "$@"
bash scripts/build_phase4h_tactile_index_v1.sh "$@"
bash scripts/train_phase4h_dino_tactile_alignment_oof_v1.sh "$@"
bash scripts/train_phase4h_dino_gate_oof_v1.sh "$@"
bash scripts/evaluate_phase4h_dino_tactile_alignment_oof_v1.sh "$@"
