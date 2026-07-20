#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/cheng/miniconda3/envs/haptic-cache/bin/python}"

cd "$PROJECT_DIR"
for SECTION in proposal_ranker_pairwise_phase35_h proposal_ranker_pairwise_phase35_hg proposal_ranker_pairwise_phase35_hgv; do
  echo "Running ${SECTION}"
  "$PYTHON_BIN" -u -m src.train_pairwise_proposal_ranker --section "$SECTION"
done
