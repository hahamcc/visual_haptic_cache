#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$PROJECT_DIR"
bash scripts/build_oof_contact_proposals.sh "$@"
bash scripts/train_proposal_ranker_masked_16.sh
