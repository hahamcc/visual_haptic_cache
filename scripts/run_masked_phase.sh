#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-${HOME}/miniconda3/envs/haptic-cache/bin/python}"

"${PYTHON_BIN}" -m src.train_contact_region \
  --config configs/default.yaml \
  --section contact_region_masked_16 "$@"

"${PYTHON_BIN}" -m src.evaluate_dual_gate \
  --config configs/default.yaml \
  --section dual_gate_masked_16

"${PYTHON_BIN}" -m src.evaluate_proposal_recall \
  --config configs/default.yaml \
  --section proposal_recall_masked_16

"${PYTHON_BIN}" -m src.audit_failure_records \
  --config configs/default.yaml \
  --section failure_record_audit_masked_16
