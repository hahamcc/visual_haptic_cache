#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/cheng/miniconda3/envs/haptic-cache/bin/python}"

cd "$PROJECT_DIR"
"$PYTHON_BIN" -u -m src.audit_phase4c_cache_confidence --section phase4c_cache_confidence_audit_v4 "$@"
