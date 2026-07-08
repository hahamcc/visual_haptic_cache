#!/usr/bin/env bash
set -euo pipefail

python -m src.train_contact_region --config configs/default.yaml "$@"
