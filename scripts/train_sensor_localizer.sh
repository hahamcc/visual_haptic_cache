#!/usr/bin/env bash
set -euo pipefail

python -m src.train_sensor_localizer --config configs/default.yaml "$@"
