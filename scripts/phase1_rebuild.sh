#!/usr/bin/env sh
set -eu

python3 -m src.sensor_localizer --config configs/default.yaml
python3 -m src.build_manifest --config configs/default.yaml
python3 -m src.detect_contact_frame --config configs/default.yaml
python3 -m src.build_region_dataset --config configs/default.yaml
