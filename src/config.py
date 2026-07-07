from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "dataset": {
        "root": "/mnt/data/chi/visgel/seen/images",
        "split": "0",
        "vision_name": "vision",
        "touch_name": "touch",
        "image_exts": [".jpg", ".jpeg", ".png"],
    },
    "makesense": {
        "images_dir": "data/makesense/images",
        "labels_csv": "data/makesense/images/labels/makesense_labels.csv",
    },
    "outputs": {
        "processed_dir": "data/processed",
        "phase1_debug_dir": "outputs/debug/phase1",
    },
}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path = "configs/default.yaml") -> dict[str, Any]:
    config_path = Path(path)
    loaded: dict[str, Any] = {}
    if config_path.exists() and config_path.read_text().strip():
        with config_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
            if not isinstance(data, dict):
                raise ValueError(f"Config must be a mapping: {config_path}")
            loaded = data
    return deep_merge(DEFAULT_CONFIG, loaded)


def project_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return Path.cwd() / path
