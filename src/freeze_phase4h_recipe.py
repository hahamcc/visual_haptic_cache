"""Freeze accepted Phase4H artifacts without reading the V2 final holdout."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .config import load_config, project_path
from .utils import write_json


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze(config_path: str, section: str) -> dict:
    cfg = load_config(config_path)[section]
    validation = load_json(project_path(cfg["validation_metrics_json"]))
    trust = load_json(project_path(cfg["trust_metrics_json"]))
    if not validation.get("accepted", False):
        raise RuntimeError("Phase4H validation is not accepted; recipe freezing is blocked")
    if not trust.get("gate", {}).get("enabled", False):
        raise RuntimeError("Phase4H cache-trust gate is disabled; recipe freezing is blocked")
    artifacts = {
        name: project_path(value)
        for name, value in cfg["artifacts"].items()
    }
    missing = [str(path) for path in artifacts.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Accepted Phase4H artifact is missing: {missing}")
    manifest = {
        "mode": "phase4h_frozen_recipe_v1",
        "status": "accepted_and_frozen",
        "artifacts": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in artifacts.items()
        },
        "validation_acceptance": validation,
        "trust_gate": trust["gate"],
        "final_holdout_policy": {
            "rows_read_during_freeze": 0,
            "allowed": True,
            "next_step": "build and evaluate the reserved V2 final holdout exactly once",
        },
    }
    write_json(project_path(cfg["manifest_json"]), manifest)
    print(
        {
            "status": manifest["status"],
            "artifacts": len(artifacts),
            "manifest_json": cfg["manifest_json"],
            "final_holdout_rows_read": 0,
        }
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze accepted Phase4H artifacts before final holdout.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--section", default="phase4h_final_recipe_v1")
    args = parser.parse_args()
    freeze(args.config, args.section)


if __name__ == "__main__":
    main()
