from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from .config import load_config, project_path
from .utils import write_json


def file_fingerprint(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Required frozen artifact is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    stat = path.stat()
    return {"path": str(path), "bytes": stat.st_size, "sha256": digest.hexdigest()}


def git_commit() -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def accepted_improves(report: dict) -> bool:
    all_values, accepted = report["all"], report["accepted"]
    return bool(
        accepted["tactile_diff_mae"] < all_values["tactile_diff_mae"]
        and accepted["tactile_ssim"] > all_values["tactile_ssim"]
        and accepted["tactile_mask_iou"] >= all_values["tactile_mask_iou"]
        and accepted["tactile_best_top1_rate"] >= all_values["tactile_best_top1_rate"]
        and accepted["tactile_best_top3_rate"] >= all_values["tactile_best_top3_rate"]
    )


def freeze(config_path: str, section: str) -> dict:
    cfg = load_config(config_path)[section]
    unified_gate = json.loads(project_path(cfg["unified_gate_json"]).read_text())
    far_gate = json.loads(project_path(cfg["far_gate_json"]).read_text())
    validation = json.loads(project_path(cfg["validation_metrics_json"]).read_text())
    if not unified_gate.get("enabled") or unified_gate.get("threshold") is None:
        raise RuntimeError("Cannot freeze: unified cache gate is not enabled with a fixed threshold.")
    if not far_gate.get("enabled") or far_gate.get("threshold") is None:
        raise RuntimeError("Cannot freeze: far cache gate is not enabled with a fixed threshold.")
    report = validation["trust"]["validation_metrics"]
    if not accepted_improves(report["all"]):
        raise RuntimeError("Cannot freeze: overall independent validation does not pass the cache-quality guard.")
    if not accepted_improves(report["far_probe75_100"]):
        raise RuntimeError("Cannot freeze: far independent validation does not pass the cache-quality guard.")
    artifacts = {
        "c2_contact_model": file_fingerprint(project_path(cfg["c2_checkpoint"])),
        "multiscale_cache_ranker": file_fingerprint(project_path(cfg["cache_ranker_checkpoint"])),
        "unified_trust_model": file_fingerprint(project_path(cfg["unified_trust_checkpoint"])),
        "far_trust_model": file_fingerprint(project_path(cfg["far_gate_checkpoint"])),
        "unified_gate": file_fingerprint(project_path(cfg["unified_gate_json"])),
        "far_gate": file_fingerprint(project_path(cfg["far_gate_json"])),
        "development_split": file_fingerprint(project_path(cfg["samples_csv"])),
        "validation_metrics": file_fingerprint(project_path(cfg["validation_metrics_json"])),
        "recipe_config": file_fingerprint(project_path(config_path)),
    }
    manifest = {
        "mode": "phase4e_frozen_final_recipe", "git_commit": git_commit(), "artifacts": artifacts,
        "recipe": {
            "contact_box": "frozen C2 V4 Top-1 48x48 box; cache components may not modify it",
            "cache_ranker": "geometry shortlist -> 48 detail plus 96 context ranker -> Top-3",
            "near_mid_gate": {"source": "unified trust", "threshold": unified_gate["threshold"]},
            "far_gate": {"source": "far-only quality gate", "threshold": far_gate["threshold"]},
            "cache_miss": "return structured miss; no tactile generator is part of this recipe",
        },
        "validation_acceptance": {
            "overall": report["all"], "far_probe75_100": report["far_probe75_100"],
            "checks": {"overall_pass": True, "far_pass": True},
        },
        "sealed_final_holdout": {
            "rule": "split-0 records rec_00950 through rec_00999", "rows_read_while_freezing": 0,
            "policy": "No final predictions, cache candidates, threshold selection, or metric reads before one explicit final evaluation run.",
        },
    }
    write_json(project_path(cfg["manifest_json"]), manifest)
    print(manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the validated Phase4E cache recipe before the one-time final holdout run.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--section", default="phase4e_final_recipe_v1")
    args = parser.parse_args()
    freeze(args.config, args.section)


if __name__ == "__main__":
    main()
