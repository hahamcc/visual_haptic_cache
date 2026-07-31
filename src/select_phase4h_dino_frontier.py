"""Select at most two predeclared frozen-DINO recipes from strict OOF results."""
from __future__ import annotations

import argparse
import json

from .config import load_config, project_path
from .utils import write_json


def satisfies(candidate: dict, baseline: dict, mae_slack: float, require_iou: bool) -> bool:
    for regime in ("all", "far_probe75_100"):
        current, base = candidate["summary"][regime], baseline["summary"][regime]
        if current["oracle_top1"] < base["oracle_top1"]:
            return False
        if require_iou and current["tactile_mask_iou"] < base["tactile_mask_iou"]:
            return False
        if current["tactile_diff_mae"] > base["tactile_diff_mae"] + mae_slack:
            return False
    return True


def select(config_path: str, section: str) -> dict:
    cfg = load_config(config_path)[section]
    with project_path(cfg["ablation_metrics_json"]).open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    recipes = report.get("recipes", [])
    if not recipes:
        raise RuntimeError("Phase4H ablation report has no completed recipes")
    by_name = {item["recipe"]["name"]: item for item in recipes}
    baseline_name = str(cfg["baseline_recipe"])
    if baseline_name not in by_name:
        raise RuntimeError(f"Frozen DINO baseline recipe {baseline_name!r} was not completed")
    baseline = by_name[baseline_name]

    mae_candidates = [
        item for item in recipes
        if satisfies(item, baseline, mae_slack=0.0, require_iou=True)
    ]
    mae_choice = min(
        mae_candidates or [baseline],
        key=lambda item: (
            item["summary"]["all"]["tactile_diff_mae"],
            item["summary"]["far_probe75_100"]["tactile_diff_mae"],
            -item["summary"]["all"]["tactile_mask_iou"],
        ),
    )
    shape_candidates = [
        item for item in recipes
        if satisfies(
            item,
            baseline,
            mae_slack=float(cfg["shape_recipe_mae_slack"]),
            require_iou=False,
        )
    ]
    shape_choice = max(
        shape_candidates or [baseline],
        key=lambda item: (
            item["summary"]["all"]["tactile_mask_iou"],
            item["summary"]["far_probe75_100"]["tactile_mask_iou"],
            item["summary"]["all"]["oracle_top1"],
            -item["summary"]["all"]["tactile_diff_mae"],
        ),
    )
    selected = []
    for role, item in (("mae", mae_choice), ("shape", shape_choice)):
        if item["recipe"]["name"] not in {entry["recipe"]["name"] for entry in selected}:
            selected.append({"role": role, **item})
    output = {
        "mode": "phase4h_frozen_dino_pareto_frontier_v1",
        "baseline_recipe": baseline_name,
        "primary_recipe": selected[0]["recipe"]["name"],
        "selected": selected[:2],
        "selection_contract": {
            "mae_recipe": "minimum overall then far MAE, with overall/far IoU and Top1 no lower than baseline",
            "shape_recipe": f"maximum overall then far IoU, with MAE slack <= {float(cfg['shape_recipe_mae_slack']):.6f} and Top1 no lower than baseline",
            "max_recipes": 2,
            "source": "strict development OOF only",
        },
        "integrity": report["integrity"],
    }
    write_json(project_path(cfg["frontier_json"]), output)
    print(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Select the Phase4H frozen-DINO OOF frontier.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--section", default="phase4h_dino_frontier_v1")
    args = parser.parse_args()
    select(args.config, args.section)


if __name__ == "__main__":
    main()
