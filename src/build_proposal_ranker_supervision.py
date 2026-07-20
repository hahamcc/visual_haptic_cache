from __future__ import annotations

import argparse
import math
from collections import Counter

from .config import load_config, project_path
from .utils import read_csv_rows, write_csv_rows, write_json


FIELDS = [
    "dataset_split", "candidate_source", "record_id", "image_name", "probe", "oof_fold",
    "case_type", "supervision_enabled", "query_weight", "candidate_rank", "candidate_role",
    "candidate_x", "candidate_y", "candidate_heatmap_score", "candidate_error_px", "candidate_box48",
]


def parse_points(value: str) -> list[tuple[float, float, float]]:
    points = []
    for item in value.split(";"):
        if not item:
            continue
        x, y, score = item.split(",")
        points.append((float(x), float(y), float(score)))
    return points


def classify(row: dict[str, str], topk: int) -> tuple[str, list[dict[str, float | bool]]]:
    target_x = float(row["target_x"])
    target_y = float(row["target_y"])
    points = parse_points(row["topk_points"])
    if len(points) != topk:
        raise ValueError(f"Expected {topk} candidates for {row['image_name']}, found {len(points)}")
    candidates = []
    for x, y, score in points:
        error = math.hypot(x - target_x, y - target_y)
        candidates.append({"x": x, "y": y, "score": score, "error": error, "box48": abs(x - target_x) <= 24.0 and abs(y - target_y) <= 24.0})
    if bool(candidates[0]["box48"]):
        case_type = "easy"
    elif any(bool(candidate["box48"]) for candidate in candidates):
        case_type = "rank_hard"
    else:
        case_type = "proposal_miss"
    return case_type, candidates


def query_weight(case_type: str, probe: int, hard_weight: float, far_multiplier: float) -> float:
    if case_type != "rank_hard":
        return 1.0
    return hard_weight * (far_multiplier if probe >= 75 else 1.0)


def candidate_rows(
    prediction_rows: list[dict[str, str]],
    source: str,
    topk: int,
    hard_weight: float,
    far_multiplier: float,
    training: bool,
    cases_by_image: dict[str, str] | None = None,
) -> tuple[list[dict[str, str]], Counter]:
    output = []
    counts: Counter = Counter()
    for row in prediction_rows:
        case_type, candidates = classify(row, topk)
        if cases_by_image is not None and cases_by_image.get(row["image_name"]) != case_type:
            raise ValueError(f"OOF case mismatch for {row['image_name']}")
        counts[case_type] += 1
        if training and case_type == "proposal_miss":
            continue
        positive_rank = min(
            (index for index, candidate in enumerate(candidates) if bool(candidate["box48"])),
            key=lambda index: float(candidates[index]["error"]),
            default=None,
        )
        for index, candidate in enumerate(candidates):
            if training and case_type == "rank_hard":
                role = "positive" if index == positive_rank else "current_top1_hard_negative" if index == 0 else "auxiliary_negative"
                enabled = "1"
            elif training and case_type == "easy":
                role = "stability_top1" if index == 0 else "stability_negative"
                enabled = "1"
            else:
                role = "validation_candidate"
                enabled = "0"
            output.append({
                "dataset_split": row["dataset_split"],
                "candidate_source": source,
                "record_id": row["record_id"],
                "image_name": row["image_name"],
                "probe": row["probe"],
                "oof_fold": row.get("oof_fold", ""),
                "case_type": case_type,
                "supervision_enabled": enabled,
                "query_weight": f"{query_weight(case_type, int(row['probe']), hard_weight, far_multiplier) if training else 0.0:.3f}",
                "candidate_rank": str(index + 1),
                "candidate_role": role,
                "candidate_x": f"{float(candidate['x']):.3f}",
                "candidate_y": f"{float(candidate['y']):.3f}",
                "candidate_heatmap_score": f"{float(candidate['score']):.6f}",
                "candidate_error_px": f"{float(candidate['error']):.3f}",
                "candidate_box48": "1" if bool(candidate["box48"]) else "0",
            })
    return output, counts


def build(config_path: str, section: str, train_only: bool) -> dict:
    config = load_config(config_path)
    cfg = config[section]
    topk = int(cfg.get("topk", 10))
    hard_weight = float(cfg.get("hard_query_weight", 4.0))
    far_multiplier = float(cfg.get("far_hard_multiplier", 1.5))
    case_rows = read_csv_rows(project_path(cfg["train_oof_cases_csv"]))
    cases_by_image = {row["image_name"]: row["case_type"] for row in case_rows}
    train_predictions = read_csv_rows(project_path(cfg["train_oof_predictions_csv"]))
    train_rows, train_counts = candidate_rows(
        train_predictions, "oof", topk, hard_weight, far_multiplier, True, cases_by_image,
    )
    write_csv_rows(project_path(cfg["train_output_csv"]), train_rows, FIELDS)
    summary = {
        "policy": "Training supervision is OOF-only. Rank-hard queries receive pairwise roles; easy queries receive stability roles; proposal misses are excluded from loss.",
        "topk": topk,
        "hard_query_weight": hard_weight,
        "far_hard_multiplier": far_multiplier,
        "train_queries": len(train_predictions),
        "train_case_counts": dict(train_counts),
        "train_supervision_rows": len(train_rows),
        "train_output_csv": str(project_path(cfg["train_output_csv"])),
    }
    if not train_only:
        validation_predictions = [
            row for row in read_csv_rows(project_path(cfg["validation_predictions_csv"]))
            if row["dataset_split"] == "val"
        ]
        validation_rows, validation_counts = candidate_rows(
            validation_predictions, "frozen_c2_refit", topk, hard_weight, far_multiplier, False,
        )
        write_csv_rows(project_path(cfg["validation_output_csv"]), validation_rows, FIELDS)
        summary.update({
            "validation_queries": len(validation_predictions),
            "validation_case_counts": dict(validation_counts),
            "validation_candidate_rows": len(validation_rows),
            "validation_output_csv": str(project_path(cfg["validation_output_csv"])),
        })
    write_json(project_path(cfg["metrics_json"]), summary)
    print(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Construct OOF train supervision and validation candidates for the Phase35 ranker.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--section", default="proposal_ranker_supervision_phase35")
    parser.add_argument("--train-only", action="store_true")
    args = parser.parse_args()
    build(args.config, args.section, args.train_only)


if __name__ == "__main__":
    main()
