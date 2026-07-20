from __future__ import annotations

import argparse
import copy
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from PIL import Image, ImageDraw

from .config import load_config, project_path
from .train_contact_region import train_contact_region
from .utils import read_csv_rows, write_csv_rows, write_json


CASE_FIELDS = [
    "dataset_split", "record_id", "image_name", "probe", "oof_fold", "case_type",
    "top1_error_px", "top10_min_error_px", "top1_box48", "top10_box48", "topk_points",
]

FOLD_MEMBERSHIP_FIELDS = ["oof_fold", "record_id", "role"]


def record_folds(records: list[str], count: int, seed: int) -> list[list[str]]:
    shuffled = sorted(records)
    random.Random(seed).shuffle(shuffled)
    return [shuffled[index::count] for index in range(count)]


def parse_points(value: str) -> list[tuple[float, float, float]]:
    points = []
    for item in value.split(";"):
        if not item:
            continue
        x, y, score = item.split(",")
        points.append((float(x), float(y), float(score)))
    return points


def classify_prediction(row: dict[str, str]) -> dict[str, str]:
    target_x = float(row["target_x"])
    target_y = float(row["target_y"])
    points = parse_points(row["topk_points"])
    if not points:
        raise ValueError(f"Missing Top-K points for {row['image_name']}")
    errors = [float(math.hypot(x - target_x, y - target_y)) for x, y, _ in points]
    boxes = [abs(x - target_x) <= 24.0 and abs(y - target_y) <= 24.0 for x, y, _ in points]
    if boxes[0]:
        case_type = "easy"
    elif any(boxes):
        case_type = "rank_hard"
    else:
        case_type = "proposal_miss"
    return {
        "dataset_split": row["dataset_split"],
        "record_id": row["record_id"],
        "image_name": row["image_name"],
        "probe": row["probe"],
        "oof_fold": row["oof_fold"],
        "case_type": case_type,
        "top1_error_px": f"{errors[0]:.3f}",
        "top10_min_error_px": f"{min(errors):.3f}",
        "top1_box48": "1" if boxes[0] else "0",
        "top10_box48": "1" if any(boxes) else "0",
        "topk_points": row["topk_points"],
    }


def draw_box(draw: ImageDraw.ImageDraw, x: float, y: float, color: str, width: int) -> None:
    draw.rectangle((x - 24, y - 24, x + 24, y + 24), outline=color, width=width)


def save_debug_overlays(
    case_rows: list[dict[str, str]],
    prediction_rows: dict[str, dict[str, str]],
    output_dir: Path,
    limit: int,
) -> None:
    if limit <= 0:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    rank_hard = [row for row in case_rows if row["case_type"] == "rank_hard"]
    ordered = sorted(rank_hard, key=lambda row: float(row["top1_error_px"]), reverse=True)[:limit]
    for case in ordered:
        row = prediction_rows[case["image_name"]]
        image = Image.open(row["vision_path"]).convert("RGB")
        draw = ImageDraw.Draw(image)
        draw_box(draw, float(row["target_x"]), float(row["target_y"]), "lime", 4)
        for rank, (x, y, _) in enumerate(parse_points(row["topk_points"])):
            color = "orange" if rank == 0 else "cyan"
            draw_box(draw, x, y, color, 2)
        draw.text((8, 8), "GT green | C2 Top1 orange | Top2-10 cyan", fill="white", stroke_width=2, stroke_fill="black")
        filename = f"{case['record_id']}_probe{int(case['probe']):03d}_{case['image_name']}"
        image.save(output_dir / filename)


def build(config_path: str, section: str, force: bool) -> dict:
    config = load_config(config_path)
    cfg = config[section]
    source_section = str(cfg["contact_model_section"])
    source_cfg = config[source_section]
    source_rows = read_csv_rows(project_path(source_cfg["samples_csv"]))
    train_rows = [row for row in source_rows if row["dataset_split"] == "train"]
    validation_rows = [row for row in source_rows if row["dataset_split"] == "val"]
    train_records = sorted({row["record_id"] for row in train_rows})
    if not train_rows or not validation_rows:
        raise ValueError("OOF requires non-empty development train and validation splits")
    folds = record_folds(train_records, int(cfg.get("folds", 3)), int(cfg.get("seed", 42)))
    root = project_path(cfg["work_dir"])
    root.mkdir(parents=True, exist_ok=True)
    combined = []
    fold_summaries = []
    membership_rows = []

    for fold_index, held_records in enumerate(folds):
        fold_dir = root / f"fold_{fold_index}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        samples_path = fold_dir / "samples.csv"
        predictions_path = fold_dir / "predictions.csv"
        held_set = set(held_records)
        fold_rows = []
        for original in train_rows + validation_rows:
            row = dict(original)
            if original["dataset_split"] == "train" and row["record_id"] in held_set:
                row["dataset_split"] = "test"
            fold_rows.append(row)
        fold_training_records = sorted({row["record_id"] for row in fold_rows if row["dataset_split"] == "train"})
        if held_set.intersection(fold_training_records):
            raise RuntimeError(f"OOF leakage in fold {fold_index}: held records appear in training rows")
        membership_rows.extend({"oof_fold": str(fold_index), "record_id": record, "role": "held"} for record in held_records)
        membership_rows.extend({"oof_fold": str(fold_index), "record_id": record, "role": "train"} for record in fold_training_records)
        membership_rows.extend({"oof_fold": str(fold_index), "record_id": record, "role": "validation"} for record in sorted({row["record_id"] for row in validation_rows}))
        write_csv_rows(samples_path, fold_rows, list(fold_rows[0]))

        fold_section = f"{section}_fold_{fold_index}"
        fold_cfg = copy.deepcopy(source_cfg)
        fold_cfg.update({
            "samples_csv": str(samples_path),
            "checkpoint_dir": str(fold_dir / "checkpoints"),
            "metrics_json": str(fold_dir / "metrics.json"),
            "predictions_csv": str(predictions_path),
            "retrieval_json": str(fold_dir / "retrieval.json"),
            "retrieval_csv": str(fold_dir / "retrieval.csv"),
            "debug_dir": str(fold_dir / "debug"),
            "retrieval_debug_dir": str(fold_dir / "retrieval_debug"),
            "topk": int(cfg.get("topk", 10)),
            "debug_samples": 0,
            "epochs": int(cfg.get("epochs", 120)),
            "seed": int(cfg.get("seed", 42)) + fold_index,
        })
        fold_config = copy.deepcopy(config)
        fold_config[fold_section] = fold_cfg
        fold_config_path = fold_dir / "config.yaml"
        fold_config_path.write_text(yaml.safe_dump(fold_config, sort_keys=False), encoding="utf-8")
        if force or not predictions_path.exists():
            train_contact_region(str(fold_config_path), section=fold_section)
        predictions = read_csv_rows(predictions_path)
        held_predictions = [row for row in predictions if row["dataset_split"] == "test" and row["record_id"] in held_set]
        for row in held_predictions:
            row["dataset_split"] = "train"
            row["oof_fold"] = str(fold_index)
        combined.extend(held_predictions)
        fold_summaries.append({
            "fold": fold_index, "held_records": held_records,
            "held_samples": len(held_predictions), "training_records": fold_training_records,
            "validation_records": sorted({row["record_id"] for row in validation_rows}),
            "predictions_csv": str(predictions_path),
        })

    output_csv = project_path(cfg["output_csv"])
    fields = list(combined[0]) if combined else []
    write_csv_rows(output_csv, combined, fields)
    case_rows = [classify_prediction(row) for row in combined]
    write_csv_rows(project_path(cfg["cases_csv"]), case_rows, CASE_FIELDS)
    write_csv_rows(project_path(cfg["fold_membership_csv"]), membership_rows, FOLD_MEMBERSHIP_FIELDS)
    by_probe: dict[str, Counter] = defaultdict(Counter)
    for row in case_rows:
        by_probe[row["probe"]][row["case_type"]] += 1
    case_counts = Counter(row["case_type"] for row in case_rows)
    prediction_rows = {row["image_name"]: row for row in combined}
    save_debug_overlays(
        case_rows, prediction_rows, project_path(cfg["debug_dir"]), int(cfg.get("debug_samples", 12)),
    )
    summary = {
        "policy": "Record-level OOF on development train only. Held records are excluded from contact-model training; development validation selects each fold checkpoint; final holdout is excluded.",
        "folds": fold_summaries, "records": len(train_records), "samples": len(combined),
        "source_split_counts": {"train": len(train_rows), "val": len(validation_rows), "final_excluded": len(source_rows) - len(train_rows) - len(validation_rows)},
        "case_counts": dict(case_counts),
        "case_counts_by_probe": {probe: dict(counts) for probe, counts in sorted(by_probe.items(), key=lambda item: int(item[0]))},
        "output_csv": str(output_csv), "cases_csv": str(project_path(cfg["cases_csv"])),
        "fold_membership_csv": str(project_path(cfg["fold_membership_csv"])),
    }
    write_json(project_path(cfg["metrics_json"]), summary)
    print(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build record-level out-of-fold C2 Top-10 proposals for ranker training.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--section", default="proposal_ranker_oof_masked_16")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    build(args.config, args.section, args.force)


if __name__ == "__main__":
    main()
