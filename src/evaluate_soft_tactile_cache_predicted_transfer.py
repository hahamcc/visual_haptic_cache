from __future__ import annotations

import argparse
import math

import numpy as np
import torch

from .build_cache_retrieval import crop_contact_patch, motion_geometry_feature, standardize, visual_patch_feature_from_patch
from .config import load_config, project_path
from .evaluate_oracle_tactile_retrieval import tactile_difference, tactile_embedding, tactile_metrics
from .train_soft_tactile_cache_ranker import CandidateGroups, SoftTactileRanker, predict, ranks
from .utils import read_csv_rows, write_csv_rows, write_json


METRICS = [
    "tactile_diff_mae", "tactile_ssim", "tactile_mask_iou", "tactile_area_delta",
    "tactile_centroid_distance", "tactile_embedding_distance",
]
QUERY_FIELDS = [
    "query_record_id", "query_image_name", "query_probe", "pred_x", "pred_y", "contact_category",
    "current_cache_record_id", "soft_ranker_cache_record_id", "tactile_oracle_cache_record_id",
    "current_rank_of_tactile_best", "soft_ranker_rank_of_tactile_best",
    *[f"{prefix}_{metric}" for prefix in ("current", "soft_ranker", "oracle") for metric in METRICS],
]


def record_number(record_id: str) -> int:
    try:
        return int(record_id.rsplit("_", 1)[-1])
    except ValueError as exc:
        raise ValueError(f"Unexpected record ID: {record_id}") from exc


def parse_topk(value: str) -> list[tuple[float, float]]:
    return [(float(item.split(",")[0]), float(item.split(",")[1])) for item in value.split(";") if item]


def category(prediction: dict[str, str], row: dict[str, str]) -> str:
    target_x, target_y = float(row["target_tip_x"]), float(row["target_tip_y"])
    points = parse_topk(prediction["topk_points"])
    if abs(points[0][0] - target_x) <= 24.0 and abs(points[0][1] - target_y) <= 24.0:
        return "top1_box48"
    if any(abs(x - target_x) <= 24.0 and abs(y - target_y) <= 24.0 for x, y in points):
        return "top10_rank_hard"
    return "top10_proposal_miss"


def summarize(rows: list[dict[str, str]], prefix: str) -> dict[str, float | int | None]:
    result: dict[str, float | int | None] = {"queries": len(rows)}
    for metric in METRICS:
        values = np.asarray([float(row[f"{prefix}_{metric}"]) for row in rows], dtype=np.float32)
        result[f"mean_{metric}"] = float(values.mean()) if len(values) else None
        result[f"median_{metric}"] = float(np.median(values)) if len(values) else None
    ranks = (
        np.ones(len(rows), dtype=np.float32)
        if prefix == "oracle"
        else np.asarray([int(row[f"{prefix}_rank_of_tactile_best"]) for row in rows], dtype=np.float32)
    )
    result["tactile_best_top1_rate"] = float(np.mean(ranks == 1)) if len(ranks) else None
    result["tactile_best_top5_rate"] = float(np.mean(ranks <= 5)) if len(ranks) else None
    result["median_tactile_best_rank"] = float(np.median(ranks)) if len(ranks) else None
    return result


def evaluate(config_path: str, section: str) -> dict:
    cfg = load_config(config_path)[section]
    rows = read_csv_rows(project_path(cfg["samples_csv"]))
    predictions = read_csv_rows(project_path(cfg["predictions_csv"]))
    final_min_record = int(cfg.get("final_holdout_min_record", 950))
    forbidden = sorted({row["record_id"] for row in rows if record_number(row["record_id"]) >= final_min_record})
    if forbidden:
        raise RuntimeError(f"Refusing to access final-holdout records: {forbidden[:5]}")
    cache_rows = [row for row in rows if row["dataset_split"] == "train"]
    val_by_name = {row["image_name"]: row for row in rows if row["dataset_split"] == "val"}
    val_predictions = [row for row in predictions if row["dataset_split"] == "val" and row["image_name"] in val_by_name]
    if len(val_predictions) != len(val_by_name):
        raise RuntimeError(f"Need one validation prediction per query, got {len(val_predictions)} for {len(val_by_name)} rows.")
    overlap = {row["record_id"] for row in cache_rows} & {row["record_id"] for row in val_by_name.values()}
    if overlap:
        raise RuntimeError(f"Train cache and validation query records overlap: {sorted(overlap)[:5]}")
    crop_size, tactile_size = int(cfg.get("cache_crop_size", 48)), int(cfg.get("tactile_size", 96))
    filter_k = min(int(cfg.get("geometry_filter_k", 32)), len(cache_rows))
    raw_cache_geometry = np.stack([motion_geometry_feature(row, float(row["target_tip_x"]), float(row["target_tip_y"])) for row in cache_rows])
    cache_geometry, geometry_mean, geometry_std = standardize(raw_cache_geometry, raw_cache_geometry)
    cache_geometry = cache_geometry.astype(np.float32)
    cache_patches = np.stack([crop_contact_patch(row["vision_path"], float(row["target_tip_x"]), float(row["target_tip_y"]), crop_size) for row in cache_rows]).astype(np.float32)
    cache_hand_raw = np.stack([visual_patch_feature_from_patch(patch) for patch in cache_patches])
    cache_hand, hand_mean, hand_std = standardize(cache_hand_raw, cache_hand_raw)
    diff_cache: dict[str, np.ndarray] = {}
    record_tactile: dict[str, np.ndarray] = {}
    for row in cache_rows:
        if row["record_id"] not in record_tactile:
            record_tactile[row["record_id"]] = tactile_difference(row["touch_path"], diff_cache, tactile_size)
    cache_tactile = np.stack([tactile_embedding(record_tactile[row["record_id"]]) for row in cache_rows]).astype(np.float32)

    query_rows, query_patches, query_geometry, query_hand, query_tactile, categories = [], [], [], [], [], []
    for prediction in val_predictions:
        row = val_by_name[prediction["image_name"]]
        x, y = float(prediction["pred_x"]), float(prediction["pred_y"])
        patch = crop_contact_patch(row["vision_path"], x, y, crop_size)
        query_rows.append(row)
        query_patches.append(patch)
        query_geometry.append((motion_geometry_feature(row, x, y) - geometry_mean) / geometry_std)
        query_hand.append((visual_patch_feature_from_patch(patch) - hand_mean) / hand_std)
        query_tactile.append(tactile_difference(row["touch_path"], diff_cache, tactile_size))
        categories.append(category(prediction, row))
    query_patches = np.stack(query_patches).astype(np.float32)
    query_geometry = np.stack(query_geometry).astype(np.float32)
    query_hand = np.stack(query_hand).astype(np.float32)
    query_tactile_images = np.stack(query_tactile).astype(np.float32)
    query_tactile = np.stack([tactile_embedding(image) for image in query_tactile_images]).astype(np.float32)
    diff_cache.clear()

    candidates, targets, current_scores = [], [], []
    for index in range(len(query_rows)):
        geometry_distances = np.linalg.norm(cache_geometry - query_geometry[index][None], axis=1)
        shortlist = np.argpartition(geometry_distances, filter_k - 1)[:filter_k]
        shortlist = shortlist[np.argsort(geometry_distances[shortlist], kind="stable")]
        visual_distances = np.linalg.norm(cache_hand[shortlist] - query_hand[index][None], axis=1)
        current = geometry_distances[shortlist] / math.sqrt(cache_geometry.shape[1]) + visual_distances / math.sqrt(cache_hand.shape[1])
        candidates.append(shortlist.astype(np.int32))
        targets.append(np.linalg.norm(cache_tactile[shortlist] - query_tactile[index][None], axis=1).astype(np.float32))
        current_scores.append(current.astype(np.float32))
    groups = CandidateGroups(np.stack(candidates), np.stack(targets), np.stack(current_scores))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = project_path(cfg["checkpoint"])
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = SoftTactileRanker(cache_geometry.shape[1], float(cfg.get("dropout", 0.1))).to(device)
    model.load_state_dict(checkpoint["model_state"])
    scores = predict(model, groups, query_patches, query_geometry, cache_patches, cache_geometry, device, int(cfg.get("batch_size", 16)))

    metric_cache: dict[str, np.ndarray] = {}
    output_rows: list[dict[str, str]] = []
    for index, query in enumerate(query_rows):
        current = int(np.argmin(groups.current_scores[index]))
        ranker = int(np.argmin(scores[index]))
        oracle = int(np.argmin(groups.targets[index]))
        selections = {"current": current, "soft_ranker": ranker, "oracle": oracle}
        values = {}
        caches = {}
        for name, local_index in selections.items():
            cache = cache_rows[int(groups.candidates[index, local_index])]
            caches[name] = cache
            values[name] = tactile_metrics(query_tactile_images[index], tactile_difference(cache["touch_path"], metric_cache, tactile_size), float(cfg.get("tactile_mask_threshold", 0.04)))
        output_rows.append({
            "query_record_id": query["record_id"], "query_image_name": query["image_name"], "query_probe": query["probe"],
            "pred_x": f"{float(val_predictions[index]['pred_x']):.3f}", "pred_y": f"{float(val_predictions[index]['pred_y']):.3f}", "contact_category": categories[index],
            "current_cache_record_id": caches["current"]["record_id"], "soft_ranker_cache_record_id": caches["soft_ranker"]["record_id"], "tactile_oracle_cache_record_id": caches["oracle"]["record_id"],
            "current_rank_of_tactile_best": str(int(ranks(groups.current_scores[index])[oracle])), "soft_ranker_rank_of_tactile_best": str(int(ranks(scores[index])[oracle])),
            **{f"{name}_{metric}": f"{metric_values[metric]:.6f}" for name, metric_values in values.items() for metric in METRICS},
        })
    summary = {
        "mode": "validation_only_frozen_soft_ranker_predicted_top1_transfer", "device": str(device), "cache_size": len(cache_rows),
        "validation_queries": len(query_rows), "geometry_filter_k": filter_k,
        "contact_category_counts": {name: sum(row["contact_category"] == name for row in output_rows) for name in ("top1_box48", "top10_rank_hard", "top10_proposal_miss")},
        "overall": {name: summarize(output_rows, name) for name in ("current", "soft_ranker", "oracle")},
        "far_probe75_100": {name: summarize([row for row in output_rows if int(row["query_probe"]) >= 75], name) for name in ("current", "soft_ranker", "oracle")},
        "by_contact_category": {category_name: {name: summarize([row for row in output_rows if row["contact_category"] == category_name], name) for name in ("current", "soft_ranker", "oracle")} for category_name in ("top1_box48", "top10_rank_hard", "top10_proposal_miss")},
        "final_holdout_min_record": final_min_record, "checkpoint": str(checkpoint_path),
        "transfer_note": "Frozen ranker uses C2 Top-1 predicted contact crops and predicted contact geometry. Ground truth is used only for offline tactile metrics and category reporting.",
    }
    write_csv_rows(project_path(cfg["output_csv"]), output_rows, QUERY_FIELDS)
    write_json(project_path(cfg["metrics_json"]), summary)
    print(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate frozen soft tactile cache ranking with C2 predicted Top-1 contact boxes.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--section", default="soft_tactile_cache_predicted_transfer_phase35_v3")
    args = parser.parse_args()
    evaluate(args.config, args.section)


if __name__ == "__main__":
    main()
