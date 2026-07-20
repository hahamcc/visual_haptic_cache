from __future__ import annotations

import argparse
import math

import numpy as np
import torch

from .build_cache_retrieval import crop_contact_patch, motion_geometry_feature, standardize, visual_patch_feature_from_patch
from .config import load_config, project_path
from .evaluate_oracle_tactile_retrieval import tactile_difference, tactile_embedding, tactile_metrics
from .train_soft_tactile_cache_ranker import CandidateGroups, SoftTactileRanker, predict
from .utils import read_csv_rows, write_csv_rows, write_json


METRICS = [
    "tactile_diff_mae", "tactile_ssim", "tactile_mask_iou", "tactile_area_delta",
    "tactile_centroid_distance", "tactile_embedding_distance",
]
CANDIDATE_FIELDS = [
    "query_record_id", "query_image_name", "query_probe", "candidate_rank", "candidate_x", "candidate_y",
    "heatmap_score", "heatmap_ratio", "candidate_box48_hit", "ranker_cache_score", "ranker_score_normalized",
    "retrieved_cache_record_id", "retrieved_cache_image_name",
]
GRID_FIELDS = [
    "heatmap_weight", "selected_candidate_box48_rate", "selected_non_top1_rate", "mean_tactile_diff_mae",
    "mean_tactile_ssim", "mean_tactile_mask_iou", "mean_tactile_embedding_distance", "far_mean_tactile_diff_mae",
    "far_mean_tactile_ssim", "far_mean_tactile_mask_iou", "eligible",
]
QUERY_FIELDS = [
    "mode", "query_record_id", "query_image_name", "query_probe", "selected_candidate_rank", "selected_candidate_x",
    "selected_candidate_y", "selected_candidate_box48_hit", "retrieved_cache_record_id", "retrieved_cache_image_name",
    *METRICS,
]


def record_number(record_id: str) -> int:
    try:
        return int(record_id.rsplit("_", 1)[-1])
    except ValueError as exc:
        raise ValueError(f"Unexpected record ID: {record_id}") from exc


def parse_points(value: str, topk: int) -> list[tuple[float, float, float]]:
    result = []
    for item in value.split(";")[:topk]:
        x, y, score = item.split(",")
        result.append((float(x), float(y), float(score)))
    if not result:
        raise ValueError("Prediction contains no Top-K points.")
    return result


def box48_hit(x: float, y: float, row: dict[str, str]) -> bool:
    return abs(x - float(row["target_tip_x"])) <= 24.0 and abs(y - float(row["target_tip_y"])) <= 24.0


def summarize(rows: list[dict[str, str]]) -> dict[str, float | int | None]:
    result: dict[str, float | int | None] = {"queries": len(rows)}
    for metric in METRICS:
        values = np.asarray([float(row[metric]) for row in rows], dtype=np.float32)
        result[f"mean_{metric}"] = float(values.mean()) if len(values) else None
        result[f"median_{metric}"] = float(np.median(values)) if len(values) else None
    result["selected_candidate_box48_rate"] = float(np.mean([row["selected_candidate_box48_hit"] == "1" for row in rows])) if rows else None
    result["selected_non_top1_rate"] = float(np.mean([row["selected_candidate_rank"] != "1" for row in rows])) if rows else None
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
    topk = int(cfg.get("topk", 10))
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

    # Expand each validation query into its C2 Top-K candidate crops. GT remains evaluation-only.
    expanded_rows, query_patches, query_geometry, query_hand, query_tactile, meta = [], [], [], [], [], []
    for query_index, prediction in enumerate(val_predictions):
        row = val_by_name[prediction["image_name"]]
        top_points = parse_points(prediction["topk_points"], topk)
        top_score = max(top_points[0][2], 1e-8)
        tactile_image = tactile_difference(row["touch_path"], diff_cache, tactile_size)
        for rank, (x, y, heat_score) in enumerate(top_points, start=1):
            patch = crop_contact_patch(row["vision_path"], x, y, crop_size)
            expanded_rows.append(row)
            query_patches.append(patch)
            query_geometry.append((motion_geometry_feature(row, x, y) - geometry_mean) / geometry_std)
            query_hand.append((visual_patch_feature_from_patch(patch) - hand_mean) / hand_std)
            query_tactile.append(tactile_image)
            meta.append({"query_index": query_index, "rank": rank, "x": x, "y": y, "heat_score": heat_score, "heat_ratio": heat_score / top_score, "box48": box48_hit(x, y, row)})
    query_patches = np.stack(query_patches).astype(np.float32)
    query_geometry = np.stack(query_geometry).astype(np.float32)
    query_hand = np.stack(query_hand).astype(np.float32)
    query_tactile_images = np.stack(query_tactile).astype(np.float32)
    query_tactile = np.stack([tactile_embedding(image) for image in query_tactile_images]).astype(np.float32)
    diff_cache.clear()
    candidates, targets, current_scores = [], [], []
    for index in range(len(expanded_rows)):
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
    ranker_local = np.argmin(scores, axis=1)
    ranker_cache_score = scores[np.arange(len(scores)), ranker_local]

    per_query: list[list[int]] = [[] for _ in val_predictions]
    for expanded_index, item in enumerate(meta):
        per_query[item["query_index"]].append(expanded_index)
    candidate_rows: list[dict[str, str]] = []
    normalized_ranker = np.zeros(len(meta), dtype=np.float32)
    for indices in per_query:
        values = ranker_cache_score[indices]
        scale = max(float(values.max() - values.min()), 1e-6)
        normalized_ranker[indices] = (values - values.min()) / scale
    for index, item in enumerate(meta):
        cache = cache_rows[int(groups.candidates[index, ranker_local[index]])]
        candidate_rows.append({
            "query_record_id": expanded_rows[index]["record_id"], "query_image_name": expanded_rows[index]["image_name"], "query_probe": expanded_rows[index]["probe"],
            "candidate_rank": str(item["rank"]), "candidate_x": f"{item['x']:.3f}", "candidate_y": f"{item['y']:.3f}",
            "heatmap_score": f"{item['heat_score']:.6f}", "heatmap_ratio": f"{item['heat_ratio']:.6f}", "candidate_box48_hit": str(int(item["box48"])),
            "ranker_cache_score": f"{ranker_cache_score[index]:.6f}", "ranker_score_normalized": f"{normalized_ranker[index]:.6f}",
            "retrieved_cache_record_id": cache["record_id"], "retrieved_cache_image_name": cache["image_name"],
        })

    metric_cache: dict[str, np.ndarray] = {}
    selected_metric_cache: dict[int, dict[str, float]] = {}
    def metrics_for(expanded_index: int) -> dict[str, float]:
        if expanded_index not in selected_metric_cache:
            cache = cache_rows[int(groups.candidates[expanded_index, ranker_local[expanded_index]])]
            selected_metric_cache[expanded_index] = tactile_metrics(
                query_tactile_images[expanded_index], tactile_difference(cache["touch_path"], metric_cache, tactile_size), float(cfg.get("tactile_mask_threshold", 0.04)),
            )
        return selected_metric_cache[expanded_index]

    def output_for_choices(mode: str, choices: list[int]) -> list[dict[str, str]]:
        output = []
        for query_index, expanded_index in enumerate(choices):
            item, row = meta[expanded_index], expanded_rows[expanded_index]
            cache = cache_rows[int(groups.candidates[expanded_index, ranker_local[expanded_index]])]
            metrics = metrics_for(expanded_index)
            output.append({
                "mode": mode, "query_record_id": row["record_id"], "query_image_name": row["image_name"], "query_probe": row["probe"],
                "selected_candidate_rank": str(item["rank"]), "selected_candidate_x": f"{item['x']:.3f}", "selected_candidate_y": f"{item['y']:.3f}",
                "selected_candidate_box48_hit": str(int(item["box48"])), "retrieved_cache_record_id": cache["record_id"], "retrieved_cache_image_name": cache["image_name"],
                **{name: f"{value:.6f}" for name, value in metrics.items()},
            })
        return output

    c2_top1_choices = [indices[0] for indices in per_query]
    ranker_only_choices = [min(indices, key=lambda index: ranker_cache_score[index]) for indices in per_query]
    baseline_rows = output_for_choices("c2_top1_cache", c2_top1_choices)
    ranker_only_rows = output_for_choices("topk_ranker_only", ranker_only_choices)
    baseline = summarize(baseline_rows)
    baseline_far = summarize([row for row in baseline_rows if int(row["query_probe"]) >= 75])
    weights = [float(value) for value in cfg.get("heatmap_weights", [0.0, 0.05, 0.1, 0.25, 0.5, 1.0])]
    grid_rows, experiments = [], []
    for weight in weights:
        joint_choices = [min(indices, key=lambda index: normalized_ranker[index] + weight * (1.0 - meta[index]["heat_ratio"])) for indices in per_query]
        rows_for_weight = output_for_choices("topk_joint", joint_choices)
        overall, far = summarize(rows_for_weight), summarize([row for row in rows_for_weight if int(row["query_probe"]) >= 75])
        eligible = (
            overall["selected_candidate_box48_rate"] >= baseline["selected_candidate_box48_rate"]
            and overall["mean_tactile_diff_mae"] <= baseline["mean_tactile_diff_mae"]
            and overall["mean_tactile_ssim"] >= baseline["mean_tactile_ssim"]
            and overall["mean_tactile_mask_iou"] >= baseline["mean_tactile_mask_iou"]
            and far["mean_tactile_diff_mae"] <= baseline_far["mean_tactile_diff_mae"]
            and far["mean_tactile_mask_iou"] >= baseline_far["mean_tactile_mask_iou"]
        )
        grid_rows.append({
            "heatmap_weight": f"{weight:.3f}", "selected_candidate_box48_rate": f"{overall['selected_candidate_box48_rate']:.6f}", "selected_non_top1_rate": f"{overall['selected_non_top1_rate']:.6f}",
            "mean_tactile_diff_mae": f"{overall['mean_tactile_diff_mae']:.6f}", "mean_tactile_ssim": f"{overall['mean_tactile_ssim']:.6f}", "mean_tactile_mask_iou": f"{overall['mean_tactile_mask_iou']:.6f}",
            "mean_tactile_embedding_distance": f"{overall['mean_tactile_embedding_distance']:.6f}", "far_mean_tactile_diff_mae": f"{far['mean_tactile_diff_mae']:.6f}", "far_mean_tactile_ssim": f"{far['mean_tactile_ssim']:.6f}",
            "far_mean_tactile_mask_iou": f"{far['mean_tactile_mask_iou']:.6f}", "eligible": str(int(eligible)),
        })
        experiments.append((overall["mean_tactile_ssim"], overall["selected_candidate_box48_rate"], rows_for_weight, eligible, len(grid_rows) - 1))
    eligible = [item for item in experiments if item[3]]
    selected = max(eligible, key=lambda item: (item[0], item[1])) if eligible else None
    if selected is None:
        selected_rows, selected_grid = baseline_rows, None
    else:
        selected_rows, selected_grid = selected[2], grid_rows[selected[4]]
    selected_rows = [dict(row, mode="topk_joint_selected") for row in selected_rows]
    output_rows = baseline_rows + ranker_only_rows + selected_rows
    summary = {
        "mode": "validation_only_topk_contact_to_soft_tactile_cache", "device": str(device), "cache_size": len(cache_rows), "validation_queries": len(val_predictions),
        "topk": topk, "geometry_filter_k": filter_k, "selection_policy": "maximize validation SSIM subject to non-worse Box48, overall MAE/SSIM/IoU, and far MAE/IoU versus C2 Top-1 cache",
        "baseline_c2_top1_cache": baseline, "topk_ranker_only": summarize(ranker_only_rows), "selected_joint_grid": selected_grid,
        "selected_topk_joint": summarize(selected_rows), "far_selected_topk_joint": summarize([row for row in selected_rows if int(row["query_probe"]) >= 75]),
        "final_holdout_min_record": final_min_record, "checkpoint": str(checkpoint_path),
        "note": "GT contact coordinates are used only for offline Box48 and tactile metrics; Top-K scoring uses C2 proposals, heatmap scores, visual crops, geometry, and frozen ranker scores.",
    }
    write_csv_rows(project_path(cfg["output_csv"]), output_rows, QUERY_FIELDS)
    write_csv_rows(project_path(cfg["candidates_csv"]), candidate_rows, CANDIDATE_FIELDS)
    write_csv_rows(project_path(cfg["grid_csv"]), grid_rows, GRID_FIELDS)
    write_json(project_path(cfg["metrics_json"]), summary)
    print(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Top-K contact proposals with frozen soft tactile cache retrieval.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--section", default="topk_soft_tactile_cache_phase35_v3")
    args = parser.parse_args()
    evaluate(args.config, args.section)


if __name__ == "__main__":
    main()
