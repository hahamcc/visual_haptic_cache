from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from .build_cache_retrieval import motion_geometry_feature, standardize, visual_patch_feature
from .config import load_config, project_path
from .evaluate_oracle_tactile_retrieval import tactile_difference, tactile_embedding, tactile_metrics
from .utils import read_csv_rows, write_csv_rows, write_json


CANDIDATE_FIELDS = [
    "query_record_id", "query_image_name", "query_probe", "cache_record_id", "cache_image_name", "cache_probe",
    "geometry_rank", "current_key_rank", "tactile_target_rank", "geometry_distance", "visual_distance",
    "current_key_distance", "tactile_embedding_distance", "is_current_selection", "is_tactile_best",
]
QUERY_FIELDS = [
    "query_record_id", "query_image_name", "query_probe", "tactile_best_rank_under_current_key",
    "tactile_best_in_current_top5", "spearman_current_key_vs_tactile", "current_cache_record_id",
    "tactile_best_cache_record_id", "current_tactile_diff_mae", "best_tactile_diff_mae",
    "current_tactile_ssim", "best_tactile_ssim", "current_tactile_mask_iou", "best_tactile_mask_iou",
    "current_tactile_area_delta", "best_tactile_area_delta", "current_tactile_centroid_distance", "best_tactile_centroid_distance",
    "current_tactile_embedding_distance", "best_tactile_embedding_distance",
]


def record_number(record_id: str) -> int:
    try:
        return int(record_id.rsplit("_", 1)[-1])
    except ValueError as exc:
        raise ValueError(f"Unexpected record ID: {record_id}") from exc


def rank_positions(values: np.ndarray) -> np.ndarray:
    positions = np.empty(len(values), dtype=np.int32)
    positions[np.argsort(values, kind="stable")] = np.arange(1, len(values) + 1)
    return positions


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.std(left) < 1e-9 or np.std(right) < 1e-9:
        return 0.0
    left_rank = rank_positions(left).astype(np.float32)
    right_rank = rank_positions(right).astype(np.float32)
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def metric_summary(rows: list[dict[str, str]], prefix: str) -> dict[str, float | int | None]:
    result: dict[str, float | int | None] = {"queries": len(rows)}
    if not rows:
        return result
    for metric in ("tactile_diff_mae", "tactile_ssim", "tactile_mask_iou", "tactile_embedding_distance"):
        values = np.asarray([float(row[f"{prefix}_{metric}"]) for row in rows], dtype=np.float32)
        result[f"mean_{metric}"] = float(values.mean())
        result[f"median_{metric}"] = float(np.median(values))
    return result


def audit(config_path: str, section: str) -> dict:
    cfg = load_config(config_path)[section]
    rows = read_csv_rows(project_path(cfg["samples_csv"]))
    cache_split, query_split = str(cfg.get("cache_split", "train")), str(cfg.get("query_split", "val"))
    cache_rows = [row for row in rows if row["dataset_split"] == cache_split]
    query_rows = [row for row in rows if row["dataset_split"] == query_split]
    final_min_record = int(cfg.get("final_holdout_min_record", 950))
    forbidden = sorted({row["record_id"] for row in rows if record_number(row["record_id"]) >= final_min_record})
    if forbidden:
        raise RuntimeError(f"Refusing to access final-holdout records: {forbidden[:5]}")
    if not cache_rows or not query_rows:
        raise RuntimeError("Need non-empty cache and validation query rows.")

    crop_size = int(cfg.get("cache_crop_size", 48))
    tactile_size = int(cfg.get("tactile_size", 96))
    tactile_threshold = float(cfg.get("tactile_mask_threshold", 0.04))
    filter_k = min(int(cfg.get("geometry_filter_k", 32)), len(cache_rows))
    geometry_weight, visual_weight = float(cfg.get("geometry_weight", 1.0)), float(cfg.get("visual_weight", 1.0))
    cache_geometry = np.stack([motion_geometry_feature(row, float(row["target_tip_x"]), float(row["target_tip_y"])) for row in cache_rows])
    cache_geometry_z, geometry_mean, geometry_std = standardize(cache_geometry, cache_geometry)
    cache_visual = np.stack([visual_patch_feature(row["vision_path"], float(row["target_tip_x"]), float(row["target_tip_y"]), crop_size) for row in cache_rows])
    cache_visual_z, visual_mean, visual_std = standardize(cache_visual, cache_visual)

    # Tactile embeddings are supervision/diagnostic targets only, never part of a cache key.
    diff_cache: dict[str, np.ndarray] = {}
    cache_embeddings = np.stack([tactile_embedding(tactile_difference(row["touch_path"], diff_cache, tactile_size)) for row in cache_rows])
    diff_cache.clear()
    candidate_rows: list[dict[str, str]] = []
    query_results: list[dict[str, str]] = []
    for query in query_rows:
        x, y = float(query["target_tip_x"]), float(query["target_tip_y"])
        query_geometry = (motion_geometry_feature(query, x, y) - geometry_mean) / geometry_std
        geometry_distances = np.linalg.norm(cache_geometry_z - query_geometry[None], axis=1)
        geometry_indices = np.argpartition(geometry_distances, filter_k - 1)[:filter_k]
        geometry_indices = geometry_indices[np.argsort(geometry_distances[geometry_indices], kind="stable")]
        query_visual = (visual_patch_feature(query["vision_path"], x, y, crop_size) - visual_mean) / visual_std
        visual_distances = np.linalg.norm(cache_visual_z[geometry_indices] - query_visual[None], axis=1)
        current_distances = (
            geometry_weight * geometry_distances[geometry_indices] / math.sqrt(cache_geometry_z.shape[1])
            + visual_weight * visual_distances / math.sqrt(cache_visual_z.shape[1])
        )
        query_diff = tactile_difference(query["touch_path"], diff_cache, tactile_size)
        tactile_distances = np.linalg.norm(cache_embeddings[geometry_indices] - tactile_embedding(query_diff)[None], axis=1)
        current_ranks, tactile_ranks = rank_positions(current_distances), rank_positions(tactile_distances)
        current_local = int(np.argmin(current_distances))
        tactile_best_local = int(np.argmin(tactile_distances))
        for local_index, cache_index in enumerate(geometry_indices):
            cache = cache_rows[int(cache_index)]
            candidate_rows.append({
                "query_record_id": query["record_id"], "query_image_name": query["image_name"], "query_probe": query["probe"],
                "cache_record_id": cache["record_id"], "cache_image_name": cache["image_name"], "cache_probe": cache["probe"],
                "geometry_rank": str(local_index + 1), "current_key_rank": str(int(current_ranks[local_index])),
                "tactile_target_rank": str(int(tactile_ranks[local_index])), "geometry_distance": f"{geometry_distances[cache_index]:.6f}",
                "visual_distance": f"{visual_distances[local_index]:.6f}", "current_key_distance": f"{current_distances[local_index]:.6f}",
                "tactile_embedding_distance": f"{tactile_distances[local_index]:.6f}", "is_current_selection": str(int(local_index == current_local)),
                "is_tactile_best": str(int(local_index == tactile_best_local)),
            })
        current_cache = cache_rows[int(geometry_indices[current_local])]
        best_cache = cache_rows[int(geometry_indices[tactile_best_local])]
        current_metrics = tactile_metrics(query_diff, tactile_difference(current_cache["touch_path"], diff_cache, tactile_size), tactile_threshold)
        best_metrics = tactile_metrics(query_diff, tactile_difference(best_cache["touch_path"], diff_cache, tactile_size), tactile_threshold)
        query_results.append({
            "query_record_id": query["record_id"], "query_image_name": query["image_name"], "query_probe": query["probe"],
            "tactile_best_rank_under_current_key": str(int(current_ranks[tactile_best_local])),
            "tactile_best_in_current_top5": str(int(current_ranks[tactile_best_local] <= 5)),
            "spearman_current_key_vs_tactile": f"{spearman(current_distances, tactile_distances):.6f}",
            "current_cache_record_id": current_cache["record_id"], "tactile_best_cache_record_id": best_cache["record_id"],
            **{f"current_{name}": f"{value:.6f}" for name, value in current_metrics.items()},
            **{f"best_{name}": f"{value:.6f}" for name, value in best_metrics.items()},
        })

    def summarize_scope(scope_rows: list[dict[str, str]]) -> dict:
        ranks = np.asarray([int(row["tactile_best_rank_under_current_key"]) for row in scope_rows], dtype=np.float32)
        correlations = np.asarray([float(row["spearman_current_key_vs_tactile"]) for row in scope_rows], dtype=np.float32)
        return {
            "queries": len(scope_rows),
            "tactile_best_current_top1_rate": float(np.mean(ranks == 1)) if len(ranks) else None,
            "tactile_best_current_top5_rate": float(np.mean(ranks <= 5)) if len(ranks) else None,
            "median_tactile_best_rank_under_current_key": float(np.median(ranks)) if len(ranks) else None,
            "mean_spearman_current_key_vs_tactile": float(correlations.mean()) if len(correlations) else None,
            "current_key": metric_summary(scope_rows, "current"),
            "tactile_oracle_within_geometry_topk": metric_summary(scope_rows, "best"),
        }

    summary = {
        "mode": "validation_only_oracle_box_cache_alignment_audit",
        "purpose": "diagnose whether a visual/geometry cache key preserves tactile similarity inside its geometry shortlist",
        "cache_split": cache_split, "query_split": query_split, "cache_size": len(cache_rows), "query_count": len(query_rows),
        "geometry_filter_k": filter_k, "final_holdout_min_record": final_min_record,
        "tactile_target_note": "Pooled tactile-difference embedding is used only to define a relative target rank; full tactile metrics are reported for selected and oracle candidates.",
        "overall": summarize_scope(query_results),
        "far_probe75_100": summarize_scope([row for row in query_results if int(row["query_probe"]) >= 75]),
        "near_mid_probe5_50": summarize_scope([row for row in query_results if int(row["query_probe"]) < 75]),
    }
    write_csv_rows(project_path(cfg["candidates_csv"]), candidate_rows, CANDIDATE_FIELDS)
    write_csv_rows(project_path(cfg["queries_csv"]), query_results, QUERY_FIELDS)
    write_json(project_path(cfg["metrics_json"]), summary)
    print(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit alignment between local cache keys and tactile similarity.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--section", default="tactile_cache_alignment_phase35_v3")
    args = parser.parse_args()
    audit(args.config, args.section)


if __name__ == "__main__":
    main()
