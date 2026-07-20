from __future__ import annotations

import argparse
import math

import numpy as np
import torch

from .build_cache_retrieval import crop_contact_patch, motion_geometry_feature, standardize, visual_patch_feature_from_patch
from .config import load_config, project_path
from .evaluate_oracle_tactile_retrieval import tactile_difference, tactile_metrics
from .train_soft_tactile_cache_ranker import CandidateGroups, SoftTactileRanker, predict
from .utils import read_csv_rows, write_csv_rows, write_json


METRICS = [
    "tactile_diff_mae",
    "tactile_ssim",
    "tactile_mask_iou",
    "tactile_area_delta",
    "tactile_centroid_distance",
    "tactile_embedding_distance",
]
QUERY_FIELDS = [
    "mode", "query_record_id", "query_image_name", "query_probe", "selected_candidate_rank",
    "selected_candidate_x", "selected_candidate_y", "selected_candidate_box48_hit",
    "retrieved_cache_record_id", "retrieved_cache_image_name", "oracle_objective_tactile_diff_mae",
    *METRICS,
]
CANDIDATE_FIELDS = [
    "query_record_id", "query_image_name", "query_probe", "candidate_rank", "candidate_x", "candidate_y",
    "candidate_box48_hit", "heatmap_score", "retrieved_cache_record_id", "retrieved_cache_image_name",
    "oracle_objective_tactile_diff_mae",
]


def record_number(record_id: str) -> int:
    try:
        return int(record_id.rsplit("_", 1)[-1])
    except ValueError as exc:
        raise ValueError(f"Unexpected record ID: {record_id}") from exc


def is_final_holdout(row: dict[str, str]) -> bool:
    # Split 1 legitimately starts again at rec_01000. Only split-0 rec_00950..999 is sealed.
    return row.get("split") == "0" and 950 <= record_number(row["record_id"]) <= 999


def parse_points(value: str, topk: int) -> list[tuple[float, float, float]]:
    points = []
    for item in value.split(";")[:topk]:
        x, y, score = item.split(",")
        points.append((float(x), float(y), float(score)))
    if not points:
        raise ValueError("Prediction contains no Top-K points.")
    return points


def box48_hit(x: float, y: float, row: dict[str, str]) -> bool:
    return abs(x - float(row["target_tip_x"])) <= 24.0 and abs(y - float(row["target_tip_y"])) <= 24.0


def summarize(rows: list[dict[str, str]]) -> dict[str, float | int | None]:
    summary: dict[str, float | int | None] = {"queries": len(rows)}
    for metric in METRICS:
        values = np.asarray([float(row[metric]) for row in rows], dtype=np.float32)
        summary[f"mean_{metric}"] = float(values.mean()) if len(values) else None
        summary[f"median_{metric}"] = float(np.median(values)) if len(values) else None
    hits = [row["selected_candidate_box48_hit"] == "1" for row in rows]
    summary["selected_candidate_box48_rate"] = float(np.mean(hits)) if hits else None
    return summary


def metric_row(
    mode: str,
    query: dict[str, str],
    candidate_rank: int,
    x: float,
    y: float,
    cache: dict[str, str],
    query_tactile: np.ndarray,
    cache_tactile: np.ndarray,
    threshold: float,
) -> dict[str, str]:
    metrics = tactile_metrics(query_tactile, cache_tactile, threshold)
    return {
        "mode": mode,
        "query_record_id": query["record_id"],
        "query_image_name": query["image_name"],
        "query_probe": query["probe"],
        "selected_candidate_rank": str(candidate_rank),
        "selected_candidate_x": f"{x:.3f}",
        "selected_candidate_y": f"{y:.3f}",
        "selected_candidate_box48_hit": str(int(box48_hit(x, y, query))),
        "retrieved_cache_record_id": cache["record_id"],
        "retrieved_cache_image_name": cache["image_name"],
        "oracle_objective_tactile_diff_mae": f"{metrics['tactile_diff_mae']:.6f}",
        **{key: f"{value:.6f}" for key, value in metrics.items()},
    }


def evaluate(config_path: str, section: str) -> dict:
    cfg = load_config(config_path)[section]
    rows = read_csv_rows(project_path(cfg["samples_csv"]))
    forbidden = [row["record_id"] for row in rows if is_final_holdout(row)]
    if forbidden:
        raise RuntimeError(f"Refusing to access sealed final holdout: {sorted(set(forbidden))[:5]}")

    cache_rows = [row for row in rows if row["dataset_split"] == "train"]
    query_by_name = {row["image_name"]: row for row in rows if row["dataset_split"] == "val"}
    predictions = [row for row in read_csv_rows(project_path(cfg["predictions_csv"])) if row.get("dataset_split") == "val"]
    predictions = [row for row in predictions if row["image_name"] in query_by_name]
    if len(predictions) != len(query_by_name):
        raise RuntimeError(f"Need one validation prediction per query, got {len(predictions)} for {len(query_by_name)} rows.")
    if any(is_final_holdout(row) for row in predictions):
        raise RuntimeError("Prediction file includes a sealed final-holdout row.")
    cache_records = {row["record_id"] for row in cache_rows}
    query_records = {row["record_id"] for row in query_by_name.values()}
    overlap = cache_records & query_records
    if overlap:
        raise RuntimeError(f"Cache/query record overlap: {sorted(overlap)[:5]}")

    topk = int(cfg.get("topk", 10))
    crop_size = int(cfg.get("cache_crop_size", 48))
    tactile_size = int(cfg.get("tactile_size", 96))
    threshold = float(cfg.get("tactile_mask_threshold", 0.04))
    filter_k = min(int(cfg.get("geometry_filter_k", 32)), len(cache_rows))

    raw_cache_geometry = np.stack([
        motion_geometry_feature(row, float(row["target_tip_x"]), float(row["target_tip_y"])) for row in cache_rows
    ])
    cache_geometry, geometry_mean, geometry_std = standardize(raw_cache_geometry, raw_cache_geometry)
    cache_geometry = cache_geometry.astype(np.float32)
    cache_patches = np.stack([
        crop_contact_patch(row["vision_path"], float(row["target_tip_x"]), float(row["target_tip_y"]), crop_size) for row in cache_rows
    ]).astype(np.float32)
    cache_hand_raw = np.stack([visual_patch_feature_from_patch(patch) for patch in cache_patches])
    cache_hand, hand_mean, hand_std = standardize(cache_hand_raw, cache_hand_raw)

    diff_cache: dict[str, np.ndarray] = {}

    def touch(row: dict[str, str]) -> np.ndarray:
        return tactile_difference(row["touch_path"], diff_cache, tactile_size)

    def candidate_group(query: dict[str, str], x: float, y: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        patch = crop_contact_patch(query["vision_path"], x, y, crop_size)
        geometry = ((motion_geometry_feature(query, x, y) - geometry_mean) / geometry_std).astype(np.float32)
        hand = ((visual_patch_feature_from_patch(patch) - hand_mean) / hand_std).astype(np.float32)
        distances = np.linalg.norm(cache_geometry - geometry[None], axis=1)
        shortlist = np.argpartition(distances, filter_k - 1)[:filter_k]
        shortlist = shortlist[np.argsort(distances[shortlist], kind="stable")]
        visual = np.linalg.norm(cache_hand[shortlist] - hand[None], axis=1)
        current = distances[shortlist] / math.sqrt(cache_geometry.shape[1]) + visual / math.sqrt(cache_hand.shape[1])
        return patch.astype(np.float32), geometry, shortlist.astype(np.int32), current.astype(np.float32), distances[shortlist].astype(np.float32)

    # Build C2 Top-1 groups once for the current frozen cache-ranker baseline (A).
    ordered_predictions = sorted(predictions, key=lambda row: row["image_name"])
    top1_groups, topk_groups, gt_groups = [], [], []
    query_tactiles: list[np.ndarray] = []
    for prediction in ordered_predictions:
        query = query_by_name[prediction["image_name"]]
        points = parse_points(prediction["topk_points"], topk)
        if len(points) != topk:
            raise RuntimeError(f"Expected {topk} points for {query['image_name']}, got {len(points)}")
        top1_groups.append(candidate_group(query, points[0][0], points[0][1]))
        topk_groups.append([candidate_group(query, x, y) for x, y, _ in points])
        gt_groups.append(candidate_group(query, float(query["target_tip_x"]), float(query["target_tip_y"])))
        query_tactiles.append(touch(query))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = project_path(cfg["checkpoint"])
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = SoftTactileRanker(cache_geometry.shape[1], float(cfg.get("dropout", 0.1))).to(device)
    model.load_state_dict(checkpoint["model_state"])
    top1_patches = np.stack([group[0] for group in top1_groups])
    top1_geometry = np.stack([group[1] for group in top1_groups])
    top1_candidates = np.stack([group[2] for group in top1_groups])
    top1_current = np.stack([group[3] for group in top1_groups])
    soft_groups = CandidateGroups(
        candidates=top1_candidates,
        targets=np.zeros_like(top1_current),
        current_scores=top1_current,
    )
    soft_scores = predict(
        model, soft_groups, top1_patches, top1_geometry, cache_patches, cache_geometry, device, int(cfg.get("batch_size", 16))
    )
    soft_local = np.argmin(soft_scores, axis=1)

    output_rows: list[dict[str, str]] = []
    candidate_rows: list[dict[str, str]] = []
    top10_cover, top1_cover = [], []
    for index, prediction in enumerate(ordered_predictions):
        query = query_by_name[prediction["image_name"]]
        query_touch = query_tactiles[index]
        points = parse_points(prediction["topk_points"], topk)

        # A: frozen C2 Top-1 contact crop + deployed frozen soft cache ranker.
        top1_x, top1_y, _ = points[0]
        soft_cache_index = int(top1_groups[index][2][soft_local[index]])
        output_rows.append(metric_row(
            "a_c2_top1_frozen_soft_cache", query, 1, top1_x, top1_y, cache_rows[soft_cache_index],
            query_touch, touch(cache_rows[soft_cache_index]), threshold,
        ))

        oracle_candidates: list[tuple[int, float, float, int, float]] = []
        for rank, ((x, y, heat_score), group) in enumerate(zip(points, topk_groups[index]), start=1):
            shortlist = group[2]
            candidate_touches = np.stack([touch(cache_rows[int(cache_index)]) for cache_index in shortlist])
            mae = np.abs(candidate_touches - query_touch[None]).mean(axis=(1, 2, 3))
            local = int(np.argmin(mae))
            cache_index = int(shortlist[local])
            oracle_candidates.append((rank, x, y, cache_index, float(mae[local])))
            candidate_rows.append({
                "query_record_id": query["record_id"], "query_image_name": query["image_name"], "query_probe": query["probe"],
                "candidate_rank": str(rank), "candidate_x": f"{x:.3f}", "candidate_y": f"{y:.3f}",
                "candidate_box48_hit": str(int(box48_hit(x, y, query))), "heatmap_score": f"{heat_score:.6f}",
                "retrieved_cache_record_id": cache_rows[cache_index]["record_id"], "retrieved_cache_image_name": cache_rows[cache_index]["image_name"],
                "oracle_objective_tactile_diff_mae": f"{mae[local]:.6f}",
            })

        # B: same C2 Top-1 crop, but an offline tactile oracle selects the cache entry.
        rank, x, y, cache_index, _ = oracle_candidates[0]
        output_rows.append(metric_row(
            "b_c2_top1_tactile_oracle_cache", query, rank, x, y, cache_rows[cache_index],
            query_touch, touch(cache_rows[cache_index]), threshold,
        ))

        # C1 is deliberately unconstrained: it exposes whether tactile-only selection would
        # abandon the physically correct contact box to obtain an easier tactile match.
        rank, x, y, cache_index, _ = min(oracle_candidates, key=lambda item: item[4])
        output_rows.append(metric_row(
            "c1_c2_top10_tactile_oracle_unconstrained", query, rank, x, y, cache_rows[cache_index],
            query_touch, touch(cache_rows[cache_index]), threshold,
        ))

        # C2 is the usable upper bound: it may choose only among Top-10 candidates which
        # already localize the contact inside the 48x48 target box. A missing candidate is
        # recorded separately instead of silently falling back to an incorrect one.
        localized = [item for item in oracle_candidates if box48_hit(item[1], item[2], query)]
        if localized:
            rank, x, y, cache_index, _ = min(localized, key=lambda item: item[4])
            output_rows.append(metric_row(
                "c2_c2_top10_box48_tactile_oracle", query, rank, x, y, cache_rows[cache_index],
                query_touch, touch(cache_rows[cache_index]), threshold,
            ))

        # D: correct (GT) contact crop with the same tactile-oracle cache selection.
        gt_patch, gt_geometry, gt_shortlist, _, _ = gt_groups[index]
        del gt_patch, gt_geometry
        gt_touches = np.stack([touch(cache_rows[int(cache_index)]) for cache_index in gt_shortlist])
        gt_mae = np.abs(gt_touches - query_touch[None]).mean(axis=(1, 2, 3))
        gt_cache_index = int(gt_shortlist[int(np.argmin(gt_mae))])
        gt_x, gt_y = float(query["target_tip_x"]), float(query["target_tip_y"])
        output_rows.append(metric_row(
            "d_gt_contact_tactile_oracle_cache", query, 0, gt_x, gt_y, cache_rows[gt_cache_index],
            query_touch, touch(cache_rows[gt_cache_index]), threshold,
        ))
        hits = [box48_hit(x, y, query) for x, y, _ in points]
        top1_cover.append(hits[0])
        top10_cover.append(any(hits))

    modes = {row["mode"] for row in output_rows}
    summary_by_mode = {mode: summarize([row for row in output_rows if row["mode"] == mode]) for mode in sorted(modes)}
    far_by_mode = {
        mode: summarize([row for row in output_rows if row["mode"] == mode and int(row["query_probe"]) >= 75])
        for mode in sorted(modes)
    }
    summary = {
        "mode": "phase4a_validation_only_topk_contact_to_tactile_oracle_decomposition",
        "device": str(device),
        "cache_size": len(cache_rows),
        "validation_queries": len(ordered_predictions),
        "topk": topk,
        "geometry_filter_k": filter_k,
        "top1_box48_rate": float(np.mean(top1_cover)),
        "top10_box48_recall": float(np.mean(top10_cover)),
        "metrics_by_mode": summary_by_mode,
        "far_metrics_by_mode": far_by_mode,
        "checkpoint": str(checkpoint_path),
        "sealed_final_holdout": "split-0 rec_00950..rec_00999; no rows were read",
        "oracle_definition": "B/C/D choose the cache entry with minimum true tactile difference-map MAE inside the geometry-filtered train cache. C1 also chooses the best of C2 Top-10 contact candidates without spatial constraint; C2 restricts that choice to candidates already inside GT Box48. These are offline upper bounds only, never online policies.",
        "interpretation": "A->B is cache-ranking headroom at the same C2 box. C1 is a failure diagnostic for tactile-only candidate choice. B->C2 is usable Top-K contact-candidate headroom, while C2->D is remaining localization/Top-K coverage headroom.",
    }
    write_csv_rows(project_path(cfg["output_csv"]), output_rows, QUERY_FIELDS)
    write_csv_rows(project_path(cfg["candidates_csv"]), candidate_rows, CANDIDATE_FIELDS)
    write_json(project_path(cfg["metrics_json"]), summary)
    print(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4A: diagnose Top-K contact-to-tactile cache headroom with validation-only oracles.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--section", default="phase4a_topk_tactile_oracle_v4")
    args = parser.parse_args()
    evaluate(args.config, args.section)


if __name__ == "__main__":
    main()
