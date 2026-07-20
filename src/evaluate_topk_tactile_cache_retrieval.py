from __future__ import annotations

import argparse
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .build_cache_retrieval import (
    apply_standardize,
    crop_contact_patch_from_image,
    motion_geometry_feature,
    standardize,
    visual_patch_feature_from_patch,
)
from .config import load_config, project_path
from .evaluate_oracle_tactile_retrieval import tactile_difference, tactile_metrics
from .utils import ensure_dir, read_csv_rows, write_csv_rows, write_json


METRIC_NAMES = [
    "tactile_diff_mae",
    "tactile_ssim",
    "tactile_mask_iou",
    "tactile_area_delta",
    "tactile_centroid_distance",
    "tactile_embedding_distance",
]

RESULT_FIELDS = [
    "mode",
    "dataset_split",
    "query_record_id",
    "query_image_name",
    "query_probe",
    "query_touch_path",
    "query_vision_path",
    "selected_candidate_rank",
    "selected_candidate_x",
    "selected_candidate_y",
    "selected_candidate_heatmap_score",
    "selected_candidate_heatmap_ratio",
    "selected_candidate_box48_hit",
    "c2_top1_box48_hit",
    "top10_box48_hit",
    "retrieved_record_id",
    "retrieved_image_name",
    "retrieved_probe",
    "retrieved_touch_path",
    "retrieved_vision_path",
    "retrieved_x",
    "retrieved_y",
    "geometry_distance",
    "visual_distance",
    "cache_distance",
    "joint_score",
    "cache_miss",
    *METRIC_NAMES,
    "random_tactile_diff_mae",
    "random_tactile_ssim",
    "random_tactile_mask_iou",
]

CANDIDATE_FIELDS = [
    "query_record_id",
    "query_image_name",
    "query_probe",
    "candidate_rank",
    "candidate_x",
    "candidate_y",
    "heatmap_score",
    "heatmap_ratio",
    "candidate_box48_hit",
    "retrieved_record_id",
    "retrieved_image_name",
    "retrieved_x",
    "retrieved_y",
    "geometry_distance",
    "visual_distance",
    "cache_distance",
    "joint_score",
    "cache_miss",
]


def parse_points(value: str, topk: int) -> list[tuple[float, float, float]]:
    points: list[tuple[float, float, float]] = []
    for item in value.split(";"):
        if not item:
            continue
        x, y, score = item.split(",")
        points.append((float(x), float(y), float(score)))
    if not points:
        raise ValueError("Prediction row has no Top-K points.")
    return points[:topk]


def record_number(record_id: str) -> int:
    try:
        return int(record_id.rsplit("_", 1)[-1])
    except ValueError as exc:
        raise ValueError(f"Unexpected record ID: {record_id}") from exc


def box48_hit(x: float, y: float, row: dict[str, str]) -> bool:
    return abs(x - float(row["target_tip_x"])) <= 24.0 and abs(y - float(row["target_tip_y"])) <= 24.0


def visual_feature(image_path: str, x: float, y: float, crop_size: int) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    patch = crop_contact_patch_from_image(image, x, y, crop_size)
    return visual_patch_feature_from_patch(patch)


def find_cache_match(
    row: dict[str, str],
    x: float,
    y: float,
    cache_rows: list[dict[str, str]],
    cache_geometry_z: np.ndarray,
    cache_visual_z: np.ndarray,
    geometry_mean: np.ndarray,
    geometry_std: np.ndarray,
    visual_mean: np.ndarray,
    visual_std: np.ndarray,
    crop_size: int,
    filter_k: int,
    geometry_weight: float,
    visual_weight: float,
) -> dict:
    query_geometry = apply_standardize(motion_geometry_feature(row, x, y)[None], geometry_mean, geometry_std)[0]
    geometry_distances = np.linalg.norm(cache_geometry_z - query_geometry[None], axis=1)
    indices = np.argpartition(geometry_distances, filter_k - 1)[:filter_k]
    query_visual = apply_standardize(visual_feature(row["vision_path"], x, y, crop_size)[None], visual_mean, visual_std)[0]
    visual_distances = np.linalg.norm(cache_visual_z[indices] - query_visual[None], axis=1)
    cache_distance = (
        geometry_weight * geometry_distances[indices] / math.sqrt(cache_geometry_z.shape[1])
        + visual_weight * visual_distances / math.sqrt(cache_visual_z.shape[1])
    )
    local_index = int(np.argmin(cache_distance))
    retrieved_index = int(indices[local_index])
    return {
        "retrieved": cache_rows[retrieved_index],
        "geometry_distance": float(geometry_distances[retrieved_index]),
        "visual_distance": float(visual_distances[local_index]),
        "cache_distance": float(cache_distance[local_index]),
    }


def calibrate_miss_threshold(
    cache_rows: list[dict[str, str]],
    cache_geometry_z: np.ndarray,
    cache_visual_z: np.ndarray,
    filter_k: int,
    geometry_weight: float,
    visual_weight: float,
    quantile: float,
    sample_count: int,
    seed: int,
) -> tuple[float, int]:
    """Calibrate with cache keys only; target tactile data is never used here."""
    rng = random.Random(seed)
    indices = list(range(len(cache_rows)))
    if len(indices) > sample_count:
        indices = rng.sample(indices, sample_count)
    distances: list[float] = []
    for index in indices:
        allowed = np.asarray([row["record_id"] != cache_rows[index]["record_id"] for row in cache_rows])
        if not allowed.any():
            continue
        geometry = np.linalg.norm(cache_geometry_z[allowed] - cache_geometry_z[index][None], axis=1)
        original_indices = np.flatnonzero(allowed)
        local_k = min(filter_k, len(original_indices))
        shortlist_local = np.argpartition(geometry, local_k - 1)[:local_k]
        shortlist = original_indices[shortlist_local]
        visual = np.linalg.norm(cache_visual_z[shortlist] - cache_visual_z[index][None], axis=1)
        combined = (
            geometry_weight * geometry[shortlist_local] / math.sqrt(cache_geometry_z.shape[1])
            + visual_weight * visual / math.sqrt(cache_visual_z.shape[1])
        )
        distances.append(float(np.min(combined)))
    return (float(np.quantile(distances, quantile)) if distances else float("inf"), len(distances))


def summarize(rows: list[dict[str, str]]) -> dict:
    result: dict[str, float | int | None] = {"queries": len(rows)}
    if not rows:
        return result
    for name in METRIC_NAMES:
        values = np.asarray([float(row[name]) for row in rows], dtype=np.float32)
        result[f"mean_{name}"] = float(values.mean())
        result[f"median_{name}"] = float(np.median(values))
    result.update(
        {
            "cache_miss_rate": float(np.mean([row["cache_miss"] == "1" for row in rows])),
            "selected_box48_rate": float(np.mean([row["selected_candidate_box48_hit"] == "1" for row in rows])),
            "c2_top1_box48_rate": float(np.mean([row["c2_top1_box48_hit"] == "1" for row in rows])),
            "top10_box48_recall": float(np.mean([row["top10_box48_hit"] == "1" for row in rows])),
            "non_top1_selection_rate": float(np.mean([row["selected_candidate_rank"] != "1" for row in rows])),
            "mean_random_tactile_diff_mae": float(np.mean([float(row["random_tactile_diff_mae"]) for row in rows])),
            "mean_random_tactile_ssim": float(np.mean([float(row["random_tactile_ssim"]) for row in rows])),
            "mean_random_tactile_mask_iou": float(np.mean([float(row["random_tactile_mask_iou"]) for row in rows])),
        }
    )
    return result


def draw_box(draw: ImageDraw.ImageDraw, x: float, y: float, color: str, box_size: int = 48) -> None:
    half = box_size / 2.0
    draw.rectangle((x - half, y - half, x + half, y + half), outline=color, width=3)


def save_debug(row: dict[str, str], output_path: Path, crop_size: int, tactile_size: int, diff_cache: dict[str, np.ndarray]) -> None:
    query_image = Image.open(row["query_vision_path"]).convert("RGB")
    overview = query_image.copy()
    draw = ImageDraw.Draw(overview)
    draw_box(draw, float(row["selected_candidate_x"]), float(row["selected_candidate_y"]), "cyan")
    overview.thumbnail((288, 192))
    candidate = Image.fromarray(
        np.uint8(np.clip(crop_contact_patch_from_image(query_image, float(row["selected_candidate_x"]), float(row["selected_candidate_y"]), crop_size) * 255.0, 0, 255))
    ).resize((192, 192))
    retrieved_image = Image.open(row["retrieved_vision_path"]).convert("RGB")
    retrieved = Image.fromarray(
        np.uint8(np.clip(crop_contact_patch_from_image(retrieved_image, float(row["retrieved_x"]), float(row["retrieved_y"]), crop_size) * 255.0, 0, 255))
    ).resize((192, 192))
    query_diff = Image.fromarray(np.uint8(np.clip(tactile_difference(row["query_touch_path"], diff_cache, tactile_size) * 3.0, 0, 1) * 255)).resize((192, 192))
    retrieved_diff = Image.fromarray(np.uint8(np.clip(tactile_difference(row["retrieved_touch_path"], diff_cache, tactile_size) * 3.0, 0, 1) * 255)).resize((192, 192))
    canvas = Image.new("RGB", (1056, 236), "black")
    canvas.paste(overview, (0, 36))
    for index, image in enumerate((candidate, retrieved, query_diff, retrieved_diff)):
        canvas.paste(image, (288 + index * 192, 36))
    label = (
        f"{row['mode']} | candidate rank={row['selected_candidate_rank']} | box48={row['selected_candidate_box48_hit']} "
        f"| MAE={float(row['tactile_diff_mae']):.4f} SSIM={float(row['tactile_ssim']):.3f} IoU={float(row['tactile_mask_iou']):.3f}"
    )
    canvas_draw = ImageDraw.Draw(canvas)
    canvas_draw.text((8, 8), label, fill="white")
    canvas_draw.text((8, 220), "query frame: cyan=selected candidate", fill="white")
    canvas_draw.text((296, 220), "query 48x48 crop", fill="white")
    canvas_draw.text((488, 220), "retrieved 48x48 crop", fill="white")
    canvas_draw.text((680, 220), "query tactile difference", fill="white")
    canvas_draw.text((872, 220), "retrieved tactile difference", fill="white")
    ensure_dir(output_path.parent)
    canvas.save(output_path)


def evaluate(config_path: str, section: str) -> dict:
    cfg = load_config(config_path)[section]
    topk = int(cfg.get("topk", 10))
    candidate_source = str(cfg.get("candidate_source", "predicted_topk"))
    if candidate_source not in {"predicted_topk", "target"}:
        raise ValueError(f"Unsupported candidate_source: {candidate_source}")
    rows = read_csv_rows(project_path(cfg["samples_csv"]))
    cache_rows = [row for row in rows if row["dataset_split"] == cfg.get("cache_split", "train")]
    query_split = str(cfg.get("query_split", "val"))
    query_rows = {row["image_name"]: row for row in rows if row["dataset_split"] == query_split}
    if candidate_source == "predicted_topk":
        predictions = read_csv_rows(project_path(cfg["predictions_csv"]))
        query_items = [(query_rows[row["image_name"]], row) for row in predictions if row["dataset_split"] == query_split and row["image_name"] in query_rows]
    else:
        query_items = [(row, None) for row in query_rows.values()]
    final_min_record = int(cfg.get("final_holdout_min_record", 950))
    accessed_records = {row["record_id"] for row in cache_rows} | {row["record_id"] for row in query_rows.values()}
    forbidden = sorted(record_id for record_id in accessed_records if record_number(record_id) >= final_min_record)
    if forbidden:
        raise RuntimeError(f"Refusing to access final-holdout records: {forbidden[:5]}")
    if not cache_rows or not query_items:
        raise RuntimeError("Need non-empty train cache and validation prediction rows.")

    crop_size, tactile_size = int(cfg.get("cache_crop_size", 48)), int(cfg.get("tactile_size", 96))
    filter_k = min(int(cfg.get("geometry_filter_k", 32)), len(cache_rows))
    geometry_weight, visual_weight = float(cfg.get("geometry_weight", 1.0)), float(cfg.get("visual_weight", 1.0))
    heatmap_weight = float(cfg.get("heatmap_weight", 0.25))
    threshold = float(cfg.get("tactile_mask_threshold", 0.04))
    seed = int(cfg.get("seed", 20260724))
    rng = random.Random(seed)

    cache_geometry = np.stack([motion_geometry_feature(row, float(row["target_tip_x"]), float(row["target_tip_y"])) for row in cache_rows])
    cache_geometry_z, geometry_mean, geometry_std = standardize(cache_geometry, cache_geometry)
    cache_visual = np.stack([visual_feature(row["vision_path"], float(row["target_tip_x"]), float(row["target_tip_y"]), crop_size) for row in cache_rows])
    cache_visual_z, visual_mean, visual_std = standardize(cache_visual, cache_visual)
    miss_threshold, calibration_count = calibrate_miss_threshold(
        cache_rows,
        cache_geometry_z,
        cache_visual_z,
        filter_k,
        geometry_weight,
        visual_weight,
        float(cfg.get("cache_miss_quantile", 0.95)),
        int(cfg.get("miss_calibration_samples", 256)),
        seed,
    )

    result_rows: list[dict[str, str]] = []
    candidate_rows: list[dict[str, str]] = []
    debug_rows: list[dict[str, str]] = []
    diff_cache: dict[str, np.ndarray] = {}
    for query, query_prediction in query_items:
        points = (
            parse_points(query_prediction["topk_points"], topk)
            if query_prediction is not None
            else [(float(query["target_tip_x"]), float(query["target_tip_y"]), 1.0)]
        )
        top_score = max(points[0][2], 1e-8)
        evaluated_candidates: list[dict] = []
        for rank, (x, y, score) in enumerate(points, start=1):
            match = find_cache_match(
                query, x, y, cache_rows, cache_geometry_z, cache_visual_z,
                geometry_mean, geometry_std, visual_mean, visual_std, crop_size,
                filter_k, geometry_weight, visual_weight,
            )
            ratio = score / top_score
            match.update({"rank": rank, "x": x, "y": y, "score": score, "ratio": ratio, "box48": box48_hit(x, y, query)})
            match["joint_score"] = match["cache_distance"] - heatmap_weight * ratio
            match["cache_miss"] = match["cache_distance"] > miss_threshold
            evaluated_candidates.append(match)
            retrieved = match["retrieved"]
            candidate_rows.append({
                "query_record_id": query["record_id"], "query_image_name": query["image_name"], "query_probe": query["probe"],
                "candidate_rank": str(rank), "candidate_x": f"{x:.3f}", "candidate_y": f"{y:.3f}",
                "heatmap_score": f"{score:.6f}", "heatmap_ratio": f"{ratio:.6f}", "candidate_box48_hit": str(int(match["box48"])),
                "retrieved_record_id": retrieved["record_id"], "retrieved_image_name": retrieved["image_name"],
                "retrieved_x": retrieved["target_tip_x"], "retrieved_y": retrieved["target_tip_y"],
                "geometry_distance": f"{match['geometry_distance']:.6f}", "visual_distance": f"{match['visual_distance']:.6f}",
                "cache_distance": f"{match['cache_distance']:.6f}", "joint_score": f"{match['joint_score']:.6f}", "cache_miss": str(int(match["cache_miss"])),
            })
        selections = (
            {"c2_top1_cache": evaluated_candidates[0], "topk_cache_joint": min(evaluated_candidates, key=lambda item: item["joint_score"])}
            if candidate_source == "predicted_topk"
            else {"oracle_box_cache": evaluated_candidates[0]}
        )
        top10_hit = any(item["box48"] for item in evaluated_candidates)
        query_diff = tactile_difference(query["touch_path"], diff_cache, tactile_size)
        random_cache = cache_rows[rng.randrange(len(cache_rows))]
        random_metrics = tactile_metrics(query_diff, tactile_difference(random_cache["touch_path"], diff_cache, tactile_size), threshold)
        for mode, selected in selections.items():
            retrieved = selected["retrieved"]
            metrics = tactile_metrics(query_diff, tactile_difference(retrieved["touch_path"], diff_cache, tactile_size), threshold)
            output = {
                "mode": mode, "dataset_split": query["dataset_split"], "query_record_id": query["record_id"],
                "query_image_name": query["image_name"], "query_probe": query["probe"], "query_touch_path": query["touch_path"], "query_vision_path": query["vision_path"],
                "selected_candidate_rank": str(selected["rank"]), "selected_candidate_x": f"{selected['x']:.3f}", "selected_candidate_y": f"{selected['y']:.3f}",
                "selected_candidate_heatmap_score": f"{selected['score']:.6f}", "selected_candidate_heatmap_ratio": f"{selected['ratio']:.6f}",
                "selected_candidate_box48_hit": str(int(selected["box48"])), "c2_top1_box48_hit": str(int(evaluated_candidates[0]["box48"])), "top10_box48_hit": str(int(top10_hit)),
                "retrieved_record_id": retrieved["record_id"], "retrieved_image_name": retrieved["image_name"], "retrieved_probe": retrieved["probe"],
                "retrieved_touch_path": retrieved["touch_path"], "retrieved_vision_path": retrieved["vision_path"], "retrieved_x": retrieved["target_tip_x"], "retrieved_y": retrieved["target_tip_y"],
                "geometry_distance": f"{selected['geometry_distance']:.6f}", "visual_distance": f"{selected['visual_distance']:.6f}",
                "cache_distance": f"{selected['cache_distance']:.6f}", "joint_score": f"{selected['joint_score']:.6f}", "cache_miss": str(int(selected["cache_miss"])),
            }
            output.update({name: f"{value:.6f}" for name, value in metrics.items()})
            output.update({f"random_{name}": f"{random_metrics[name]:.6f}" for name in ("tactile_diff_mae", "tactile_ssim", "tactile_mask_iou")})
            result_rows.append(output)
            if mode == "topk_cache_joint":
                debug_rows.append(output)

    metrics: dict[str, object] = {
        "mode": "validation_only_oracle_local_tactile_cache" if candidate_source == "target" else "validation_only_predicted_topk_local_tactile_cache",
        "purpose": "diagnostic; cache matching is never trained or selected on final holdout",
        "cache_split": cfg.get("cache_split", "train"), "query_split": query_split,
        "candidate_source": candidate_source, "cache_size": len(cache_rows), "query_count": len(query_items), "topk": topk, "geometry_filter_k": filter_k,
        "heatmap_weight": heatmap_weight, "cache_miss_threshold": miss_threshold, "miss_calibration_count": calibration_count,
        "final_holdout_min_record": final_min_record,
        "by_mode": {},
    }
    for mode in sorted({row["mode"] for row in result_rows}):
        mode_rows = [row for row in result_rows if row["mode"] == mode]
        metrics["by_mode"][mode] = {
            "overall": summarize(mode_rows),
            "far_probe75_100": summarize([row for row in mode_rows if int(row["query_probe"]) >= 75]),
            "near_mid_probe5_50": summarize([row for row in mode_rows if int(row["query_probe"]) < 75]),
        }

    output_csv = project_path(cfg["output_csv"])
    candidates_csv = project_path(cfg["candidates_csv"])
    write_csv_rows(output_csv, result_rows, RESULT_FIELDS)
    write_csv_rows(candidates_csv, candidate_rows, CANDIDATE_FIELDS)
    debug_dir = project_path(cfg["debug_dir"])
    for index, row in enumerate(sorted(debug_rows, key=lambda item: float(item["tactile_diff_mae"]), reverse=True)[: int(cfg.get("debug_samples", 20))]):
        save_debug(row, debug_dir / f"{index:03d}_{row['query_image_name']}", crop_size, tactile_size, diff_cache)
    write_json(project_path(cfg["metrics_json"]), metrics)
    print(metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate predicted Top-K contact boxes with local tactile cache retrieval.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--section", default="topk_tactile_cache_retrieval_phase35_v3")
    args = parser.parse_args()
    evaluate(args.config, args.section)


if __name__ == "__main__":
    main()
