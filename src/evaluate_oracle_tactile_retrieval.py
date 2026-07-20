from __future__ import annotations

import argparse
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .build_cache_retrieval import visual_patch_feature
from .config import load_config, project_path
from .utils import ensure_dir, parse_frame_id, read_csv_rows, write_csv_rows, write_json


FIELDS = [
    "dataset_split", "query_record_id", "query_image_name", "query_probe",
    "retrieved_record_id", "retrieved_image_name", "retrieved_probe", "filter_rank",
    "geometry_distance", "visual_distance", "combined_distance", "cache_miss",
    "tactile_diff_mae", "tactile_ssim", "tactile_mask_iou", "tactile_area_delta",
    "tactile_centroid_distance", "tactile_embedding_distance", "random_tactile_diff_mae",
    "random_tactile_ssim", "random_tactile_mask_iou", "query_touch_path", "retrieved_touch_path",
    "query_vision_path", "retrieved_vision_path", "query_x", "query_y", "retrieved_x", "retrieved_y",
]


def crop(image: Image.Image, x: float, y: float, size: int) -> Image.Image:
    left, top = int(round(x - size / 2)), int(round(y - size / 2))
    canvas = Image.new("RGB", (size, size), "black")
    source = image.crop((max(left, 0), max(top, 0), min(left + size, image.width), min(top + size, image.height)))
    canvas.paste(source, (max(-left, 0), max(-top, 0)))
    return canvas


def tactile_difference(path: str, cache: dict[str, np.ndarray], size: int) -> np.ndarray:
    if path in cache:
        return cache[path]
    image_path = Path(path)
    image = Image.open(image_path).convert("RGB").resize((size, size), Image.BILINEAR)
    frames = sorted(
        [item for item in image_path.parent.iterdir() if item.is_file() and item.suffix.lower() in {".jpg", ".jpeg", ".png"}],
        key=lambda item: parse_frame_id(item) if parse_frame_id(item) is not None else -1,
    )
    reference_path = frames[0] if frames else image_path
    reference = Image.open(reference_path).convert("RGB").resize((size, size), Image.BILINEAR)
    diff = np.abs(np.asarray(image, dtype=np.float32) / 255.0 - np.asarray(reference, dtype=np.float32) / 255.0)
    cache[path] = diff
    return diff


def tactile_embedding(diff: np.ndarray, grid: int = 8) -> np.ndarray:
    gray = (0.299 * diff[:, :, 0] + 0.587 * diff[:, :, 1] + 0.114 * diff[:, :, 2]).astype(np.float32)
    height, width = gray.shape
    cell_h, cell_w = height // grid, width // grid
    pooled = gray[: cell_h * grid, : cell_w * grid].reshape(grid, cell_h, grid, cell_w).mean(axis=(1, 3))
    return np.concatenate([pooled.reshape(-1), diff.reshape(-1, 3).mean(axis=0), diff.reshape(-1, 3).std(axis=0)]).astype(np.float32)


def tactile_metrics(left: np.ndarray, right: np.ndarray, threshold: float) -> dict[str, float]:
    left_gray = 0.299 * left[:, :, 0] + 0.587 * left[:, :, 1] + 0.114 * left[:, :, 2]
    right_gray = 0.299 * right[:, :, 0] + 0.587 * right[:, :, 1] + 0.114 * right[:, :, 2]
    left_mask, right_mask = left_gray >= threshold, right_gray >= threshold
    union = np.logical_or(left_mask, right_mask).sum()
    iou = float(np.logical_and(left_mask, right_mask).sum() / union) if union else 1.0
    left_area, right_area = float(left_mask.mean()), float(right_mask.mean())

    def centroid(mask: np.ndarray) -> np.ndarray:
        ys, xs = np.where(mask)
        if not len(xs):
            return np.asarray([0.5, 0.5], dtype=np.float32)
        return np.asarray([xs.mean() / max(mask.shape[1] - 1, 1), ys.mean() / max(mask.shape[0] - 1, 1)], dtype=np.float32)

    mu_l, mu_r = float(left_gray.mean()), float(right_gray.mean())
    var_l, var_r = float(left_gray.var()), float(right_gray.var())
    cov = float(((left_gray - mu_l) * (right_gray - mu_r)).mean())
    c1, c2 = 0.01**2, 0.03**2
    ssim = ((2 * mu_l * mu_r + c1) * (2 * cov + c2)) / ((mu_l**2 + mu_r**2 + c1) * (var_l + var_r + c2))
    return {
        "tactile_diff_mae": float(np.abs(left - right).mean()),
        "tactile_ssim": float(ssim),
        "tactile_mask_iou": iou,
        "tactile_area_delta": abs(left_area - right_area),
        "tactile_centroid_distance": float(np.linalg.norm(centroid(left_mask) - centroid(right_mask))),
        "tactile_embedding_distance": float(np.linalg.norm(tactile_embedding(left) - tactile_embedding(right))),
    }


def geometry_feature(row: dict[str, str]) -> np.ndarray:
    width, height = float(row["image_width"]), float(row["image_height"])
    dx = (float(row["target_tip_x"]) - float(row["tip_x"])) / width
    dy = (float(row["target_tip_y"]) - float(row["tip_y"])) / height
    return np.asarray([
        float(row["probe"]) / 100.0, dx, dy,
        float(row["direction_x"]), float(row["direction_y"]),
        (float(row["tip_x"]) - float(row["base_x"])) / width,
        (float(row["tip_y"]) - float(row["base_y"])) / height,
    ], dtype=np.float32)


def standardize(matrix: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean, std = reference.mean(axis=0), reference.std(axis=0)
    std[std < 1e-6] = 1.0
    return (matrix - mean) / std, mean, std


def summarize(rows: list[dict[str, str]], prefix: str = "") -> dict:
    metric_names = ["tactile_diff_mae", "tactile_ssim", "tactile_mask_iou", "tactile_area_delta", "tactile_centroid_distance", "tactile_embedding_distance"]
    result = {"queries": len(rows)}
    for name in metric_names:
        values = [float(row[f"{prefix}{name}"]) for row in rows if row.get(f"{prefix}{name}", "") != ""]
        result[f"mean_{name}"] = float(np.mean(values)) if values else None
        result[f"median_{name}"] = float(np.median(values)) if values else None
    result["cache_miss_rate"] = float(np.mean([row["cache_miss"] == "1" for row in rows])) if rows else None
    return result


def save_debug(row: dict[str, str], output_path: Path, size: int, diff_cache: dict[str, np.ndarray]) -> None:
    query = crop(Image.open(row["query_vision_path"]).convert("RGB"), float(row["query_x"]), float(row["query_y"]), size).resize((192, 192))
    retrieved = crop(Image.open(row["retrieved_vision_path"]).convert("RGB"), float(row["retrieved_x"]), float(row["retrieved_y"]), size).resize((192, 192))
    query_diff = Image.fromarray(np.uint8(np.clip(tactile_difference(row["query_touch_path"], diff_cache, size) * 3.0, 0, 1) * 255)).resize((192, 192))
    retrieved_diff = Image.fromarray(np.uint8(np.clip(tactile_difference(row["retrieved_touch_path"], diff_cache, size) * 3.0, 0, 1) * 255)).resize((192, 192))
    canvas = Image.new("RGB", (768, 224), "black")
    for index, image in enumerate((query, retrieved, query_diff, retrieved_diff)):
        canvas.paste(image, (index * 192, 32))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), f"oracle box cache | MAE={float(row['tactile_diff_mae']):.4f} SSIM={float(row['tactile_ssim']):.3f} IoU={float(row['tactile_mask_iou']):.3f}", fill="white")
    ensure_dir(output_path.parent)
    canvas.save(output_path)


def evaluate(config_path: str, section: str) -> dict:
    cfg = load_config(config_path)[section]
    rows = read_csv_rows(project_path(cfg["samples_csv"]))
    cache_rows = [row for row in rows if row["dataset_split"] == cfg.get("cache_split", "train")]
    query_splits = {str(value) for value in cfg.get("query_splits", ["val"])}
    query_rows = [row for row in rows if row["dataset_split"] in query_splits]
    crop_size, tactile_size = int(cfg.get("cache_crop_size", 48)), int(cfg.get("tactile_size", 96))
    filter_k = min(int(cfg.get("filter_k", 32)), len(cache_rows))
    geometry_weight, visual_weight = float(cfg.get("geometry_weight", 1.0)), float(cfg.get("visual_weight", 1.0))
    threshold = float(cfg.get("tactile_mask_threshold", 0.04))
    rng = random.Random(int(cfg.get("seed", 42)))
    cache_geometry = np.stack([geometry_feature(row) for row in cache_rows])
    cache_geometry_z, mean, std = standardize(cache_geometry, cache_geometry)
    cache_visual = np.stack([visual_patch_feature(row["vision_path"], float(row["target_tip_x"]), float(row["target_tip_y"]), crop_size) for row in cache_rows])
    cache_visual_z, visual_mean, visual_std = standardize(cache_visual, cache_visual)
    diff_cache: dict[str, np.ndarray] = {}
    output_rows, debug_rows, calibration_distances = [], [], []

    # Leave-one-record-out train distances calibrate cache-miss threshold without query labels.
    for index, row in enumerate(cache_rows):
        allowed = np.asarray([item["record_id"] != row["record_id"] for item in cache_rows])
        if not allowed.any():
            continue
        dists = np.linalg.norm(cache_geometry_z[allowed] - cache_geometry_z[index], axis=1)
        calibration_distances.append(float(np.min(dists)))
    miss_threshold = float(np.quantile(calibration_distances, float(cfg.get("cache_miss_quantile", 0.95)))) if calibration_distances else float("inf")

    for query_index, query in enumerate(query_rows):
        query_geometry = (geometry_feature(query) - mean) / std
        geometry_distances = np.linalg.norm(cache_geometry_z - query_geometry[None], axis=1)
        candidate_indices = np.argsort(geometry_distances)[:filter_k]
        query_visual = visual_patch_feature(query["vision_path"], float(query["target_tip_x"]), float(query["target_tip_y"]), crop_size)
        query_visual_z = (query_visual - visual_mean) / visual_std
        visual_distances = np.linalg.norm(cache_visual_z[candidate_indices] - query_visual_z[None], axis=1)
        combined = geometry_weight * geometry_distances[candidate_indices] / math.sqrt(cache_geometry_z.shape[1]) + visual_weight * visual_distances / math.sqrt(cache_visual_z.shape[1])
        local_index = int(np.argmin(combined))
        retrieved_index = int(candidate_indices[local_index])
        retrieved = cache_rows[retrieved_index]
        query_diff = tactile_difference(query["touch_path"], diff_cache, tactile_size)
        retrieved_diff = tactile_difference(retrieved["touch_path"], diff_cache, tactile_size)
        metrics = tactile_metrics(query_diff, retrieved_diff, threshold)
        random_index = rng.randrange(len(cache_rows))
        random_metrics = tactile_metrics(query_diff, tactile_difference(cache_rows[random_index]["touch_path"], diff_cache, tactile_size), threshold)
        combined_distance = float(combined[local_index])
        row = {
            "dataset_split": query["dataset_split"], "query_record_id": query["record_id"], "query_image_name": query["image_name"], "query_probe": query["probe"],
            "retrieved_record_id": retrieved["record_id"], "retrieved_image_name": retrieved["image_name"], "retrieved_probe": retrieved["probe"],
            "filter_rank": str(local_index + 1), "geometry_distance": f"{float(geometry_distances[retrieved_index]):.6f}",
            "visual_distance": f"{float(visual_distances[local_index]):.6f}", "combined_distance": f"{combined_distance:.6f}",
            "cache_miss": "1" if combined_distance > miss_threshold else "0",
            "query_touch_path": query["touch_path"], "retrieved_touch_path": retrieved["touch_path"],
            "query_vision_path": query["vision_path"], "retrieved_vision_path": retrieved["vision_path"],
            "query_x": query["target_tip_x"], "query_y": query["target_tip_y"], "retrieved_x": retrieved["target_tip_x"], "retrieved_y": retrieved["target_tip_y"],
        }
        row.update({name: f"{value:.6f}" for name, value in metrics.items()})
        row.update({f"random_{name}": f"{value:.6f}" for name, value in random_metrics.items() if name in {"tactile_diff_mae", "tactile_ssim", "tactile_mask_iou"}})
        output_rows.append(row)
        debug_rows.append(row)

    output_csv = project_path(cfg["output_csv"])
    write_csv_rows(output_csv, output_rows, FIELDS)
    debug_dir = project_path(cfg["debug_dir"])
    for index, row in enumerate(sorted(debug_rows, key=lambda item: float(item["tactile_diff_mae"]), reverse=True)[: int(cfg.get("debug_samples", 20))]):
        save_debug(row, debug_dir / f"{index:03d}_{row['query_image_name']}", tactile_size, diff_cache)
    summary = {
        "mode": "oracle_contact_box_two_stage_retrieval", "cache_split": cfg.get("cache_split", "train"), "query_splits": sorted(query_splits),
        "cache_size": len(cache_rows), "filter_k": filter_k, "miss_threshold": miss_threshold,
        "two_stage": summarize(output_rows), "random_baseline": {
            "mean_tactile_diff_mae": float(np.mean([float(row["random_tactile_diff_mae"]) for row in output_rows])) if output_rows else None,
            "mean_tactile_ssim": float(np.mean([float(row["random_tactile_ssim"]) for row in output_rows])) if output_rows else None,
            "mean_tactile_mask_iou": float(np.mean([float(row["random_tactile_mask_iou"]) for row in output_rows])) if output_rows else None,
        },
        "metrics_note": "Tactile embedding is a fixed pooled difference-map descriptor; it is an evaluation baseline, not a learned tactile encoder.",
        "output_csv": str(output_csv), "debug_dir": str(debug_dir),
    }
    write_json(project_path(cfg["metrics_json"]), summary)
    print(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate oracle-contact-box two-stage visual cache retrieval with tactile metrics.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--section", default="oracle_box_tactile_retrieval")
    args = parser.parse_args()
    evaluate(args.config, args.section)


if __name__ == "__main__":
    main()
