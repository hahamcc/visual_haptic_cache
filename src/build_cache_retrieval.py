from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .config import load_config, project_path
from .utils import ensure_dir, read_csv_rows, write_csv_rows, write_json


RETRIEVAL_FIELDS = [
    "mode",
    "dataset_split",
    "query_split",
    "query_record_id",
    "query_image_name",
    "query_vision_path",
    "query_touch_path",
    "query_pred_x",
    "query_pred_y",
    "query_target_x",
    "query_target_y",
    "query_probe",
    "retrieved_split",
    "retrieved_record_id",
    "retrieved_image_name",
    "retrieved_vision_path",
    "retrieved_touch_path",
    "retrieved_target_x",
    "retrieved_target_y",
    "retrieved_probe",
    "distance",
    "motion_distance",
    "visual_distance",
    "probe_delta",
    "direction_cosine",
    "query_gt_retrieved_gt_error_px",
    "same_record",
]

COMPARISON_FIELDS = [
    "query_image_name",
    "query_record_id",
    "query_probe",
    "direct_retrieved_image_name",
    "direct_distance",
    "hybrid_retrieved_image_name",
    "hybrid_distance",
    "changed",
]


def read_manifest_touch_paths(path: Path) -> dict[tuple[str, str, int], str]:
    if not path.exists():
        return {}
    return {
        (row["split"], row["record_id"], int(row["frame_id"])): row["touch_path"]
        for row in read_csv_rows(path)
    }


def attach_touch_paths(rows: list[dict[str, str]], manifest_csv: Path) -> None:
    touch_by_key = read_manifest_touch_paths(manifest_csv)
    for row in rows:
        if row.get("touch_path"):
            continue
        split = row["split"]
        record_id = row["record_id"]
        contact_frame_text = row.get("contact_frame_from_name") or row.get("contact_frame_detected") or row.get("contact_frame")
        if not contact_frame_text:
            row["touch_path"] = ""
            continue
        contact_frame = int(contact_frame_text)
        current_frame = int(row["frame_id"])
        row["touch_path"] = touch_by_key.get(
            (split, record_id, contact_frame),
            touch_by_key.get((split, record_id, current_frame), ""),
        )


def draw_box(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    box_size: int,
    color: str,
    width: int = 3,
) -> None:
    half = box_size / 2.0
    draw.rectangle((x - half, y - half, x + half, y + half), outline=color, width=width)


def crop_contact_patch(image_path: str, x: float, y: float, crop_size: int) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    left = int(round(x - crop_size / 2.0))
    top = int(round(y - crop_size / 2.0))
    right = left + crop_size
    bottom = top + crop_size

    src_left = max(0, left)
    src_top = max(0, top)
    src_right = min(image.width, right)
    src_bottom = min(image.height, bottom)

    canvas = Image.new("RGB", (crop_size, crop_size), "black")
    if src_right > src_left and src_bottom > src_top:
        patch = image.crop((src_left, src_top, src_right, src_bottom))
        canvas.paste(patch, (src_left - left, src_top - top))
    return np.asarray(canvas, dtype=np.float32) / 255.0


def pooled_mean(arr: np.ndarray, grid: int) -> np.ndarray:
    height, width = arr.shape[:2]
    trimmed = arr[: height - height % grid, : width - width % grid]
    if trimmed.size == 0:
        return np.zeros(grid * grid * (arr.shape[2] if arr.ndim == 3 else 1), dtype=np.float32)
    cell_h = trimmed.shape[0] // grid
    cell_w = trimmed.shape[1] // grid
    if arr.ndim == 3:
        channels = arr.shape[2]
        pooled = trimmed.reshape(grid, cell_h, grid, cell_w, channels).mean(axis=(1, 3))
    else:
        pooled = trimmed.reshape(grid, cell_h, grid, cell_w).mean(axis=(1, 3))
    return pooled.reshape(-1).astype(np.float32)


def visual_patch_feature(image_path: str, x: float, y: float, crop_size: int) -> np.ndarray:
    patch = crop_contact_patch(image_path, x, y, crop_size)
    gray = (
        0.299 * patch[:, :, 0]
        + 0.587 * patch[:, :, 1]
        + 0.114 * patch[:, :, 2]
    ).astype(np.float32)

    gx = np.diff(gray, axis=1, append=gray[:, -1:])
    gy = np.diff(gray, axis=0, append=gray[-1:, :])
    edge = np.sqrt(gx * gx + gy * gy).astype(np.float32)

    center_margin = max(1, crop_size // 4)
    center = patch[center_margin:-center_margin, center_margin:-center_margin]
    border = patch.copy()
    border[center_margin:-center_margin, center_margin:-center_margin] = 0.0
    border_mask = np.ones(gray.shape, dtype=bool)
    border_mask[center_margin:-center_margin, center_margin:-center_margin] = False

    center_mean = center.reshape(-1, 3).mean(axis=0) if center.size else np.zeros(3, dtype=np.float32)
    border_mean = patch[border_mask].reshape(-1, 3).mean(axis=0) if border_mask.any() else np.zeros(3, dtype=np.float32)
    edge_hist, _ = np.histogram(edge, bins=np.asarray([0.0, 0.02, 0.05, 0.1, 0.2, 0.4, 2.0]))
    edge_hist = edge_hist.astype(np.float32)
    edge_hist = edge_hist / max(float(edge_hist.sum()), 1.0)

    stats = np.concatenate(
        [
            patch.reshape(-1, 3).mean(axis=0),
            patch.reshape(-1, 3).std(axis=0),
            np.asarray(
                [
                    float(gray.mean()),
                    float(gray.std()),
                    float(edge.mean()),
                    float(edge.std()),
                    float(np.quantile(edge, 0.75)),
                    float(np.quantile(edge, 0.90)),
                ],
                dtype=np.float32,
            ),
            center_mean.astype(np.float32),
            border_mean.astype(np.float32),
            (center_mean - border_mean).astype(np.float32),
            edge_hist,
        ],
        axis=0,
    )
    layout = np.concatenate([pooled_mean(patch, 4), pooled_mean(edge, 4)], axis=0)
    return np.concatenate([stats.astype(np.float32), layout], axis=0)


def motion_geometry_feature(row: dict[str, str], x: float, y: float) -> np.ndarray:
    width = float(row["image_width"])
    height = float(row["image_height"])
    denom = max(width, height)
    tip_x = float(row["tip_x"])
    tip_y = float(row["tip_y"])
    base_x = float(row["base_x"])
    base_y = float(row["base_y"])
    probe = max(float(row["probe"]), 1.0)
    sensor_dx = tip_x - base_x
    sensor_dy = tip_y - base_y
    rel_tip_x = (x - tip_x) / width
    rel_tip_y = (y - tip_y) / height
    distance_from_tip = math.hypot(x - tip_x, y - tip_y) / denom
    contact_frame = float(row.get("contact_frame_from_name") or row.get("contact_frame_detected") or row.get("contact_frame") or 1.0)

    return np.asarray(
        [
            x / width,
            y / height,
            tip_x / width,
            tip_y / height,
            base_x / width,
            base_y / height,
            rel_tip_x,
            rel_tip_y,
            float(row["direction_x"]),
            float(row["direction_y"]),
            float(row["probe"]) / 100.0,
            float(row["frame_id"]) / max(contact_frame, 1.0),
            sensor_dx / width,
            sensor_dy / height,
            math.hypot(sensor_dx, sensor_dy) / denom,
            distance_from_tip,
            distance_from_tip / probe,
        ],
        dtype=np.float32,
    )


def standardize(matrix: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = reference.mean(axis=0)
    std = reference.std(axis=0)
    std[std < 1e-6] = 1.0
    return (matrix - mean) / std, mean, std


def apply_standardize(matrix: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (matrix - mean) / std


def parse_float(value: str) -> float:
    return float(value) if value != "" else 0.0


def make_item(pred: dict[str, str], source: dict[str, str], x: float, y: float, crop_size: int) -> dict:
    return {
        "pred": pred,
        "source": source,
        "x": x,
        "y": y,
        "motion": motion_geometry_feature(source, x, y),
        "visual": visual_patch_feature(source["vision_path"], x, y, crop_size),
    }


def build_items(
    predictions: list[dict[str, str]],
    rows_by_name: dict[str, dict[str, str]],
    crop_size: int,
) -> tuple[list[dict], list[dict]]:
    cache_items: list[dict] = []
    query_items: list[dict] = []
    for pred in predictions:
        source = rows_by_name[pred["image_name"]]
        if pred["dataset_split"] == "train":
            x = float(source["target_tip_x"])
            y = float(source["target_tip_y"])
            cache_items.append(make_item(pred, source, x, y, crop_size))
        elif pred["dataset_split"] in {"val", "test"}:
            x = float(pred["pred_x"])
            y = float(pred["pred_y"])
            query_items.append(make_item(pred, source, x, y, crop_size))
    return cache_items, query_items


def combine_features(
    cache_items: list[dict],
    query_items: list[dict],
    mode: str,
    motion_weight: float,
    visual_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    cache_motion = np.stack([item["motion"] for item in cache_items], axis=0)
    query_motion = np.stack([item["motion"] for item in query_items], axis=0)
    cache_motion_z, motion_mean, motion_std = standardize(cache_motion, cache_motion)
    query_motion_z = apply_standardize(query_motion, motion_mean, motion_std)
    motion_scale = motion_weight / math.sqrt(cache_motion_z.shape[1])

    if mode == "direct":
        return cache_motion_z * motion_scale, query_motion_z * motion_scale, cache_motion_z, query_motion_z

    cache_visual = np.stack([item["visual"] for item in cache_items], axis=0)
    query_visual = np.stack([item["visual"] for item in query_items], axis=0)
    cache_visual_z, visual_mean, visual_std = standardize(cache_visual, cache_visual)
    query_visual_z = apply_standardize(query_visual, visual_mean, visual_std)
    visual_scale = visual_weight / math.sqrt(cache_visual_z.shape[1])
    cache_key = np.concatenate([cache_motion_z * motion_scale, cache_visual_z * visual_scale], axis=1)
    query_key = np.concatenate([query_motion_z * motion_scale, query_visual_z * visual_scale], axis=1)
    return cache_key, query_key, cache_visual_z, query_visual_z


def direction_cosine(query: dict[str, str], retrieved: dict[str, str]) -> float:
    qx = float(query["direction_x"])
    qy = float(query["direction_y"])
    rx = float(retrieved["direction_x"])
    ry = float(retrieved["direction_y"])
    denom = max(math.hypot(qx, qy) * math.hypot(rx, ry), 1e-8)
    return (qx * rx + qy * ry) / denom


def save_retrieval_debug(row: dict[str, str], output_path: Path, box_size: int) -> None:
    query = Image.open(row["query_vision_path"]).convert("RGB")
    retrieved = Image.open(row["retrieved_vision_path"]).convert("RGB")
    touch_path = row["retrieved_touch_path"]
    if touch_path and Path(touch_path).exists():
        touch = Image.open(touch_path).convert("RGB").resize(query.size)
    else:
        touch = Image.new("RGB", query.size, "black")

    query_draw = ImageDraw.Draw(query)
    draw_box(query_draw, float(row["query_pred_x"]), float(row["query_pred_y"]), box_size, "magenta", 4)
    draw_box(query_draw, float(row["query_target_x"]), float(row["query_target_y"]), box_size, "lime", 3)

    retrieved_draw = ImageDraw.Draw(retrieved)
    draw_box(retrieved_draw, float(row["retrieved_target_x"]), float(row["retrieved_target_y"]), box_size, "lime", 4)

    canvas = Image.new("RGB", (query.width * 3, query.height), "black")
    canvas.paste(query, (0, 0))
    canvas.paste(retrieved, (query.width, 0))
    canvas.paste(touch, (query.width * 2, 0))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, canvas.width, 32), fill="black")
    draw.text(
        (8, 9),
        (
            f"{row['mode']} cache | query pred/gt | retrieved train | touch "
            f"dist={float(row['distance']):.4f}"
        ),
        fill="white",
    )
    ensure_dir(output_path.parent)
    canvas.save(output_path)


def build_retrieval_mode(
    mode: str,
    cache_items: list[dict],
    query_items: list[dict],
    output_csv: Path,
    output_json: Path,
    debug_dir: Path,
    debug_samples: int,
    box_size: int,
    motion_weight: float,
    visual_weight: float,
) -> tuple[dict, list[dict[str, str]]]:
    if not cache_items or not query_items:
        write_csv_rows(output_csv, [], RETRIEVAL_FIELDS)
        summary = {"mode": mode, "cache_size": len(cache_items), "queries": len(query_items)}
        write_json(output_json, summary)
        return summary, []

    cache_key, query_key, cache_group, query_group = combine_features(
        cache_items,
        query_items,
        mode,
        motion_weight,
        visual_weight,
    )

    retrieval_rows: list[dict[str, str]] = []
    distances: list[float] = []
    motion_distances: list[float] = []
    visual_distances: list[float] = []
    probe_deltas: list[float] = []
    direction_cosines: list[float] = []
    gt_retrieved_errors: list[float] = []
    same_record_hits: list[bool] = []

    for query_idx, item in enumerate(query_items):
        dists = np.linalg.norm(cache_key - query_key[query_idx][None, :], axis=1)
        best_idx = int(np.argmin(dists))
        best_item = cache_items[best_idx]
        pred = item["pred"]
        source = item["source"]
        best_pred = best_item["pred"]
        best_source = best_item["source"]

        motion_distance = float(
            np.linalg.norm(
                motion_geometry_feature(best_source, best_item["x"], best_item["y"])
                - motion_geometry_feature(source, item["x"], item["y"])
            )
        )
        if mode == "direct" or cache_group is None or query_group is None:
            visual_distance_text = ""
        else:
            visual_distance = float(np.linalg.norm(cache_group[best_idx] - query_group[query_idx]))
            visual_distances.append(visual_distance)
            visual_distance_text = f"{visual_distance:.6f}"

        probe_delta = abs(float(source["probe"]) - float(best_source["probe"]))
        cosine = direction_cosine(source, best_source)
        gt_error = math.hypot(
            float(source["target_tip_x"]) - float(best_source["target_tip_x"]),
            float(source["target_tip_y"]) - float(best_source["target_tip_y"]),
        )
        same_record = source["record_id"] == best_source["record_id"]
        distance = float(dists[best_idx])
        distances.append(distance)
        motion_distances.append(motion_distance)
        probe_deltas.append(probe_delta)
        direction_cosines.append(cosine)
        gt_retrieved_errors.append(gt_error)
        same_record_hits.append(same_record)

        retrieval_rows.append(
            {
                "mode": mode,
                "dataset_split": pred["dataset_split"],
                "query_split": source["split"],
                "query_record_id": source["record_id"],
                "query_image_name": source["image_name"],
                "query_vision_path": source["vision_path"],
                "query_touch_path": source.get("touch_path", ""),
                "query_pred_x": f"{item['x']:.3f}",
                "query_pred_y": f"{item['y']:.3f}",
                "query_target_x": f"{float(source['target_tip_x']):.3f}",
                "query_target_y": f"{float(source['target_tip_y']):.3f}",
                "query_probe": source["probe"],
                "retrieved_split": best_source["split"],
                "retrieved_record_id": best_source["record_id"],
                "retrieved_image_name": best_source["image_name"],
                "retrieved_vision_path": best_source["vision_path"],
                "retrieved_touch_path": best_source.get("touch_path", best_pred.get("touch_path", "")),
                "retrieved_target_x": f"{float(best_source['target_tip_x']):.3f}",
                "retrieved_target_y": f"{float(best_source['target_tip_y']):.3f}",
                "retrieved_probe": best_source["probe"],
                "distance": f"{distance:.6f}",
                "motion_distance": f"{motion_distance:.6f}",
                "visual_distance": visual_distance_text,
                "probe_delta": f"{probe_delta:.3f}",
                "direction_cosine": f"{cosine:.6f}",
                "query_gt_retrieved_gt_error_px": f"{gt_error:.3f}",
                "same_record": "1" if same_record else "0",
            }
        )

    write_csv_rows(output_csv, retrieval_rows, RETRIEVAL_FIELDS)
    for idx, row in enumerate(retrieval_rows[:debug_samples]):
        output_path = debug_dir / f"{idx:03d}_{Path(row['query_image_name']).stem}_{mode}_retrieval.jpg"
        save_retrieval_debug(row, output_path, box_size)

    summary = {
        "mode": mode,
        "cache_size": len(cache_items),
        "queries": len(query_items),
        "feature_dims": int(cache_key.shape[1]),
        "motion_weight": motion_weight,
        "visual_weight": visual_weight if mode == "hybrid" else 0.0,
        "mean_distance": float(np.mean(distances)) if distances else None,
        "median_distance": float(np.median(distances)) if distances else None,
        "mean_motion_distance_raw": float(np.mean(motion_distances)) if motion_distances else None,
        "mean_visual_distance_z": float(np.mean(visual_distances)) if visual_distances else None,
        "mean_probe_delta": float(np.mean(probe_deltas)) if probe_deltas else None,
        "probe_match_rate": float(np.mean([delta == 0.0 for delta in probe_deltas])) if probe_deltas else None,
        "mean_direction_cosine": float(np.mean(direction_cosines)) if direction_cosines else None,
        "median_query_gt_retrieved_gt_error_px": float(np.median(gt_retrieved_errors)) if gt_retrieved_errors else None,
        "same_record_rate": float(np.mean(same_record_hits)) if same_record_hits else None,
        "output_csv": str(output_csv),
        "debug_dir": str(debug_dir),
    }
    write_json(output_json, summary)
    return summary, retrieval_rows


def compare_modes(
    direct_rows: list[dict[str, str]],
    hybrid_rows: list[dict[str, str]],
    output_json: Path,
    output_csv: Path,
) -> dict:
    hybrid_by_query = {row["query_image_name"]: row for row in hybrid_rows}
    comparison_rows = []
    changed = []
    for direct in direct_rows:
        hybrid = hybrid_by_query.get(direct["query_image_name"])
        if hybrid is None:
            continue
        is_changed = direct["retrieved_image_name"] != hybrid["retrieved_image_name"]
        changed.append(is_changed)
        comparison_rows.append(
            {
                "query_image_name": direct["query_image_name"],
                "query_record_id": direct["query_record_id"],
                "query_probe": direct["query_probe"],
                "direct_retrieved_image_name": direct["retrieved_image_name"],
                "direct_distance": direct["distance"],
                "hybrid_retrieved_image_name": hybrid["retrieved_image_name"],
                "hybrid_distance": hybrid["distance"],
                "changed": "1" if is_changed else "0",
            }
        )
    write_csv_rows(output_csv, comparison_rows, COMPARISON_FIELDS)
    summary = {
        "queries": len(comparison_rows),
        "retrieved_image_agreement_rate": float(1.0 - np.mean(changed)) if changed else None,
        "hybrid_changed_queries": int(np.sum(changed)) if changed else 0,
        "comparison_csv": str(output_csv),
    }
    write_json(output_json, summary)
    return summary


def build_cache_retrieval(
    config_path: str,
    debug_samples_override: int | None = None,
    motion_weight_override: float | None = None,
    visual_weight_override: float | None = None,
    section: str = "contact_region",
) -> dict:
    cfg = load_config(config_path)
    if section not in cfg:
        raise KeyError(f"Missing config section: {section}")
    region_cfg = cfg[section]
    rows = read_csv_rows(project_path(region_cfg["samples_csv"]))
    attach_touch_paths(rows, project_path(cfg["manifest"]["output_csv"]))
    rows_by_name = {row["image_name"]: row for row in rows}
    predictions = read_csv_rows(project_path(region_cfg["predictions_csv"]))

    crop_size = int(region_cfg.get("cache_crop_size", region_cfg.get("contact_box_size", 48)))
    box_size = int(region_cfg.get("contact_box_size", 48))
    debug_samples = int(debug_samples_override or region_cfg.get("debug_samples", 30))
    motion_weight = float(
        motion_weight_override
        if motion_weight_override is not None
        else region_cfg.get("cache_motion_weight", 0.8)
    )
    visual_weight = float(
        visual_weight_override
        if visual_weight_override is not None
        else region_cfg.get("cache_visual_weight", 1.0)
    )
    cache_items, query_items = build_items(predictions, rows_by_name, crop_size)

    direct_summary, direct_rows = build_retrieval_mode(
        "direct",
        cache_items,
        query_items,
        project_path(region_cfg.get("retrieval_direct_csv", "outputs/metrics/contact_region_retrieval_direct.csv")),
        project_path(region_cfg.get("retrieval_direct_json", "outputs/metrics/contact_region_retrieval_direct.json")),
        project_path(region_cfg.get("retrieval_direct_debug_dir", "outputs/debug/phase2/retrieval_direct")),
        debug_samples,
        box_size,
        motion_weight,
        visual_weight,
    )
    hybrid_summary, hybrid_rows = build_retrieval_mode(
        "hybrid",
        cache_items,
        query_items,
        project_path(region_cfg.get("retrieval_hybrid_csv", "outputs/metrics/contact_region_retrieval_hybrid.csv")),
        project_path(region_cfg.get("retrieval_hybrid_json", "outputs/metrics/contact_region_retrieval_hybrid.json")),
        project_path(region_cfg.get("retrieval_hybrid_debug_dir", "outputs/debug/phase2/retrieval_hybrid")),
        debug_samples,
        box_size,
        motion_weight,
        visual_weight,
    )
    comparison = compare_modes(
        direct_rows,
        hybrid_rows,
        project_path(region_cfg.get("retrieval_compare_json", "outputs/metrics/contact_region_retrieval_compare.json")),
        project_path(region_cfg.get("retrieval_compare_csv", "outputs/metrics/contact_region_retrieval_compare.csv")),
    )
    summary = {
        "config_section": section,
        "cache_crop_size": crop_size,
        "contact_box_size": box_size,
        "direct": direct_summary,
        "hybrid": hybrid_summary,
        "comparison": comparison,
    }
    print(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build direct and hybrid train-cache retrieval outputs.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--section", default="contact_region")
    parser.add_argument("--debug-samples", type=int, default=None)
    parser.add_argument("--motion-weight", type=float, default=None)
    parser.add_argument("--visual-weight", type=float, default=None)
    args = parser.parse_args()
    build_cache_retrieval(args.config, args.debug_samples, args.motion_weight, args.visual_weight, args.section)


if __name__ == "__main__":
    main()
