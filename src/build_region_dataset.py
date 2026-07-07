from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .config import load_config, project_path
from .utils import (
    ensure_dir,
    read_csv_rows,
    save_label_overlay,
    write_csv_rows,
    write_json,
)


FIELDS = [
    "dataset_split",
    "split",
    "record_id",
    "image_name",
    "image_path",
    "vision_path",
    "probe",
    "frame_id",
    "contact_frame_from_name",
    "contact_frame_detected",
    "tip_x",
    "tip_y",
    "base_x",
    "base_y",
    "direction_x",
    "direction_y",
    "target_tip_x",
    "target_tip_y",
    "image_width",
    "image_height",
    "heatmap_path",
]


def _record_splits(records: list[tuple[str, str]], train: float, val: float) -> dict[tuple[str, str], str]:
    records = sorted(records)
    n = len(records)
    train_end = int(round(n * train))
    val_end = train_end + int(round(n * val))
    result = {}
    for idx, record in enumerate(records):
        if idx < train_end:
            result[record] = "train"
        elif idx < val_end:
            result[record] = "val"
        else:
            result[record] = "test"
    return result


def _make_heatmap(width: int, height: int, x: float, y: float, sigma: float) -> np.ndarray:
    xs = np.arange(width, dtype=np.float32)
    ys = np.arange(height, dtype=np.float32)[:, None]
    heatmap = np.exp(-((xs - x) ** 2 + (ys - y) ** 2) / (2.0 * sigma * sigma))
    return heatmap.astype(np.float32)


def _save_heatmap_preview(path: Path, heatmap: np.ndarray) -> None:
    arr = np.clip(heatmap / max(float(heatmap.max()), 1e-8) * 255.0, 0, 255).astype(np.uint8)
    image = Image.fromarray(arr, mode="L")
    ensure_dir(path.parent)
    image.save(path)


def build_region_dataset(config_path: str) -> dict:
    cfg = load_config(config_path)
    localizer_cfg = cfg["sensor_localizer"]
    region_cfg = cfg["region_dataset"]
    manifest_csv = project_path(cfg["manifest"]["output_csv"])
    contact_csv = project_path(cfg["contact_detection"]["output_csv"])
    sensor_labels_csv = project_path(localizer_cfg["labels_output_csv"])
    sensor_tracks_csv = project_path(localizer_cfg["tracks_output_csv"])
    output_csv = project_path(region_cfg["output_csv"])
    heatmap_dir = project_path(region_cfg["heatmap_dir"])
    debug_dir = project_path(region_cfg["debug_dir"])
    debug_samples = int(region_cfg.get("debug_samples", 0))
    heatmap_w = int(region_cfg["heatmap_width"])
    heatmap_h = int(region_cfg["heatmap_height"])
    sigma = float(region_cfg["gaussian_sigma"])

    manifest_by_key = {}
    if manifest_csv.exists():
        for row in read_csv_rows(manifest_csv):
            manifest_by_key[(row["split"], row["record_id"], int(row["frame_id"]))] = row

    contact_by_record = {}
    if contact_csv.exists():
        for row in read_csv_rows(contact_csv):
            if row["contact_frame"]:
                contact_by_record[(row["split"], row["record_id"])] = int(row["contact_frame"])

    track_by_key = {}
    for row in read_csv_rows(sensor_tracks_csv):
        track_by_key[(row["split"], row["record_id"], int(row["frame_id"]))] = row

    label_rows = read_csv_rows(sensor_labels_csv)
    record_keys = sorted({(row["split"], row["record_id"]) for row in label_rows})
    split_map = _record_splits(
        record_keys,
        float(region_cfg.get("split_train", 0.8)),
        float(region_cfg.get("split_val", 0.1)),
    )

    output_rows: list[dict] = []
    for idx, row in enumerate(label_rows):
        frame_id = int(row["frame_id"])
        key = (row["split"], row["record_id"], frame_id)
        track = track_by_key.get(key)
        if track is None:
            continue
        image_w = int(float(row["image_width"]))
        image_h = int(float(row["image_height"]))
        target_x = float(track["target_tip_x"])
        target_y = float(track["target_tip_y"])
        hm_x = target_x / image_w * heatmap_w
        hm_y = target_y / image_h * heatmap_h
        heatmap = _make_heatmap(heatmap_w, heatmap_h, hm_x, hm_y, sigma)
        heatmap_path = heatmap_dir / f"{Path(row['image_name']).stem}.npy"
        ensure_dir(heatmap_path.parent)
        np.save(heatmap_path, heatmap)

        if idx < debug_samples:
            overlay_path = debug_dir / "overlays" / f"{idx:03d}_{Path(row['image_name']).stem}.jpg"
            save_label_overlay(
                row["image_path"],
                overlay_path,
                (float(row["tip_x"]), float(row["tip_y"])),
                (float(row["base_x"]), float(row["base_y"])),
                (target_x, target_y),
            )
            heatmap_preview = debug_dir / "heatmaps" / f"{idx:03d}_{Path(row['image_name']).stem}_heatmap.jpg"
            _save_heatmap_preview(heatmap_preview, heatmap)

        manifest_row = manifest_by_key.get(key)
        output_rows.append(
            {
                "dataset_split": split_map[(row["split"], row["record_id"])],
                "split": row["split"],
                "record_id": row["record_id"],
                "image_name": row["image_name"],
                "image_path": row["image_path"],
                "vision_path": manifest_row["vision_path"] if manifest_row else "",
                "probe": row["probe"],
                "frame_id": frame_id,
                "contact_frame_from_name": row["contact_frame_from_name"],
                "contact_frame_detected": contact_by_record.get((row["split"], row["record_id"]), ""),
                "tip_x": row["tip_x"],
                "tip_y": row["tip_y"],
                "base_x": row["base_x"],
                "base_y": row["base_y"],
                "direction_x": row["direction_x"],
                "direction_y": row["direction_y"],
                "target_tip_x": f"{target_x:.3f}",
                "target_tip_y": f"{target_y:.3f}",
                "image_width": image_w,
                "image_height": image_h,
                "heatmap_path": str(heatmap_path),
            }
        )

    write_csv_rows(output_csv, output_rows, FIELDS)
    counts = defaultdict(int)
    for row in output_rows:
        counts[row["dataset_split"]] += 1
    contact_errors = []
    for split, record_id in record_keys:
        detected = contact_by_record.get((split, record_id))
        name_contacts = {
            int(row["contact_frame_from_name"])
            for row in label_rows
            if row["split"] == split and row["record_id"] == record_id
        }
        if detected is not None and len(name_contacts) == 1:
            name_contact = next(iter(name_contacts))
            contact_errors.append(
                {
                    "split": split,
                    "record_id": record_id,
                    "detected_contact_frame": detected,
                    "name_contact_frame": name_contact,
                    "error_frames": detected - name_contact,
                    "abs_error_frames": abs(detected - name_contact),
                }
            )
    abs_errors = [item["abs_error_frames"] for item in contact_errors]
    summary = {
        "samples": len(output_rows),
        "records": len(record_keys),
        "split_counts": dict(counts),
        "contact_detection_comparison": {
            "records": len(contact_errors),
            "mean_abs_error_frames": float(np.mean(abs_errors)) if abs_errors else None,
            "median_abs_error_frames": float(np.median(abs_errors)) if abs_errors else None,
            "max_abs_error_frames": int(max(abs_errors)) if abs_errors else None,
            "outliers_abs_error_gt_10": [
                item for item in contact_errors if item["abs_error_frames"] > 10
            ],
        },
        "output_csv": str(output_csv),
        "heatmap_dir": str(heatmap_dir),
    }
    write_json(output_csv.parent / "region_dataset_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Phase 1 contact region dataset from sensor labels.")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    summary = build_region_dataset(args.config)
    print(summary)


if __name__ == "__main__":
    main()
