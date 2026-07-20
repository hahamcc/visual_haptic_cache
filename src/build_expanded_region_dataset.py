from __future__ import annotations

import argparse
import csv
import math
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from .audit_dataset_expansion import detect_contact_for_record, frame_map
from .config import load_config, project_path
from .train_sensor_localizer import TinyUNet, peak_xy
from .utils import ensure_dir, normalized_vector, read_csv_rows, write_csv_rows, write_json


SAMPLE_FIELDS = [
    "dataset_split",
    "record_partition",
    "split",
    "record_id",
    "image_name",
    "image_path",
    "vision_path",
    "touch_path",
    "probe",
    "frame_id",
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
    "tip_confidence",
    "base_confidence",
    "target_tip_confidence",
    "target_base_confidence",
    "tip_base_distance",
    "target_tip_base_distance",
    "sequence_ready",
    "trajectory_real_point_count",
    "trajectory_history_span_frames",
    "trajectory_padding_ratio",
    "trajectory_max_frame_gap",
    "trajectory_cumulative_displacement",
    "heatmap_path",
]

RECORD_SPLIT_FIELDS = ["split", "record_id", "dataset_split", "record_partition", "split_seed"]

TRACK_FIELDS = [
    "split",
    "record_id",
    "frame_id",
    "vision_path",
    "tip_x",
    "tip_y",
    "base_x",
    "base_y",
    "direction_x",
    "direction_y",
    "tip_confidence",
    "base_confidence",
    "tip_base_distance",
    "image_width",
    "image_height",
]

CONTACT_FIELDS = [
    "split",
    "record_id",
    "status",
    "contact_frame",
    "contact_score",
    "threshold",
    "max_score",
    "baseline_mean",
    "baseline_std",
    "common_frames",
]

SKIPPED_FIELDS = [
    "split",
    "record_id",
    "frame_id",
    "probe",
    "reason",
    "detail",
]


def record_splits(records: list[tuple[str, str]], train: float, val: float) -> dict[tuple[str, str], str]:
    records = sorted(records)
    train_end = int(round(len(records) * train))
    val_end = train_end + int(round(len(records) * val))
    out = {}
    for idx, record in enumerate(records):
        if idx < train_end:
            out[record] = "train"
        elif idx < val_end:
            out[record] = "val"
        else:
            out[record] = "test"
    return out


def fixed_record_splits(
    records: list[tuple[str, str]],
    train: float,
    val: float,
    seed: int,
    manifest_path: Path | None,
    final_holdout_count: int | None = None,
    final_holdout_min_record: int | None = None,
) -> tuple[dict[tuple[str, str], str], dict[tuple[str, str], str]]:
    """Create or reuse a predeclared record split before any labels are generated."""
    records = sorted(records)
    if manifest_path and manifest_path.exists():
        with manifest_path.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        mapping = {(row["split"], row["record_id"]): row["dataset_split"] for row in rows}
        partitions = {(row["split"], row["record_id"]): row.get("record_partition", row["dataset_split"]) for row in rows}
        missing = [record for record in records if record not in mapping]
        if missing:
            raise ValueError(f"Fixed record manifest is missing selected records: {missing[:5]}")
        return mapping, partitions

    rng = random.Random(seed)
    total = len(records)
    train_count = int(round(total * train))
    val_count = int(round(total * val))
    requested_holdout = int(final_holdout_count) if final_holdout_count is not None else total - train_count - val_count
    if train_count + val_count + requested_holdout > total:
        raise ValueError("Configured record split counts exceed selected records")
    if final_holdout_min_record is None:
        holdout_candidates = list(records)
    else:
        holdout_candidates = []
        for record in records:
            match = re.search(r"(\d+)$", record[1])
            if match and int(match.group(1)) >= final_holdout_min_record:
                holdout_candidates.append(record)
    if len(holdout_candidates) < requested_holdout:
        raise ValueError("Not enough eligible records for the requested fixed final holdout")
    rng.shuffle(holdout_candidates)
    holdout_records = set(holdout_candidates[:requested_holdout])
    shuffled = [record for record in records if record not in holdout_records]
    rng.shuffle(shuffled)
    mapping: dict[tuple[str, str], str] = {}
    partitions: dict[tuple[str, str], str] = {}
    manifest_rows = []
    for index, record in enumerate(shuffled):
        if index < train_count:
            dataset_split, partition = "train", "train"
        elif index < train_count + val_count:
            dataset_split, partition = "val", "validation"
        else:
            raise RuntimeError("Unexpected leftover record after fixed split allocation")
        mapping[record] = dataset_split
        partitions[record] = partition
        manifest_rows.append(
            {
                "split": record[0], "record_id": record[1], "dataset_split": dataset_split,
                "record_partition": partition, "split_seed": str(seed),
            }
        )
    for record in sorted(holdout_records):
        mapping[record] = "test"
        partitions[record] = "final_holdout"
        manifest_rows.append(
            {
                "split": record[0], "record_id": record[1], "dataset_split": "test",
                "record_partition": "final_holdout", "split_seed": str(seed),
            }
        )
    if manifest_path:
        write_csv_rows(manifest_path, manifest_rows, RECORD_SPLIT_FIELDS)
    return mapping, partitions


def make_heatmap(width: int, height: int, x: float, y: float, sigma: float) -> np.ndarray:
    xs = np.arange(width, dtype=np.float32)
    ys = np.arange(height, dtype=np.float32)[:, None]
    heatmap = np.exp(-((xs - x) ** 2 + (ys - y) ** 2) / (2.0 * sigma * sigma))
    return heatmap.astype(np.float32)


def parse_record_range(start: int, limit: int | None) -> tuple[int, int | None]:
    return start, None if limit is None or limit <= 0 else start + limit


def select_records(vision_split_root: Path, start: int, limit: int | None) -> list[str]:
    record_ids = sorted(path.name for path in vision_split_root.iterdir() if path.is_dir())
    begin, end = parse_record_range(start, limit)
    return record_ids[begin:end]


def validate_purpose_partition(manifest_path: Path, record_ids: list[str], expected_partition: str) -> None:
    """Prevent a data-building section from accidentally consuming a sealed record range."""
    rows = read_csv_rows(manifest_path)
    partition_by_record = {row["record_id"]: row["partition"] for row in rows}
    missing = [record_id for record_id in record_ids if record_id not in partition_by_record]
    mismatched = [record_id for record_id in record_ids if partition_by_record.get(record_id) != expected_partition]
    if missing or mismatched:
        raise ValueError(
            f"Purpose-partition validation failed for {manifest_path}: missing={missing}, "
            f"expected={expected_partition}, mismatched={mismatched}"
        )


def load_sensor_model(checkpoint_path: Path, device: torch.device) -> tuple[TinyUNet, dict]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing sensor localizer checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_cfg = checkpoint.get("config", {})
    model = TinyUNet().to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, model_cfg


def predict_sensor_points(
    model: TinyUNet,
    image_path: Path,
    input_width: int,
    input_height: int,
    device: torch.device,
) -> dict[str, float | int | str]:
    image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size
    resized = image.resize((input_width, input_height), Image.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))[None, :, :, :]
    tensor = torch.from_numpy(arr).to(device)
    with torch.no_grad():
        pred = model(tensor)[0].detach().cpu()
    tip_x_small, tip_y_small = peak_xy(pred[0])
    base_x_small, base_y_small = peak_xy(pred[1])
    tip_score = float(pred[0, int(tip_y_small), int(tip_x_small)].item())
    base_score = float(pred[1, int(base_y_small), int(base_x_small)].item())
    tip_x = tip_x_small / input_width * orig_w
    tip_y = tip_y_small / input_height * orig_h
    base_x = base_x_small / input_width * orig_w
    base_y = base_y_small / input_height * orig_h
    direction_x, direction_y = normalized_vector(tip_x - base_x, tip_y - base_y)
    tip_base_distance = math.hypot(tip_x - base_x, tip_y - base_y)
    return {
        "tip_x": tip_x,
        "tip_y": tip_y,
        "base_x": base_x,
        "base_y": base_y,
        "direction_x": direction_x,
        "direction_y": direction_y,
        "tip_confidence": tip_score,
        "base_confidence": base_score,
        "tip_base_distance": tip_base_distance,
        "image_width": orig_w,
        "image_height": orig_h,
        "vision_path": str(image_path),
    }


def format_track(split: str, record_id: str, frame_id: int, pred: dict[str, float | int | str]) -> dict[str, str]:
    return {
        "split": split,
        "record_id": record_id,
        "frame_id": str(frame_id),
        "vision_path": str(pred["vision_path"]),
        "tip_x": f"{float(pred['tip_x']):.3f}",
        "tip_y": f"{float(pred['tip_y']):.3f}",
        "base_x": f"{float(pred['base_x']):.3f}",
        "base_y": f"{float(pred['base_y']):.3f}",
        "direction_x": f"{float(pred['direction_x']):.6f}",
        "direction_y": f"{float(pred['direction_y']):.6f}",
        "tip_confidence": f"{float(pred['tip_confidence']):.6f}",
        "base_confidence": f"{float(pred['base_confidence']):.6f}",
        "tip_base_distance": f"{float(pred['tip_base_distance']):.3f}",
        "image_width": str(int(pred["image_width"])),
        "image_height": str(int(pred["image_height"])),
    }


def box_inside_image(x: float, y: float, box_size: int, width: int, height: int) -> bool:
    half = box_size / 2.0
    return x - half >= 0 and y - half >= 0 and x + half <= width and y + half <= height


def quality_reason(
    current: dict[str, float | int | str],
    target: dict[str, float | int | str],
    min_confidence: float,
    min_tip_base_distance: float,
    max_tip_base_distance: float,
    contact_box_size: int,
) -> tuple[bool, str, str]:
    for prefix, pred in (("current", current), ("target", target)):
        if float(pred["tip_confidence"]) < min_confidence:
            return False, f"{prefix}_tip_low_confidence", f"{float(pred['tip_confidence']):.6f}"
        if float(pred["base_confidence"]) < min_confidence:
            return False, f"{prefix}_base_low_confidence", f"{float(pred['base_confidence']):.6f}"
        distance = float(pred["tip_base_distance"])
        if distance < min_tip_base_distance or distance > max_tip_base_distance:
            return False, f"{prefix}_tip_base_distance_out_of_range", f"{distance:.3f}"
    if not box_inside_image(
        float(target["tip_x"]),
        float(target["tip_y"]),
        contact_box_size,
        int(target["image_width"]),
        int(target["image_height"]),
    ):
        return False, "target_box_outside_image", f"{target['tip_x']},{target['tip_y']}"
    return True, "", ""


def draw_box(draw: ImageDraw.ImageDraw, x: float, y: float, box_size: int, color: str, width: int = 3) -> None:
    half = box_size / 2.0
    draw.rectangle((x - half, y - half, x + half, y + half), outline=color, width=width)


def save_debug_overlay(
    output_path: Path,
    current_path: Path,
    current: dict[str, float | int | str],
    target: dict[str, float | int | str],
    contact_box_size: int,
    title: str,
) -> None:
    image = Image.open(current_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    tip = (float(current["tip_x"]), float(current["tip_y"]))
    base = (float(current["base_x"]), float(current["base_y"]))
    target_tip = (float(target["tip_x"]), float(target["tip_y"]))
    draw.line((base[0], base[1], tip[0], tip[1]), fill="yellow", width=3)
    draw.ellipse((tip[0] - 5, tip[1] - 5, tip[0] + 5, tip[1] + 5), outline="magenta", width=3)
    draw.ellipse((base[0] - 5, base[1] - 5, base[0] + 5, base[1] + 5), outline="cyan", width=3)
    draw_box(draw, target_tip[0], target_tip[1], contact_box_size, "lime", 4)
    draw.line((tip[0], tip[1], target_tip[0], target_tip[1]), fill="white", width=2)
    draw.rectangle((0, 0, image.width, 28), fill="black")
    draw.text((8, 8), title, fill="white")
    ensure_dir(output_path.parent)
    image.save(output_path)


def detect_contact_rows(
    split: str,
    record_id: str,
    touch_by_frame: dict[int, Path],
    common_frames: set[int],
    cfg: dict,
    audit_cfg: dict,
) -> tuple[dict[str, str], int | None]:
    result = detect_contact_for_record(
        {frame: touch_by_frame[frame] for frame in sorted(common_frames)},
        (int(audit_cfg.get("contact_resize_width", 48)), int(audit_cfg.get("contact_resize_height", 48))),
        int(cfg["contact_detection"]["baseline_frames"]),
        int(cfg["contact_detection"]["min_frame"]),
        float(cfg["contact_detection"]["threshold_abs"]),
        float(cfg["contact_detection"]["threshold_std_factor"]),
        float(cfg["contact_detection"]["threshold_peak_ratio"]),
        int(cfg["contact_detection"]["consecutive_frames"]),
    )
    contact_frame = result.get("contact_frame")
    row = {
        "split": split,
        "record_id": record_id,
        "status": str(result["status"]),
        "contact_frame": str(contact_frame if contact_frame is not None else ""),
        "contact_score": f"{float(result.get('contact_score', 0.0)):.6f}" if contact_frame is not None else "",
        "threshold": f"{float(result.get('threshold', 0.0)):.6f}" if "threshold" in result else "",
        "max_score": f"{float(result.get('max_score', 0.0)):.6f}" if "max_score" in result else "",
        "baseline_mean": f"{float(result.get('baseline_mean', 0.0)):.6f}" if "baseline_mean" in result else "",
        "baseline_std": f"{float(result.get('baseline_std', 0.0)):.6f}" if "baseline_std" in result else "",
        "common_frames": str(len(common_frames)),
    }
    return row, int(contact_frame) if contact_frame is not None else None


def build_expanded_region_dataset(
    config_path: str,
    split_override: str | None = None,
    record_start_override: int | None = None,
    record_limit_override: int | None = None,
    section: str = "expanded_region_dataset",
) -> dict:
    cfg = load_config(config_path)
    dataset_cfg = cfg["dataset"]
    if section not in cfg:
        raise KeyError(f"Missing config section: {section}")
    expansion_cfg = cfg[section]
    audit_cfg = cfg.get("dataset_expansion_audit", {})
    sensor_model_cfg = cfg["sensor_localizer"]["model"]
    region_cfg = cfg["region_dataset"]

    root = Path(dataset_cfg["root"])
    split = str(split_override or expansion_cfg.get("split", dataset_cfg.get("split", "0")))
    record_start = int(record_start_override if record_start_override is not None else expansion_cfg.get("record_start", 0))
    record_limit = int(record_limit_override if record_limit_override is not None else expansion_cfg.get("record_limit", 50))
    exts = list(dataset_cfg["image_exts"])
    vision_root = root / dataset_cfg["vision_name"] / split
    touch_root = root / dataset_cfg["touch_name"] / split
    selected_records = select_records(vision_root, record_start, record_limit)
    purpose_manifest_value = expansion_cfg.get("purpose_partition_manifest")
    if purpose_manifest_value:
        validate_purpose_partition(
            project_path(purpose_manifest_value), selected_records, str(expansion_cfg["expected_purpose_partition"])
        )
    excluded_records = {str(item) for item in expansion_cfg.get("excluded_records", [])}
    planned_records = [(split, record_id) for record_id in selected_records if record_id not in excluded_records]
    manifest_value = expansion_cfg.get("record_split_manifest")
    manifest_path = project_path(manifest_value) if manifest_value else None
    planned_split_map, planned_partitions = fixed_record_splits(
        planned_records,
        float(expansion_cfg.get("split_train", region_cfg.get("split_train", 0.8))),
        float(expansion_cfg.get("split_val", region_cfg.get("split_val", 0.1))),
        int(expansion_cfg.get("record_split_seed", 42)),
        manifest_path,
        expansion_cfg.get("final_holdout_count"),
        expansion_cfg.get("final_holdout_min_record"),
    )

    checkpoint_path = project_path(expansion_cfg.get("sensor_checkpoint", f"{sensor_model_cfg['checkpoint_dir']}/best.pt"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint_model_cfg = load_sensor_model(checkpoint_path, device)
    input_width = int(checkpoint_model_cfg.get("input_width", sensor_model_cfg["input_width"]))
    input_height = int(checkpoint_model_cfg.get("input_height", sensor_model_cfg["input_height"]))

    output_csv = project_path(expansion_cfg["output_csv"])
    track_csv = project_path(expansion_cfg["tracks_csv"])
    contact_csv = project_path(expansion_cfg["contact_csv"])
    skipped_csv = project_path(expansion_cfg["skipped_csv"])
    summary_json = project_path(expansion_cfg["summary_json"])
    heatmap_dir = project_path(expansion_cfg["heatmap_dir"])
    debug_dir = project_path(expansion_cfg["debug_dir"])
    debug_samples = int(expansion_cfg.get("debug_samples", 24))

    ttc_values = [int(item) for item in expansion_cfg.get("ttc_values", audit_cfg.get("ttc_values", [5, 10, 20, 30, 50, 75, 100]))]
    sequence_offsets = [int(item) for item in expansion_cfg.get("sequence_offsets", audit_cfg.get("sequence_offsets", [15, 10, 5, 0]))]
    trajectory_history_frames = int(expansion_cfg.get("trajectory_history_frames", max(sequence_offsets)))
    trajectory_track_stride = max(int(expansion_cfg.get("trajectory_track_stride", 5)), 1)
    trajectory_offsets = list(range(0, trajectory_history_frames + 1, trajectory_track_stride))
    heatmap_w = int(region_cfg["heatmap_width"])
    heatmap_h = int(region_cfg["heatmap_height"])
    sigma = float(region_cfg["gaussian_sigma"])
    contact_box_size = int(expansion_cfg.get("contact_box_size", cfg["contact_region"].get("contact_box_size", 48)))
    min_confidence = float(expansion_cfg.get("min_confidence", 0.25))
    min_tip_base_distance = float(expansion_cfg.get("min_tip_base_distance", 6.0))
    max_tip_base_distance = float(expansion_cfg.get("max_tip_base_distance", 45.0))

    start_time = time.time()
    contact_rows: list[dict[str, str]] = []
    track_rows: list[dict[str, str]] = []
    skipped_rows: list[dict[str, str]] = []
    sample_rows: list[dict[str, str]] = []
    predictions: dict[tuple[str, str, int], dict[str, float | int | str]] = {}
    records_with_samples: set[tuple[str, str]] = set()
    debug_written = 0

    for record_id in selected_records:
        if record_id in excluded_records:
            skipped_rows.append(
                {"split": split, "record_id": record_id, "frame_id": "", "probe": "", "reason": "excluded_record", "detail": ""}
            )
            continue
        vision_by_frame = frame_map(vision_root / record_id, exts)
        touch_by_frame = frame_map(touch_root / record_id, exts)
        common_frames = set(vision_by_frame) & set(touch_by_frame)
        if not common_frames:
            skipped_rows.append(
                {"split": split, "record_id": record_id, "frame_id": "", "probe": "", "reason": "no_common_frames", "detail": ""}
            )
            continue

        contact_row, contact_frame = detect_contact_rows(split, record_id, touch_by_frame, common_frames, cfg, audit_cfg)
        contact_rows.append(contact_row)
        if contact_frame is None:
            skipped_rows.append(
                {
                    "split": split,
                    "record_id": record_id,
                    "frame_id": "",
                    "probe": "",
                    "reason": f"contact_{contact_row['status']}",
                    "detail": "",
                }
            )
            continue
        if contact_frame not in common_frames:
            skipped_rows.append(
                {
                    "split": split,
                    "record_id": record_id,
                    "frame_id": str(contact_frame),
                    "probe": "",
                    "reason": "contact_frame_missing_common_frame",
                    "detail": "",
                }
            )
            continue

        needed_frames = {contact_frame}
        for ttc in ttc_values:
            current_frame = contact_frame - ttc
            needed_frames.add(current_frame)
            for offset in sequence_offsets:
                needed_frames.add(current_frame - offset)
            for offset in trajectory_offsets:
                needed_frames.add(current_frame - offset)
        needed_frames = {frame for frame in needed_frames if frame in common_frames}

        for frame in sorted(needed_frames):
            key = (split, record_id, frame)
            pred = predict_sensor_points(model, vision_by_frame[frame], input_width, input_height, device)
            predictions[key] = pred
            track_rows.append(format_track(split, record_id, frame, pred))

        target = predictions.get((split, record_id, contact_frame))
        if target is None:
            continue

        for ttc in ttc_values:
            frame_id = contact_frame - ttc
            if frame_id not in common_frames:
                skipped_rows.append(
                    {
                        "split": split,
                        "record_id": record_id,
                        "frame_id": str(frame_id),
                        "probe": str(ttc),
                        "reason": "precontact_frame_missing",
                        "detail": "",
                    }
                )
                continue
            current = predictions.get((split, record_id, frame_id))
            if current is None:
                current = predict_sensor_points(model, vision_by_frame[frame_id], input_width, input_height, device)
                predictions[(split, record_id, frame_id)] = current
                track_rows.append(format_track(split, record_id, frame_id, current))

            ok, reason, detail = quality_reason(
                current,
                target,
                min_confidence,
                min_tip_base_distance,
                max_tip_base_distance,
                contact_box_size,
            )
            if not ok:
                skipped_rows.append(
                    {
                        "split": split,
                        "record_id": record_id,
                        "frame_id": str(frame_id),
                        "probe": str(ttc),
                        "reason": reason,
                        "detail": detail,
                    }
                )
                continue

            image_w = int(current["image_width"])
            image_h = int(current["image_height"])
            hm_x = float(target["tip_x"]) / image_w * heatmap_w
            hm_y = float(target["tip_y"]) / image_h * heatmap_h
            heatmap = make_heatmap(heatmap_w, heatmap_h, hm_x, hm_y, sigma)
            image_name = f"{split}_{record_id}_probe{ttc:03d}_frame{frame_id:06d}.jpg"
            heatmap_path = heatmap_dir / f"{Path(image_name).stem}.npy"
            ensure_dir(heatmap_path.parent)
            np.save(heatmap_path, heatmap)
            sequence_ready = all((frame_id - offset) in common_frames for offset in sequence_offsets)
            history_frames = list(range(frame_id - trajectory_history_frames + 1, frame_id + 1))
            history_predictions = [
                predictions[(split, record_id, history_frame)]
                for history_frame in history_frames
                if (split, record_id, history_frame) in predictions
            ]
            history_ids = [
                history_frame
                for history_frame in history_frames
                if (split, record_id, history_frame) in predictions
            ]
            if len(history_predictions) >= 2:
                cumulative_displacement = sum(
                    math.hypot(float(right["tip_x"]) - float(left["tip_x"]), float(right["tip_y"]) - float(left["tip_y"]))
                    for left, right in zip(history_predictions[:-1], history_predictions[1:])
                )
                history_span = history_ids[-1] - history_ids[0]
                max_gap = max(right - left for left, right in zip(history_ids[:-1], history_ids[1:]))
            else:
                cumulative_displacement = 0.0
                history_span = 0
                max_gap = 0
            sample_rows.append(
                {
                    "dataset_split": planned_split_map[(split, record_id)],
                    "record_partition": planned_partitions[(split, record_id)],
                    "split": split,
                    "record_id": record_id,
                    "image_name": image_name,
                    "image_path": str(vision_by_frame[frame_id]),
                    "vision_path": str(vision_by_frame[frame_id]),
                    "touch_path": str(touch_by_frame[contact_frame]),
                    "probe": str(ttc),
                    "frame_id": str(frame_id),
                    "contact_frame_detected": str(contact_frame),
                    "tip_x": f"{float(current['tip_x']):.3f}",
                    "tip_y": f"{float(current['tip_y']):.3f}",
                    "base_x": f"{float(current['base_x']):.3f}",
                    "base_y": f"{float(current['base_y']):.3f}",
                    "direction_x": f"{float(current['direction_x']):.6f}",
                    "direction_y": f"{float(current['direction_y']):.6f}",
                    "target_tip_x": f"{float(target['tip_x']):.3f}",
                    "target_tip_y": f"{float(target['tip_y']):.3f}",
                    "image_width": str(image_w),
                    "image_height": str(image_h),
                    "tip_confidence": f"{float(current['tip_confidence']):.6f}",
                    "base_confidence": f"{float(current['base_confidence']):.6f}",
                    "target_tip_confidence": f"{float(target['tip_confidence']):.6f}",
                    "target_base_confidence": f"{float(target['base_confidence']):.6f}",
                    "tip_base_distance": f"{float(current['tip_base_distance']):.3f}",
                    "target_tip_base_distance": f"{float(target['tip_base_distance']):.3f}",
                    "sequence_ready": "1" if sequence_ready else "0",
                    "trajectory_real_point_count": str(len(history_predictions)),
                    "trajectory_history_span_frames": str(history_span),
                    "trajectory_padding_ratio": f"{1.0 - len(history_predictions) / max(trajectory_history_frames, 1):.6f}",
                    "trajectory_max_frame_gap": str(max_gap),
                    "trajectory_cumulative_displacement": f"{cumulative_displacement:.3f}",
                    "heatmap_path": str(heatmap_path),
                }
            )
            records_with_samples.add((split, record_id))

            if debug_written < debug_samples and planned_split_map[(split, record_id)] != "test":
                debug_path = debug_dir / "overlays" / f"{debug_written:03d}_{Path(image_name).stem}.jpg"
                save_debug_overlay(
                    debug_path,
                    vision_by_frame[frame_id],
                    current,
                    target,
                    contact_box_size,
                    f"{record_id} ttc={ttc} contact={contact_frame}",
                )
                debug_written += 1

    write_csv_rows(output_csv, sample_rows, SAMPLE_FIELDS)
    write_csv_rows(track_csv, track_rows, TRACK_FIELDS)
    write_csv_rows(contact_csv, contact_rows, CONTACT_FIELDS)
    write_csv_rows(skipped_csv, skipped_rows, SKIPPED_FIELDS)

    sample_counts = Counter(row["dataset_split"] for row in sample_rows)
    skipped_counts = Counter(row["reason"] for row in skipped_rows)
    contact_counts = Counter(row["status"] for row in contact_rows)
    summary = {
        "device": str(device),
        "config_section": section,
        "split": split,
        "record_start": record_start,
        "record_limit": record_limit,
        "selected_records": len(selected_records),
        "planned_record_split_counts": dict(Counter(planned_split_map.values())),
        "record_split_manifest": str(manifest_path) if manifest_path else "",
        "purpose_partition_manifest": str(project_path(purpose_manifest_value)) if purpose_manifest_value else "",
        "expected_purpose_partition": expansion_cfg.get("expected_purpose_partition", ""),
        "record_split_seed": int(expansion_cfg.get("record_split_seed", 42)),
        "final_holdout_count": int(expansion_cfg.get("final_holdout_count", 0)),
        "final_holdout_min_record": expansion_cfg.get("final_holdout_min_record"),
        "excluded_records": sorted(excluded_records),
        "contact_status_counts": dict(contact_counts),
        "records_with_samples": len(records_with_samples),
        "samples": len(sample_rows),
        "sample_split_counts": dict(sample_counts),
        "tracks": len(track_rows),
        "skipped": len(skipped_rows),
        "skipped_counts": dict(skipped_counts),
        "ttc_values": ttc_values,
        "sequence_offsets": sequence_offsets,
        "trajectory_history_frames": trajectory_history_frames,
        "trajectory_track_stride": trajectory_track_stride,
        "quality_filter": {
            "min_confidence": min_confidence,
            "min_tip_base_distance": min_tip_base_distance,
            "max_tip_base_distance": max_tip_base_distance,
            "contact_box_size": contact_box_size,
        },
        "sensor_checkpoint": str(checkpoint_path),
        "output_csv": str(output_csv),
        "tracks_csv": str(track_csv),
        "contact_csv": str(contact_csv),
        "skipped_csv": str(skipped_csv),
        "heatmap_dir": str(heatmap_dir),
        "debug_dir": str(debug_dir),
        "elapsed_seconds": round(time.time() - start_time, 2),
        "note": "Raw RGB/touch files stay under /mnt/data; this script writes only small labels, heatmaps, and debug overlays.",
    }
    write_json(summary_json, summary)
    print(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an automatically labeled expanded contact-region dataset.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split", default=None)
    parser.add_argument("--record-start", type=int, default=None)
    parser.add_argument("--record-limit", type=int, default=None)
    parser.add_argument("--section", default="expanded_region_dataset")
    args = parser.parse_args()
    build_expanded_region_dataset(args.config, args.split, args.record_start, args.record_limit, args.section)


if __name__ == "__main__":
    main()
