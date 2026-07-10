from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from .config import load_config, project_path
from .utils import ensure_dir, list_image_files, parse_frame_id, read_csv_rows, write_csv_rows, write_json


RECORD_FIELDS = [
    "split",
    "record_id",
    "vision_frames",
    "touch_frames",
    "common_frames",
    "vision_only_frames",
    "touch_only_frames",
    "min_common_frame",
    "max_common_frame",
    "common_span",
    "missing_inside_span",
    "aligned",
    "sequence_ready",
    "already_labeled",
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
    "ttc_samples",
    "sequence_samples",
    "usable_ttc_values",
    "usable_sequence_ttc_values",
]


def frame_map(directory: Path, exts: list[str]) -> dict[int, Path]:
    if not directory.exists():
        return {}
    out: dict[int, Path] = {}
    for path in list_image_files(directory, exts):
        frame_id = parse_frame_id(path)
        if frame_id is not None:
            out[frame_id] = path
    return out


def parse_splits(value: str | Iterable[str]) -> list[str]:
    if isinstance(value, str):
        if value.lower() == "all":
            return []
        return [part.strip() for part in value.split(",") if part.strip()]
    return [str(item) for item in value]


def available_splits(root: Path, vision_name: str) -> list[str]:
    vision_root = root / vision_name
    if not vision_root.exists():
        raise FileNotFoundError(f"Missing vision root: {vision_root}")
    return sorted(path.name for path in vision_root.iterdir() if path.is_dir())


def load_existing_region_records(path: Path) -> tuple[set[tuple[str, str]], int]:
    if not path.exists():
        return set(), 0
    rows = read_csv_rows(path)
    return {(row["split"], row["record_id"]) for row in rows}, len(rows)


def audit_records(
    root: Path,
    splits: list[str],
    vision_name: str,
    touch_name: str,
    exts: list[str],
    min_common_frames: int,
    sequence_offsets: list[int],
    labeled_records: set[tuple[str, str]],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for split in splits:
        vision_split = root / vision_name / split
        touch_split = root / touch_name / split
        if not vision_split.exists() or not touch_split.exists():
            continue
        record_ids = sorted(
            {path.name for path in vision_split.iterdir() if path.is_dir()}
            | {path.name for path in touch_split.iterdir() if path.is_dir()}
        )
        for record_id in record_ids:
            vision_by_frame = frame_map(vision_split / record_id, exts)
            touch_by_frame = frame_map(touch_split / record_id, exts)
            vision_frames = set(vision_by_frame)
            touch_frames = set(touch_by_frame)
            common = sorted(vision_frames & touch_frames)
            if common:
                min_frame = min(common)
                max_frame = max(common)
                span = max_frame - min_frame + 1
                missing_inside = span - len(common)
            else:
                min_frame = ""
                max_frame = ""
                span = 0
                missing_inside = 0
            sequence_ready = len(common) >= min_common_frames and span >= (max(sequence_offsets) + 1 if sequence_offsets else 1)
            rows.append(
                {
                    "split": split,
                    "record_id": record_id,
                    "vision_frames": str(len(vision_frames)),
                    "touch_frames": str(len(touch_frames)),
                    "common_frames": str(len(common)),
                    "vision_only_frames": str(len(vision_frames - touch_frames)),
                    "touch_only_frames": str(len(touch_frames - vision_frames)),
                    "min_common_frame": str(min_frame),
                    "max_common_frame": str(max_frame),
                    "common_span": str(span),
                    "missing_inside_span": str(missing_inside),
                    "aligned": "1" if len(common) >= min_common_frames else "0",
                    "sequence_ready": "1" if sequence_ready else "0",
                    "already_labeled": "1" if (split, record_id) in labeled_records else "0",
                }
            )
    return rows


def load_touch_array(path: Path, size: tuple[int, int]) -> np.ndarray:
    image = Image.open(path).convert("L").resize(size)
    return np.asarray(image, dtype=np.float32)


def first_consecutive(scores: list[tuple[int, float]], threshold: float, min_frame: int, consecutive: int):
    run: list[tuple[int, float]] = []
    for frame_id, score in scores:
        if frame_id < min_frame:
            continue
        if score >= threshold:
            run.append((frame_id, score))
            if len(run) >= consecutive:
                return run[0]
        else:
            run = []
    return None


def detect_contact_for_record(
    touch_paths: dict[int, Path],
    resize_size: tuple[int, int],
    baseline_frames: int,
    min_frame: int,
    threshold_abs: float,
    threshold_std_factor: float,
    threshold_peak_ratio: float,
    consecutive: int,
) -> dict[str, float | int | str]:
    frames = sorted(touch_paths)
    if len(frames) < max(baseline_frames + consecutive, min_frame + consecutive):
        return {"status": "too_short"}
    baseline_arrays = [load_touch_array(touch_paths[frame], resize_size) for frame in frames[:baseline_frames]]
    baseline = np.mean(np.stack(baseline_arrays, axis=0), axis=0)
    scores: list[tuple[int, float]] = []
    for frame in frames:
        arr = load_touch_array(touch_paths[frame], resize_size)
        scores.append((frame, float(np.mean(np.abs(arr - baseline)))))

    base_scores = [score for _, score in scores[:baseline_frames]]
    baseline_mean = float(np.mean(base_scores))
    baseline_std = float(np.std(base_scores))
    max_score = float(max(score for _, score in scores))
    threshold = max(
        threshold_abs,
        baseline_mean + threshold_std_factor * baseline_std,
        threshold_peak_ratio * max_score,
    )
    detected = first_consecutive(scores, threshold, min_frame, consecutive)
    if detected is None:
        return {
            "status": "not_found",
            "threshold": threshold,
            "max_score": max_score,
            "baseline_mean": baseline_mean,
            "baseline_std": baseline_std,
        }
    contact_frame, contact_score = detected
    return {
        "status": "ok",
        "contact_frame": contact_frame,
        "contact_score": contact_score,
        "threshold": threshold,
        "max_score": max_score,
        "baseline_mean": baseline_mean,
        "baseline_std": baseline_std,
    }


def audit_contact_sample(
    root: Path,
    candidate_records: list[dict[str, str]],
    vision_name: str,
    touch_name: str,
    exts: list[str],
    sample_limit: int,
    seed: int,
    resize_size: tuple[int, int],
    baseline_frames: int,
    min_frame: int,
    threshold_abs: float,
    threshold_std_factor: float,
    threshold_peak_ratio: float,
    consecutive: int,
    ttc_values: list[int],
    sequence_offsets: list[int],
) -> list[dict[str, str]]:
    aligned = [row for row in candidate_records if row["aligned"] == "1"]
    if sample_limit <= 0:
        return []
    rng = random.Random(seed)
    sampled = aligned[:]
    rng.shuffle(sampled)
    sampled = sampled[:sample_limit]

    rows: list[dict[str, str]] = []
    for row in sampled:
        split = row["split"]
        record_id = row["record_id"]
        vision_by_frame = frame_map(root / vision_name / split / record_id, exts)
        touch_by_frame = frame_map(root / touch_name / split / record_id, exts)
        common_frames = set(vision_by_frame) & set(touch_by_frame)
        result = detect_contact_for_record(
            {frame: touch_by_frame[frame] for frame in sorted(common_frames)},
            resize_size,
            baseline_frames,
            min_frame,
            threshold_abs,
            threshold_std_factor,
            threshold_peak_ratio,
            consecutive,
        )
        contact_frame = result.get("contact_frame", "")
        usable_ttc: list[int] = []
        usable_sequence_ttc: list[int] = []
        if result["status"] == "ok":
            contact = int(contact_frame)
            for ttc in ttc_values:
                current_frame = contact - ttc
                if current_frame in common_frames:
                    usable_ttc.append(ttc)
                    if all((current_frame - offset) in common_frames for offset in sequence_offsets):
                        usable_sequence_ttc.append(ttc)
        rows.append(
            {
                "split": split,
                "record_id": record_id,
                "status": str(result["status"]),
                "contact_frame": str(contact_frame),
                "contact_score": f"{float(result.get('contact_score', 0.0)):.6f}" if contact_frame != "" else "",
                "threshold": f"{float(result.get('threshold', 0.0)):.6f}" if "threshold" in result else "",
                "max_score": f"{float(result.get('max_score', 0.0)):.6f}" if "max_score" in result else "",
                "baseline_mean": f"{float(result.get('baseline_mean', 0.0)):.6f}" if "baseline_mean" in result else "",
                "baseline_std": f"{float(result.get('baseline_std', 0.0)):.6f}" if "baseline_std" in result else "",
                "common_frames": row["common_frames"],
                "ttc_samples": str(len(usable_ttc)),
                "sequence_samples": str(len(usable_sequence_ttc)),
                "usable_ttc_values": ",".join(str(item) for item in usable_ttc),
                "usable_sequence_ttc_values": ",".join(str(item) for item in usable_sequence_ttc),
            }
        )
    return rows


def summarize_numbers(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"min": None, "median": None, "mean": None, "max": None}
    return {
        "min": min(values),
        "median": float(statistics.median(values)),
        "mean": float(statistics.mean(values)),
        "max": max(values),
    }


def read_history_summaries(root: Path) -> list[dict[str, str | int | float]]:
    if not root.exists():
        return []
    out = []
    for path in sorted(root.glob("*/summary.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        sample_count = (
            data.get("num_samples")
            or data.get("samples")
            or data.get("total_samples")
            or data.get("candidate_samples")
        )
        if isinstance(sample_count, str) and not sample_count.isdigit():
            sample_count = None
        out.append(
            {
                "name": path.parent.name,
                "path": str(path),
                "num_samples": int(sample_count) if sample_count is not None else None,
            }
        )
    return out


def build_summary(
    record_rows: list[dict[str, str]],
    contact_rows: list[dict[str, str]],
    current_region_sample_count: int,
    history_root: Path,
) -> dict:
    split_counts = Counter(row["split"] for row in record_rows)
    aligned_rows = [row for row in record_rows if row["aligned"] == "1"]
    sequence_ready_rows = [row for row in record_rows if row["sequence_ready"] == "1"]
    labeled_rows = [row for row in record_rows if row["already_labeled"] == "1"]
    common_counts = [int(row["common_frames"]) for row in record_rows]
    aligned_common_counts = [int(row["common_frames"]) for row in aligned_rows]
    found_rows = [row for row in contact_rows if row["status"] == "ok"]
    ttc_counts = [int(row["ttc_samples"]) for row in found_rows]
    sequence_counts = [int(row["sequence_samples"]) for row in found_rows]
    contact_detection_rate = len(found_rows) / len(contact_rows) if contact_rows else None
    estimated_ttc_samples = None
    estimated_sequence_samples = None
    if contact_detection_rate is not None and found_rows:
        estimated_ttc_samples = round(len(aligned_rows) * contact_detection_rate * statistics.mean(ttc_counts))
        estimated_sequence_samples = round(len(aligned_rows) * contact_detection_rate * statistics.mean(sequence_counts))
    history = read_history_summaries(history_root)
    history_with_counts = [item for item in history if item["num_samples"] is not None]
    history_with_counts = sorted(history_with_counts, key=lambda item: int(item["num_samples"]), reverse=True)[:12]
    return {
        "records": len(record_rows),
        "records_by_split": dict(sorted(split_counts.items())),
        "aligned_records": len(aligned_rows),
        "sequence_ready_records": len(sequence_ready_rows),
        "already_labeled_records": len(labeled_rows),
        "current_region_samples": current_region_sample_count,
        "common_frame_counts_all": summarize_numbers(common_counts),
        "common_frame_counts_aligned": summarize_numbers(aligned_common_counts),
        "contact_audit": {
            "sampled_records": len(contact_rows),
            "found": len(found_rows),
            "not_found_or_failed": len(contact_rows) - len(found_rows),
            "detection_rate": contact_detection_rate,
            "ttc_samples_per_detected_record": summarize_numbers(ttc_counts),
            "sequence_samples_per_detected_record": summarize_numbers(sequence_counts),
            "estimated_ttc_samples_if_scaled_to_aligned_records": estimated_ttc_samples,
            "estimated_sequence_samples_if_scaled_to_aligned_records": estimated_sequence_samples,
        },
        "historical_artifacts": history_with_counts,
    }


def count_record_dirs_by_split(root: Path, stream_name: str) -> dict[str, int]:
    stream_root = root / stream_name
    if not stream_root.exists():
        return {}
    counts: dict[str, int] = {}
    for split_dir in sorted(path for path in stream_root.iterdir() if path.is_dir()):
        counts[split_dir.name] = sum(1 for path in split_dir.iterdir() if path.is_dir())
    return counts


def audit_dataset_expansion(
    config_path: str,
    splits_override: str | None = None,
    contact_sample_limit_override: int | None = None,
) -> dict:
    cfg = load_config(config_path)
    dataset_cfg = cfg["dataset"]
    audit_cfg = cfg.get("dataset_expansion_audit", {})
    root = Path(dataset_cfg["root"])
    vision_name = str(dataset_cfg["vision_name"])
    touch_name = str(dataset_cfg["touch_name"])
    exts = list(dataset_cfg["image_exts"])
    splits = parse_splits(splits_override or audit_cfg.get("splits", dataset_cfg.get("split", "0")))
    if not splits:
        splits = available_splits(root, vision_name)

    min_common_frames = int(audit_cfg.get("min_common_frames", 120))
    ttc_values = [int(item) for item in audit_cfg.get("ttc_values", [5, 10, 20, 30, 50, 75, 100])]
    sequence_offsets = [int(item) for item in audit_cfg.get("sequence_offsets", [15, 10, 5, 0])]
    seed = int(audit_cfg.get("seed", 42))
    contact_sample_limit = int(
        contact_sample_limit_override
        if contact_sample_limit_override is not None
        else audit_cfg.get("contact_sample_limit", 100)
    )

    existing_region_path = project_path(cfg["region_dataset"]["output_csv"])
    labeled_records, current_region_sample_count = load_existing_region_records(existing_region_path)
    record_rows = audit_records(
        root,
        splits,
        vision_name,
        touch_name,
        exts,
        min_common_frames,
        sequence_offsets,
        labeled_records,
    )

    contact_cfg = cfg["contact_detection"]
    contact_rows = audit_contact_sample(
        root,
        record_rows,
        vision_name,
        touch_name,
        exts,
        contact_sample_limit,
        seed,
        (int(audit_cfg.get("contact_resize_width", contact_cfg["resize_width"])), int(audit_cfg.get("contact_resize_height", contact_cfg["resize_height"]))),
        int(contact_cfg["baseline_frames"]),
        int(contact_cfg["min_frame"]),
        float(contact_cfg["threshold_abs"]),
        float(contact_cfg["threshold_std_factor"]),
        float(contact_cfg["threshold_peak_ratio"]),
        int(contact_cfg["consecutive_frames"]),
        ttc_values,
        sequence_offsets,
    )

    records_csv = project_path(audit_cfg.get("records_csv", "data/processed/dataset_expansion_audit_records.csv"))
    contact_csv = project_path(audit_cfg.get("contact_csv", "data/processed/dataset_expansion_contact_sample.csv"))
    summary_json = project_path(audit_cfg.get("summary_json", "outputs/metrics/dataset_expansion_audit.json"))
    write_csv_rows(records_csv, record_rows, RECORD_FIELDS)
    write_csv_rows(contact_csv, contact_rows, CONTACT_FIELDS)

    summary = build_summary(
        record_rows,
        contact_rows,
        current_region_sample_count,
        Path(audit_cfg.get("history_root", "/mnt/data/cheng/contact_policy")),
    )
    summary.update(
        {
            "dataset_root": str(root),
            "splits": splits,
            "available_vision_record_dirs_by_split": count_record_dirs_by_split(root, vision_name),
            "available_touch_record_dirs_by_split": count_record_dirs_by_split(root, touch_name),
            "records_csv": str(records_csv),
            "contact_csv": str(contact_csv),
            "summary_json": str(summary_json),
            "ttc_values": ttc_values,
            "sequence_offsets": sequence_offsets,
            "note": "Only small CSV/JSON audit files are written under the project; raw RGB/touch data stays in /mnt/data.",
        }
    )
    write_json(summary_json, summary)
    print(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit how much VisGel data can be used for dataset expansion.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--splits", default=None, help="Comma-separated splits or 'all'.")
    parser.add_argument("--contact-sample-limit", type=int, default=None)
    args = parser.parse_args()
    audit_dataset_expansion(args.config, args.splits, args.contact_sample_limit)


if __name__ == "__main__":
    main()
