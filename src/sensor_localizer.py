from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from .config import load_config, project_path
from .utils import (
    bbox_center,
    ensure_dir,
    normalized_vector,
    parse_makesense_image_name,
    read_csv_rows,
    save_label_overlay,
    write_csv_rows,
    write_json,
)


LABEL_FIELDS = [
    "split",
    "record_id",
    "image_name",
    "image_path",
    "probe",
    "frame_id",
    "contact_frame_from_name",
    "image_width",
    "image_height",
    "tip_x",
    "tip_y",
    "base_x",
    "base_y",
    "direction_x",
    "direction_y",
    "complete",
]

TRACK_FIELDS = [
    "split",
    "record_id",
    "frame_id",
    "contact_frame_from_name",
    "tip_x",
    "tip_y",
    "base_x",
    "base_y",
    "direction_x",
    "direction_y",
    "target_tip_x",
    "target_tip_y",
    "source",
]


def _fit_line(xs: list[float], ys: list[float]):
    if len(xs) < 2:
        return float(ys[0]), 0.0
    slope, intercept = np.polyfit(np.asarray(xs, dtype=float), np.asarray(ys, dtype=float), deg=1)
    return float(intercept), float(slope)


def _predict(line: tuple[float, float], frame_id: int) -> float:
    intercept, slope = line
    return intercept + slope * frame_id


def prepare_sensor_labels(config_path: str) -> dict:
    cfg = load_config(config_path)
    ms_cfg = cfg["makesense"]
    localizer_cfg = cfg["sensor_localizer"]
    labels_csv = project_path(ms_cfg["labels_csv"])
    images_dir = project_path(ms_cfg["images_dir"])
    label_output = project_path(localizer_cfg["labels_output_csv"])
    tracks_output = project_path(localizer_cfg["tracks_output_csv"])
    records_output = project_path(localizer_cfg["records_output_txt"])
    summary_output = project_path(localizer_cfg["summary_json"])
    debug_dir = project_path(localizer_cfg["debug_dir"])
    debug_samples = int(localizer_cfg.get("debug_samples", 0))

    raw_rows = read_csv_rows(labels_csv)
    grouped: dict[str, dict] = defaultdict(dict)
    metadata: dict[str, dict] = {}
    for row in raw_rows:
        image_name = row["image_name"]
        label_name = row["label_name"]
        grouped[image_name][label_name] = row
        metadata[image_name] = parse_makesense_image_name(image_name)

    label_rows: list[dict] = []
    incomplete_images: list[str] = []
    for image_name in sorted(grouped):
        labels = grouped[image_name]
        meta = metadata[image_name]
        tip_row = labels.get("sensor_tip")
        base_row = labels.get("sensor_base")
        complete = tip_row is not None and base_row is not None
        if not complete:
            incomplete_images.append(image_name)
            continue
        tip_x, tip_y = bbox_center(tip_row)
        base_x, base_y = bbox_center(base_row)
        direction_x, direction_y = normalized_vector(tip_x - base_x, tip_y - base_y)
        label_rows.append(
            {
                "split": meta["split"],
                "record_id": meta["record_id"],
                "image_name": image_name,
                "image_path": str(images_dir / image_name),
                "probe": meta["probe"],
                "frame_id": meta["frame_id"],
                "contact_frame_from_name": meta["contact_frame_from_name"],
                "image_width": tip_row["image_width"],
                "image_height": tip_row["image_height"],
                "tip_x": f"{tip_x:.3f}",
                "tip_y": f"{tip_y:.3f}",
                "base_x": f"{base_x:.3f}",
                "base_y": f"{base_y:.3f}",
                "direction_x": f"{direction_x:.6f}",
                "direction_y": f"{direction_y:.6f}",
                "complete": "1",
            }
        )

    write_csv_rows(label_output, label_rows, LABEL_FIELDS)

    by_record: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in label_rows:
        by_record[(row["split"], row["record_id"])].append(row)

    track_rows: list[dict] = []
    contact_inconsistencies: list[dict] = []
    for (split, record_id), rows in sorted(by_record.items()):
        rows = sorted(rows, key=lambda r: int(r["frame_id"]))
        contact_frames = {int(r["contact_frame_from_name"]) for r in rows}
        if len(contact_frames) != 1:
            contact_inconsistencies.append(
                {"split": split, "record_id": record_id, "contact_frames": sorted(contact_frames)}
            )
        contact_frame = int(round(sum(contact_frames) / len(contact_frames)))
        frames = [int(r["frame_id"]) for r in rows]
        tip_xs = [float(r["tip_x"]) for r in rows]
        tip_ys = [float(r["tip_y"]) for r in rows]
        base_xs = [float(r["base_x"]) for r in rows]
        base_ys = [float(r["base_y"]) for r in rows]
        tip_x_line = _fit_line(frames, tip_xs)
        tip_y_line = _fit_line(frames, tip_ys)
        base_x_line = _fit_line(frames, base_xs)
        base_y_line = _fit_line(frames, base_ys)
        target_tip_x = _predict(tip_x_line, contact_frame)
        target_tip_y = _predict(tip_y_line, contact_frame)
        labeled_frames = {int(r["frame_id"]): r for r in rows}
        for frame_id in range(min(frames), contact_frame + 1):
            if frame_id in labeled_frames:
                row = labeled_frames[frame_id]
                tip_x = float(row["tip_x"])
                tip_y = float(row["tip_y"])
                base_x = float(row["base_x"])
                base_y = float(row["base_y"])
                source = "label"
            else:
                tip_x = _predict(tip_x_line, frame_id)
                tip_y = _predict(tip_y_line, frame_id)
                base_x = _predict(base_x_line, frame_id)
                base_y = _predict(base_y_line, frame_id)
                source = "interpolated" if min(frames) <= frame_id <= max(frames) else "extrapolated"
            direction_x, direction_y = normalized_vector(tip_x - base_x, tip_y - base_y)
            track_rows.append(
                {
                    "split": split,
                    "record_id": record_id,
                    "frame_id": frame_id,
                    "contact_frame_from_name": contact_frame,
                    "tip_x": f"{tip_x:.3f}",
                    "tip_y": f"{tip_y:.3f}",
                    "base_x": f"{base_x:.3f}",
                    "base_y": f"{base_y:.3f}",
                    "direction_x": f"{direction_x:.6f}",
                    "direction_y": f"{direction_y:.6f}",
                    "target_tip_x": f"{target_tip_x:.3f}",
                    "target_tip_y": f"{target_tip_y:.3f}",
                    "source": source,
                }
            )

    write_csv_rows(tracks_output, track_rows, TRACK_FIELDS)
    ensure_dir(records_output.parent)
    records_output.write_text(
        "\n".join(f"{split}/{record_id}" for split, record_id in sorted(by_record)) + "\n",
        encoding="utf-8",
    )

    for idx, row in enumerate(label_rows[:debug_samples]):
        record_tracks = [
            tr
            for tr in track_rows
            if tr["split"] == row["split"]
            and tr["record_id"] == row["record_id"]
            and int(tr["frame_id"]) == int(row["frame_id"])
        ]
        target = None
        if record_tracks:
            target = (float(record_tracks[0]["target_tip_x"]), float(record_tracks[0]["target_tip_y"]))
        output_path = debug_dir / f"{idx:03d}_{Path(row['image_name']).stem}_sensor.jpg"
        save_label_overlay(
            row["image_path"],
            output_path,
            (float(row["tip_x"]), float(row["tip_y"])),
            (float(row["base_x"]), float(row["base_y"])),
            target,
        )

    summary = {
        "raw_rows": len(raw_rows),
        "complete_images": len(label_rows),
        "incomplete_images": incomplete_images,
        "records": len(by_record),
        "track_rows": len(track_rows),
        "contact_inconsistencies": contact_inconsistencies,
        "labels_output_csv": str(label_output),
        "tracks_output_csv": str(tracks_output),
        "records_output_txt": str(records_output),
    }
    write_json(summary_output, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare makesense sensor labels and interpolated tracks.")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    summary = prepare_sensor_labels(args.config)
    print(summary)


if __name__ == "__main__":
    main()
