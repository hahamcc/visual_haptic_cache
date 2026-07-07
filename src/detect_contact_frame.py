from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .config import load_config, project_path
from .utils import ensure_dir, read_csv_rows, write_csv_rows, write_json


FIELDS = [
    "split",
    "record_id",
    "contact_frame",
    "contact_score",
    "threshold",
    "max_score",
    "baseline_mean",
    "baseline_std",
    "frame_count",
    "status",
]


def _read_records(path: Path | None) -> set[tuple[str, str]] | None:
    if path is None or not path.exists():
        return None
    records = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        if "/" in text:
            split, record_id = text.split("/", 1)
        elif "," in text:
            split, record_id = text.split(",", 1)
        else:
            split, record_id = "0", text
        records.add((split.strip(), record_id.strip()))
    return records


def _load_touch_array(path: str, size: tuple[int, int]) -> np.ndarray:
    image = Image.open(path).convert("L").resize(size)
    return np.asarray(image, dtype=np.float32)


def _first_consecutive(scores: list[tuple[int, float]], threshold: float, min_frame: int, consecutive: int):
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


def _save_score_chart(path: Path, scores: list[tuple[int, float]], threshold: float, contact_frame: int | None) -> None:
    if not scores:
        return
    width, height = 720, 220
    margin = 28
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    max_score = max(score for _, score in scores) or 1.0
    min_frame = min(frame for frame, _ in scores)
    max_frame = max(frame for frame, _ in scores)
    span = max(1, max_frame - min_frame)

    def xy(frame: int, score: float) -> tuple[float, float]:
        x = margin + (frame - min_frame) / span * (width - 2 * margin)
        y = height - margin - score / max_score * (height - 2 * margin)
        return x, y

    draw.rectangle((margin, margin, width - margin, height - margin), outline="gray")
    prev = None
    for frame, score in scores:
        point = xy(frame, score)
        if prev is not None:
            draw.line((prev[0], prev[1], point[0], point[1]), fill="blue", width=2)
        prev = point
    y_thr = xy(min_frame, threshold)[1]
    draw.line((margin, y_thr, width - margin, y_thr), fill="red", width=2)
    if contact_frame is not None:
        x_contact = xy(contact_frame, 0)[0]
        draw.line((x_contact, margin, x_contact, height - margin), fill="green", width=2)
    ensure_dir(path.parent)
    image.save(path)


def detect_contact_frames(config_path: str, manifest_csv: str | None = None, records_file: str | None = None) -> dict:
    cfg = load_config(config_path)
    contact_cfg = cfg["contact_detection"]
    manifest_path = project_path(manifest_csv or cfg["manifest"]["output_csv"])
    output_csv = project_path(contact_cfg["output_csv"])
    summary_json = project_path(contact_cfg["summary_json"])
    records_path = project_path(records_file or contact_cfg["records_file"])
    records_filter = _read_records(records_path)

    resize_size = (int(contact_cfg["resize_width"]), int(contact_cfg["resize_height"]))
    baseline_frames = int(contact_cfg["baseline_frames"])
    min_frame = int(contact_cfg["min_frame"])
    threshold_abs = float(contact_cfg["threshold_abs"])
    threshold_std_factor = float(contact_cfg["threshold_std_factor"])
    threshold_peak_ratio = float(contact_cfg["threshold_peak_ratio"])
    consecutive = int(contact_cfg["consecutive_frames"])
    debug_records = int(contact_cfg.get("debug_records", 0))
    debug_dir = project_path(contact_cfg["debug_dir"])

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in read_csv_rows(manifest_path):
        key = (row["split"], row["record_id"])
        if records_filter is not None and key not in records_filter:
            continue
        grouped[key].append(row)

    output_rows: list[dict] = []
    for record_idx, ((split, record_id), rows) in enumerate(sorted(grouped.items())):
        rows = sorted(rows, key=lambda r: int(r["frame_id"]))
        if not rows:
            continue
        baseline_arrays = [
            _load_touch_array(row["touch_path"], resize_size) for row in rows[: max(1, baseline_frames)]
        ]
        baseline = np.mean(np.stack(baseline_arrays, axis=0), axis=0)
        scores: list[tuple[int, float]] = []
        for row in rows:
            frame_id = int(row["frame_id"])
            arr = _load_touch_array(row["touch_path"], resize_size)
            score = float(np.mean(np.abs(arr - baseline)))
            scores.append((frame_id, score))

        base_scores = [score for _, score in scores[: max(1, baseline_frames)]]
        baseline_mean = float(np.mean(base_scores))
        baseline_std = float(np.std(base_scores))
        max_score = float(max(score for _, score in scores))
        threshold = max(
            threshold_abs,
            baseline_mean + threshold_std_factor * baseline_std,
            threshold_peak_ratio * max_score,
        )
        detected = _first_consecutive(scores, threshold, min_frame, consecutive)
        if detected is None:
            contact_frame = ""
            contact_score = ""
            status = "not_found"
        else:
            contact_frame, contact_score = detected
            status = "ok"
        output_rows.append(
            {
                "split": split,
                "record_id": record_id,
                "contact_frame": contact_frame,
                "contact_score": f"{contact_score:.6f}" if contact_score != "" else "",
                "threshold": f"{threshold:.6f}",
                "max_score": f"{max_score:.6f}",
                "baseline_mean": f"{baseline_mean:.6f}",
                "baseline_std": f"{baseline_std:.6f}",
                "frame_count": len(rows),
                "status": status,
            }
        )
        if record_idx < debug_records:
            chart_path = debug_dir / f"{split}_{record_id}_contact_scores.jpg"
            _save_score_chart(chart_path, scores, threshold, int(contact_frame) if contact_frame != "" else None)

    write_csv_rows(output_csv, output_rows, FIELDS)
    summary = {
        "manifest_csv": str(manifest_path),
        "records_file": str(records_path) if records_path.exists() else None,
        "records": len(output_rows),
        "found": sum(1 for row in output_rows if row["status"] == "ok"),
        "not_found": sum(1 for row in output_rows if row["status"] != "ok"),
        "output_csv": str(output_csv),
    }
    write_json(summary_json, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect contact frames from tactile image differences.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--manifest")
    parser.add_argument("--records-file")
    args = parser.parse_args()
    summary = detect_contact_frames(args.config, args.manifest, args.records_file)
    print(summary)


if __name__ == "__main__":
    main()
