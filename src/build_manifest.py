from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from .config import load_config, project_path
from .utils import ensure_dir, list_image_files, parse_frame_id, write_csv_rows, write_json


FIELDS = [
    "split",
    "record_id",
    "frame_id",
    "vision_path",
    "touch_path",
]


def _read_records_file(path: Path | None) -> set[tuple[str, str]] | None:
    if path is None or not path.exists():
        return None
    records: set[tuple[str, str]] = set()
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


def build_manifest(config_path: str, output_csv: str | None = None, records_file: str | None = None) -> dict:
    cfg = load_config(config_path)
    dataset = cfg["dataset"]
    root = Path(dataset["root"])
    split = str(dataset["split"])
    exts = dataset["image_exts"]
    vision_root = root / dataset["vision_name"] / split
    touch_root = root / dataset["touch_name"] / split
    if not vision_root.exists():
        raise FileNotFoundError(f"Missing vision root: {vision_root}")
    if not touch_root.exists():
        raise FileNotFoundError(f"Missing touch root: {touch_root}")

    output = project_path(output_csv or cfg["manifest"]["output_csv"])
    summary_path = project_path(cfg["manifest"]["summary_json"])
    records_filter = _read_records_file(project_path(records_file) if records_file else None)

    rows: list[dict] = []
    skipped: dict[str, int] = defaultdict(int)
    record_dirs = sorted(p for p in vision_root.iterdir() if p.is_dir())
    for vision_record_dir in record_dirs:
        record_id = vision_record_dir.name
        if records_filter is not None and (split, record_id) not in records_filter:
            continue
        touch_record_dir = touch_root / record_id
        if not touch_record_dir.exists():
            skipped["missing_touch_record"] += 1
            continue

        vision_by_frame = {
            frame_id: p
            for p in list_image_files(vision_record_dir, exts)
            if (frame_id := parse_frame_id(p)) is not None
        }
        touch_by_frame = {
            frame_id: p
            for p in list_image_files(touch_record_dir, exts)
            if (frame_id := parse_frame_id(p)) is not None
        }
        common_frames = sorted(set(vision_by_frame) & set(touch_by_frame))
        skipped["vision_only_frames"] += len(set(vision_by_frame) - set(touch_by_frame))
        skipped["touch_only_frames"] += len(set(touch_by_frame) - set(vision_by_frame))
        for frame_id in common_frames:
            rows.append(
                {
                    "split": split,
                    "record_id": record_id,
                    "frame_id": frame_id,
                    "vision_path": str(vision_by_frame[frame_id]),
                    "touch_path": str(touch_by_frame[frame_id]),
                }
            )

    write_csv_rows(output, rows, FIELDS)
    summary = {
        "dataset_root": str(root),
        "split": split,
        "records": len({row["record_id"] for row in rows}),
        "frames": len(rows),
        "output_csv": str(output),
        "skipped": dict(skipped),
    }
    write_json(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build aligned RGB/touch frame manifest.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output")
    parser.add_argument("--records-file")
    args = parser.parse_args()
    summary = build_manifest(args.config, args.output, args.records_file)
    print(summary)


if __name__ == "__main__":
    main()
