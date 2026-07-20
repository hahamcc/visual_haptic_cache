from __future__ import annotations

import argparse
from collections import Counter

from .config import load_config, project_path
from .utils import read_csv_rows, write_csv_rows, write_json


MANIFEST_FIELDS = ["source", "split", "record_id", "dataset_split", "record_partition"]


def build(config_path: str, section: str) -> dict:
    config = load_config(config_path)
    cfg = config[section]
    allowed_splits = {str(value) for value in cfg.get("allowed_splits", ["train", "val"])}
    min_trajectory_real_points = int(cfg.get("min_trajectory_real_points", 0))
    if allowed_splits != {"train", "val"}:
        raise ValueError("Phase35 development pool must contain only train and validation rows")

    sample_rows: list[dict[str, str]] = []
    track_rows: list[dict[str, str]] = []
    manifest_rows: list[dict[str, str]] = []
    sample_fields: list[str] | None = None
    track_fields: list[str] | None = None
    seen_records: set[tuple[str, str]] = set()
    source_summaries = []

    for source in cfg["sources"]:
        name = str(source["name"])
        all_samples = read_csv_rows(project_path(source["samples_csv"]))
        if not all_samples:
            raise ValueError(f"No sample rows in source {name}")
        if sample_fields is None:
            sample_fields = list(all_samples[0])
        elif list(all_samples[0]) != sample_fields:
            raise ValueError(f"Sample schema mismatch in source {name}")

        split_samples = [row for row in all_samples if row["dataset_split"] in allowed_splits]
        included_samples = [
            row for row in split_samples
            if int(row.get("trajectory_real_point_count", min_trajectory_real_points)) >= min_trajectory_real_points
        ]
        source_records = {(row["split"], row["record_id"]) for row in included_samples}
        overlap = seen_records.intersection(source_records)
        if overlap:
            raise ValueError(f"Record overlap across development sources: {sorted(overlap)[:5]}")
        seen_records.update(source_records)
        sample_rows.extend(included_samples)

        all_tracks = read_csv_rows(project_path(source["tracks_csv"]))
        if all_tracks:
            if track_fields is None:
                track_fields = list(all_tracks[0])
            elif list(all_tracks[0]) != track_fields:
                raise ValueError(f"Track schema mismatch in source {name}")
            included_tracks = [row for row in all_tracks if (row["split"], row["record_id"]) in source_records]
            track_rows.extend(included_tracks)
        else:
            included_tracks = []

        for split, record_id in sorted(source_records):
            row = next(item for item in included_samples if item["split"] == split and item["record_id"] == record_id)
            manifest_rows.append({
                "source": name,
                "split": split,
                "record_id": record_id,
                "dataset_split": row["dataset_split"],
                "record_partition": row.get("record_partition", row["dataset_split"]),
            })
        source_summaries.append({
            "name": name,
            "input_samples": len(all_samples),
            "included_samples": len(included_samples),
            "excluded_non_development_samples": len(all_samples) - len(split_samples),
            "excluded_short_trajectory_samples": len(split_samples) - len(included_samples),
            "included_records": len(source_records),
            "included_tracks": len(included_tracks),
        })

    if not sample_rows or sample_fields is None or track_fields is None:
        raise ValueError("Development pool has no usable samples or tracks")
    if any(row["dataset_split"] not in allowed_splits for row in sample_rows):
        raise RuntimeError("A non-development split entered the merged pool")

    write_csv_rows(project_path(cfg["output_csv"]), sample_rows, sample_fields)
    write_csv_rows(project_path(cfg["tracks_csv"]), track_rows, track_fields)
    write_csv_rows(project_path(cfg["record_manifest_csv"]), manifest_rows, MANIFEST_FIELDS)
    summary = {
        "policy": "Merged Phase35 development pool. Only train and validation rows are included; sealed final-holdout rows are excluded.",
        "allowed_splits": sorted(allowed_splits),
        "min_trajectory_real_points": min_trajectory_real_points,
        "sources": source_summaries,
        "samples": len(sample_rows),
        "tracks": len(track_rows),
        "record_split_counts": dict(Counter(row["dataset_split"] for row in manifest_rows)),
        "sample_split_counts": dict(Counter(row["dataset_split"] for row in sample_rows)),
        "output_csv": str(project_path(cfg["output_csv"])),
        "tracks_csv": str(project_path(cfg["tracks_csv"])),
        "record_manifest_csv": str(project_path(cfg["record_manifest_csv"])),
    }
    write_json(project_path(cfg["summary_json"]), summary)
    print(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge sealed Phase35 train/validation pools without copying heatmaps.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--section", default="phase35_development_pool")
    args = parser.parse_args()
    build(args.config, args.section)


if __name__ == "__main__":
    main()
