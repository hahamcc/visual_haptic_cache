"""Reserve a development-only raw candidate pool for Phase4I.6."""
from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config, project_path
from .phase4h_dino_adaptation import final_holdout_keys
from .utils import read_csv_rows, write_csv_rows, write_json


FIELDS = ["split", "record_id", "record_index", "partition", "purpose"]


def select_candidate_records(
    root: Path,
    vision_name: str,
    touch_name: str,
    split: str,
    record_start: int,
    record_limit: int,
    forbidden_keys: set[tuple[str, str]],
    forbidden_record_ids: set[str],
) -> list[str]:
    vision_root = root / vision_name / split
    touch_root = root / touch_name / split
    if not vision_root.is_dir() or not touch_root.is_dir():
        raise FileNotFoundError(
            f"Phase4I.6 raw split is incomplete: {vision_root}, {touch_root}"
        )
    vision_records = sorted(
        path.name for path in vision_root.iterdir() if path.is_dir()
    )
    selected = vision_records[record_start : record_start + record_limit]
    if len(selected) != record_limit:
        raise RuntimeError(
            "Phase4I.6 raw candidate range is incomplete: "
            f"{len(selected)} != {record_limit}"
        )
    missing_touch = [
        record_id
        for record_id in selected
        if not (touch_root / record_id).is_dir()
    ]
    if missing_touch:
        raise RuntimeError(
            f"Phase4I.6 candidate records miss touch data: {missing_touch[:3]}"
        )
    overlap = [
        record_id
        for record_id in selected
        if (split, record_id) in forbidden_keys
        or record_id in forbidden_record_ids
    ]
    if overlap:
        raise RuntimeError(
            "Phase4I.6 candidate pool overlaps a frozen record set: "
            f"{overlap[:3]}"
        )
    return selected


def reserve(config_path: str, section: str) -> dict:
    config = load_config(config_path)
    cfg = config[section]
    dataset = config["dataset"]
    split = str(cfg["split"])
    development = read_csv_rows(
        project_path(cfg["development_samples_csv"])
    )
    development_keys = {
        (row["split"], row["record_id"]) for row in development
    }
    development_ids = {row["record_id"] for row in development}
    sealed = final_holdout_keys(project_path(cfg["final_partition_csv"]))
    additional_forbidden_rows = []
    for manifest in cfg.get("additional_forbidden_partition_csvs", []):
        additional_forbidden_rows.extend(
            read_csv_rows(project_path(str(manifest)))
        )
    additional_forbidden_keys = {
        (str(row["split"]), str(row["record_id"]))
        for row in additional_forbidden_rows
    }
    additional_forbidden_ids = {
        str(row["record_id"]) for row in additional_forbidden_rows
    }
    selected = select_candidate_records(
        Path(dataset["root"]),
        str(dataset["vision_name"]),
        str(dataset["touch_name"]),
        split,
        int(cfg["record_start"]),
        int(cfg["record_limit"]),
        development_keys | sealed | additional_forbidden_keys,
        development_ids | additional_forbidden_ids,
    )
    expected_first = str(cfg["expected_first_record"])
    expected_last = str(cfg["expected_last_record"])
    if selected[0] != expected_first or selected[-1] != expected_last:
        raise RuntimeError(
            "Phase4I.6 candidate range identity changed: "
            f"{selected[0]}..{selected[-1]} != "
            f"{expected_first}..{expected_last}"
        )
    rows = [
        {
            "split": split,
            "record_id": record_id,
            "record_index": record_id.rsplit("_", 1)[-1],
            "partition": "development_candidate",
            "purpose": "phase4i6_temporal_far_candidate_selection",
        }
        for record_id in selected
    ]
    write_csv_rows(project_path(cfg["output_csv"]), rows, FIELDS)
    summary = {
        "mode": "phase4i6_raw_candidate_partition_v1",
        "raw_dataset_root": str(dataset["root"]),
        "split": split,
        "candidate_records": len(rows),
        "record_range": [selected[0], selected[-1]],
        "partition": "development_candidate",
        "development_overlap": 0,
        "sealed_final_overlap": 0,
        "additional_forbidden_overlap": 0,
        "additional_forbidden_records": len(additional_forbidden_ids),
        "raw_completeness": {"vision": True, "touch": True},
        "integrity": {
            "raw_directory_names_only": True,
            "raw_images_read": 0,
            "query_tactile_input": False,
            "sealed_final_holdout_rows_read": 0,
            "development_validation_rows_read": 0,
        },
        "output_csv": str(project_path(cfg["output_csv"])),
        "next_action": (
            "build probe75/100 labels and exact 32-frame trajectories for "
            "the frozen development candidate pool"
        ),
    }
    write_json(project_path(cfg["summary_json"]), summary)
    print(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reserve the Phase4I.6 raw development candidate pool."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--section",
        default="phase4i6_raw_candidate_partition_v1",
    )
    args = parser.parse_args()
    reserve(args.config, args.section)


if __name__ == "__main__":
    main()
