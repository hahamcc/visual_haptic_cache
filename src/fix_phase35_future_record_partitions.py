from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config, project_path
from .utils import write_csv_rows, write_json


FIELDS = ["split", "record_id", "record_index", "partition", "purpose"]


def record_ids(start: int, limit: int) -> list[str]:
    return [f"rec_{index:05d}" for index in range(start, start + limit)]


def development_record_groups(cfg: dict, default_split: str) -> list[tuple[str, list[str]]]:
    """Read explicit development ranges, including later dataset splits when needed."""
    sources = cfg.get("development_sources")
    if not sources:
        ranges = cfg.get("development_ranges")
        if not ranges:
            ranges = [{"start": cfg["development_start"], "limit": cfg["development_limit"]}]
        sources = [{"split": default_split, "ranges": ranges}]

    groups = []
    seen: set[tuple[str, str]] = set()
    for source in sources:
        split = str(source["split"])
        records = [
            record_id
            for item in source["ranges"]
            for record_id in record_ids(int(item["start"]), int(item["limit"]))
        ]
        keys = {(split, record_id) for record_id in records}
        if len(keys) != len(records) or seen.intersection(keys):
            raise ValueError("Development ranges overlap")
        seen.update(keys)
        groups.append((split, records))
    return groups


def fix_partitions(config_path: str, section: str) -> dict:
    config = load_config(config_path)
    dataset_cfg = config["dataset"]
    cfg = config[section]
    root = Path(dataset_cfg["root"])
    default_split = str(dataset_cfg["split"])
    development_groups = development_record_groups(cfg, default_split)
    final_split = str(cfg.get("final_holdout_split", default_split))
    development = [(split, record_id) for split, records in development_groups for record_id in records]
    final_holdout = record_ids(int(cfg["final_holdout_start"]), int(cfg["final_holdout_limit"]))
    final = [(final_split, record_id) for record_id in final_holdout]
    overlap = set(development) & set(final)
    if overlap:
        raise ValueError(f"Development/final partitions overlap: {sorted(overlap)}")

    missing = {"vision": [], "touch": []}
    for split, record_id in development + final:
        vision_path = root / str(dataset_cfg["vision_name"]) / split / record_id
        touch_path = root / str(dataset_cfg["touch_name"]) / split / record_id
        if not vision_path.is_dir():
            missing["vision"].append(f"{split}/{record_id}")
        if not touch_path.is_dir():
            missing["touch"].append(f"{split}/{record_id}")
    if missing["vision"] or missing["touch"]:
        raise FileNotFoundError(f"Partition records missing from raw dataset: {missing}")

    rows = [
        {"split": split, "record_id": record_id, "record_index": str(index), "partition": partition, "purpose": purpose}
        for partition, purpose, records in (
            ("development", "data_building_and_model_selection", development),
            ("final_holdout", "one_time_final_evaluation_only", final),
        )
        for split, record_id in records
        for index in (int(record_id.rsplit("_", 1)[1]),)
    ]
    write_csv_rows(project_path(cfg["output_csv"]), rows, FIELDS)
    summary = {
        "policy": "The final holdout is partitioned before data building and must not be used for model selection or interim prediction review.",
        "raw_dataset_root": str(root),
        "splits": sorted({split for split, _ in development + final}),
        "development_records": len(development),
        "development_ranges": [
            {"split": split, "range": [records[0], records[-1]]}
            for split, records in development_groups
        ],
        "final_holdout_records": len(final_holdout),
        "final_holdout_range": {"split": final_split, "range": [final_holdout[0], final_holdout[-1]]},
        "raw_completeness": {name: len(values) == 0 for name, values in missing.items()},
    }
    write_json(project_path(cfg["summary_json"]), summary)
    print(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Fix the Phase35 development/final record partition before building new labels.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--section", default="phase35_future_record_partitions")
    args = parser.parse_args()
    fix_partitions(args.config, args.section)


if __name__ == "__main__":
    main()
