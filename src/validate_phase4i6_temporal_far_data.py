"""Validate completed Phase4I.6 record-disjoint temporal-far data."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from .config import load_config, project_path
from .phase4h_dino_adaptation import final_holdout_keys
from .plan_phase4i6_temporal_far_collection import (
    COLLECTION_FIELDS,
    validate_collection_plan,
)
from .temporal_progress import (
    masked_trajectory_features,
    read_trajectory_tracks,
)
from .utils import read_csv_rows, write_csv_rows, write_json


QUERY_FIELDS = [
    "collection_slot",
    "planned_pair_id",
    "pair_variant",
    "pair_design",
    "new_split",
    "new_record_id",
    "object_id",
    "contact_region_id",
    "probe",
    "image_name",
    "frame_id",
    "contact_frame_detected",
    "observed_ttc",
    "sequence_ready",
    "real_point_count",
    "history_span",
    "padding_ratio",
    "max_frame_gap",
    "vision_path_exists",
    "touch_path_exists",
    "passed",
    "failure_reasons",
]

RECORD_FIELDS = [
    "collection_slot",
    "planned_pair_id",
    "pair_variant",
    "pair_design",
    "new_split",
    "new_record_id",
    "object_id",
    "contact_region_id",
    "queries",
    "probes",
    "passed_queries",
    "record_passed",
    "failure_reasons",
]

REQUIRED_COMPLETION_FIELDS = (
    "new_split",
    "new_record_id",
    "object_id",
    "contact_region_id",
)


def record_key(row: dict[str, str]) -> tuple[str, str]:
    return row["new_split"], row["new_record_id"]


def validate_completed_plan(
    rows: list[dict[str, str]],
    development_keys: set[tuple[str, str]],
    development_record_ids: set[str],
    sealed_keys: set[tuple[str, str]],
    expected_records: int,
) -> None:
    if len(rows) != expected_records:
        raise RuntimeError(
            "Phase4I.6 completed plan record count changed: "
            f"{len(rows)} != {expected_records}"
        )
    missing = [
        f"{row['collection_slot']}:{field}"
        for row in rows
        for field in REQUIRED_COMPLETION_FIELDS
        if not row.get(field, "").strip()
    ]
    if missing:
        raise RuntimeError(
            "Phase4I.6 completed plan has blank identity fields: "
            + ", ".join(missing[:8])
        )
    keys = [record_key(row) for row in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Phase4I.6 new split/record keys must be unique")
    record_ids = [row["new_record_id"] for row in rows]
    if len(record_ids) != len(set(record_ids)):
        raise RuntimeError("Phase4I.6 new record IDs must be globally unique")
    references = {row["reference_record_id"] for row in rows}
    reused_reference = sorted(set(record_ids) & references)
    if reused_reference:
        raise RuntimeError(
            "Phase4I.6 reused reference record IDs: "
            f"{reused_reference[:3]}"
        )
    development_overlap = sorted(set(keys) & development_keys)
    if development_overlap:
        raise RuntimeError(
            "Phase4I.6 overlaps development records: "
            f"{development_overlap[:3]}"
        )
    reused_development_ids = sorted(set(record_ids) & development_record_ids)
    if reused_development_ids:
        raise RuntimeError(
            "Phase4I.6 reused development record IDs: "
            f"{reused_development_ids[:3]}"
        )
    final_overlap = sorted(set(keys) & sealed_keys)
    if final_overlap:
        raise RuntimeError(
            "Phase4I.6 overlaps sealed final holdout: "
            f"{final_overlap[:3]}"
        )
    validate_collection_plan(rows)
    pairs: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        pairs[row["planned_pair_id"]].append(row)
    for pair_id, pair_rows in pairs.items():
        for field in ("object_id", "contact_region_id"):
            if len({row[field] for row in pair_rows}) != 1:
                raise RuntimeError(
                    f"Phase4I.6 {pair_id} does not share {field}"
                )


def evaluate_query(
    plan_row: dict[str, str],
    sample: dict[str, str],
    tracks: dict[tuple[str, str], list[dict[str, float]]],
    cfg: dict,
) -> dict[str, str]:
    probe = int(sample["probe"])
    observed_ttc = (
        int(sample["contact_frame_detected"]) - int(sample["frame_id"])
    )
    _, _, quality = masked_trajectory_features(
        sample,
        tracks,
        history_frames=int(cfg["history_frames"]),
        spatial_scale_px=float(cfg["spatial_scale_px"]),
        speed_scale_px=float(cfg["speed_scale_px"]),
    )
    vision_exists = Path(sample["vision_path"]).exists()
    touch_exists = Path(sample["touch_path"]).exists()
    failures = []
    if observed_ttc != probe:
        failures.append("contact_frame_minus_frame_id_mismatch")
    if sample.get("sequence_ready", "0") != "1":
        failures.append("sequence_not_ready")
    if quality["real_point_count"] < float(cfg["minimum_real_point_count"]):
        failures.append("insufficient_real_points")
    if quality["history_span_frames"] < float(cfg["minimum_history_span"]):
        failures.append("insufficient_history_span")
    if quality["padding_ratio"] > float(cfg["maximum_padding_ratio"]):
        failures.append("padding_ratio_exceeded")
    if quality["max_frame_gap"] > float(cfg["maximum_frame_gap"]):
        failures.append("frame_gap_exceeded")
    if bool(cfg["require_paths_exist"]) and not vision_exists:
        failures.append("vision_path_missing")
    if bool(cfg["require_paths_exist"]) and not touch_exists:
        failures.append("touch_path_missing")
    return {
        "collection_slot": plan_row["collection_slot"],
        "planned_pair_id": plan_row["planned_pair_id"],
        "pair_variant": plan_row["pair_variant"],
        "pair_design": plan_row["pair_design"],
        "new_split": plan_row["new_split"],
        "new_record_id": plan_row["new_record_id"],
        "object_id": plan_row["object_id"],
        "contact_region_id": plan_row["contact_region_id"],
        "probe": str(probe),
        "image_name": sample["image_name"],
        "frame_id": sample["frame_id"],
        "contact_frame_detected": sample["contact_frame_detected"],
        "observed_ttc": str(observed_ttc),
        "sequence_ready": sample.get("sequence_ready", "0"),
        "real_point_count": f"{quality['real_point_count']:.6f}",
        "history_span": f"{quality['history_span_frames']:.6f}",
        "padding_ratio": f"{quality['padding_ratio']:.6f}",
        "max_frame_gap": f"{quality['max_frame_gap']:.6f}",
        "vision_path_exists": str(int(vision_exists)),
        "touch_path_exists": str(int(touch_exists)),
        "passed": str(int(not failures)),
        "failure_reasons": "|".join(failures),
    }


def build_record_rows(
    plan_rows: list[dict[str, str]],
    query_rows: list[dict[str, str]],
    required_probes: set[int],
) -> list[dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in query_rows:
        grouped[row["collection_slot"]].append(row)
    output = []
    for plan_row in plan_rows:
        items = grouped.get(plan_row["collection_slot"], [])
        observed_probes = {int(row["probe"]) for row in items}
        failures = []
        if observed_probes != required_probes:
            failures.append("required_probe_set_incomplete")
        if len(items) != len(required_probes):
            failures.append("duplicate_or_missing_probe_queries")
        if any(row["passed"] != "1" for row in items):
            failures.append("query_quality_failed")
        output.append(
            {
                "collection_slot": plan_row["collection_slot"],
                "planned_pair_id": plan_row["planned_pair_id"],
                "pair_variant": plan_row["pair_variant"],
                "pair_design": plan_row["pair_design"],
                "new_split": plan_row["new_split"],
                "new_record_id": plan_row["new_record_id"],
                "object_id": plan_row["object_id"],
                "contact_region_id": plan_row["contact_region_id"],
                "queries": str(len(items)),
                "probes": "|".join(
                    str(value) for value in sorted(observed_probes)
                ),
                "passed_queries": str(
                    sum(row["passed"] == "1" for row in items)
                ),
                "record_passed": str(int(not failures)),
                "failure_reasons": "|".join(failures),
            }
        )
    return output


def validate(config_path: str, section: str) -> dict:
    cfg = load_config(config_path)[section]
    plan_path = project_path(cfg["completed_plan_csv"])
    samples_path = project_path(cfg["new_samples_csv"])
    tracks_path = project_path(cfg["new_motion_tracks_csv"])
    for label, path in (
        ("completed plan", plan_path),
        ("new samples", samples_path),
        ("new motion tracks", tracks_path),
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"Phase4I.6 {label} is required: {path}"
            )
    plan_rows = read_csv_rows(plan_path)
    development = read_csv_rows(project_path(cfg["development_samples_csv"]))
    development_keys = {
        (row["split"], row["record_id"]) for row in development
    }
    development_ids = {row["record_id"] for row in development}
    sealed = final_holdout_keys(project_path(cfg["final_partition_csv"]))
    validate_completed_plan(
        plan_rows,
        development_keys,
        development_ids,
        sealed,
        int(cfg["expected_new_records"]),
    )

    planned_by_key = {record_key(row): row for row in plan_rows}
    samples = read_csv_rows(samples_path)
    selected_samples = [
        row
        for row in samples
        if (row["split"], row["record_id"]) in planned_by_key
    ]
    unplanned = sorted(
        {
            (row["split"], row["record_id"])
            for row in samples
            if (row["split"], row["record_id"]) not in planned_by_key
        }
    )
    if unplanned and not bool(cfg["allow_unplanned_records"]):
        raise RuntimeError(
            f"Phase4I.6 samples contain unplanned records: {unplanned[:3]}"
        )
    tracks = read_trajectory_tracks(tracks_path)
    query_rows = [
        evaluate_query(
            planned_by_key[(sample["split"], sample["record_id"])],
            sample,
            tracks,
            cfg,
        )
        for sample in selected_samples
        if int(sample["probe"]) in {int(value) for value in cfg["probe_focus"]}
    ]
    required_probes = {int(value) for value in cfg["probe_focus"]}
    record_rows = build_record_rows(
        plan_rows,
        query_rows,
        required_probes,
    )
    write_csv_rows(
        project_path(cfg["query_output_csv"]),
        query_rows,
        QUERY_FIELDS,
    )
    write_csv_rows(
        project_path(cfg["record_output_csv"]),
        record_rows,
        RECORD_FIELDS,
    )
    accepted_records = sum(
        row["record_passed"] == "1" for row in record_rows
    )
    accepted_queries = sum(row["passed"] == "1" for row in query_rows)
    expected_queries = int(cfg["expected_new_far_queries"])
    accepted = (
        accepted_records == int(cfg["expected_new_records"])
        and accepted_queries == expected_queries
        and len(query_rows) == expected_queries
    )
    summary = {
        "mode": "phase4i6_temporal_far_data_validation_v1",
        "accepted": accepted,
        "records": len(record_rows),
        "accepted_records": accepted_records,
        "queries": len(query_rows),
        "accepted_queries": accepted_queries,
        "probe_counts": dict(Counter(row["probe"] for row in query_rows)),
        "query_failure_counts": dict(
            Counter(
                reason
                for row in query_rows
                for reason in row["failure_reasons"].split("|")
                if reason
            )
        ),
        "record_failure_counts": dict(
            Counter(
                reason
                for row in record_rows
                for reason in row["failure_reasons"].split("|")
                if reason
            )
        ),
        "contract": {
            "expected_new_records": int(cfg["expected_new_records"]),
            "expected_new_far_queries": expected_queries,
            "probe_focus": sorted(required_probes),
            "minimum_real_point_count": int(
                cfg["minimum_real_point_count"]
            ),
            "minimum_history_span": int(cfg["minimum_history_span"]),
            "maximum_padding_ratio": float(cfg["maximum_padding_ratio"]),
            "maximum_frame_gap": int(cfg["maximum_frame_gap"]),
            "record_disjoint_from_development": True,
            "record_disjoint_from_final_holdout": True,
        },
        "integrity": {
            "query_tactile_input": False,
            "model_retrained": False,
            "threshold_retuned": False,
            "sealed_final_holdout_rows_read": 0,
            "development_validation_rows_read": 0,
        },
        "outputs": {
            "query_csv": str(project_path(cfg["query_output_csv"])),
            "record_csv": str(project_path(cfg["record_output_csv"])),
        },
        "next_action": (
            "build frozen Phase4I.6 train/OOF partitions and retrain TTC"
            if accepted
            else "repair rejected records before any TTC retraining"
        ),
    }
    write_json(project_path(cfg["metrics_json"]), summary)
    print(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate completed Phase4I.6 temporal-far data."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--section",
        default="phase4i6_temporal_far_data_validation_v1",
    )
    args = parser.parse_args()
    report = validate(args.config, args.section)
    if not report["accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
