"""Plan record-disjoint temporal-far data collection from strict OOF misses."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from .config import load_config, project_path
from .phase4h_dino_adaptation import final_holdout_keys
from .utils import read_csv_rows, write_csv_rows, write_json


REFERENCE_FIELDS = [
    "reference_rank",
    "query_record_id",
    "query_image_name",
    "query_probe",
    "oof_fold",
    "phase4i3_changed_cache",
    "retrieval_outcome",
    "mae_or_ssim_harm",
    "high_confidence_false_negative",
    "far_miss_margin",
    "true_ttc",
    "temporal_predicted_ttc",
    "temporal_ttc_absolute_error",
    "temporal_ttc_underestimate",
    "base_predicted_ttc",
    "ttc_entropy",
    "trajectory_stability",
    "visual_first_last_cosine_distance",
    "corridor_approach_reduction",
    "priority_score",
    "case_types",
    "selected_as_collection_reference",
    "acquisition_reason",
]

COLLECTION_FIELDS = [
    "collection_slot",
    "new_record_id",
    "new_split",
    "object_id",
    "contact_region_id",
    "planned_pair_id",
    "pair_variant",
    "pair_design",
    "reference_record_id",
    "reference_query_image_name",
    "probe_focus",
    "shared_motion_profile",
    "paired_variant",
    "target_failure_mode",
    "minimum_real_point_count",
    "minimum_history_span",
    "maximum_padding_ratio",
    "maximum_frame_gap",
    "record_disjoint_required",
    "reference_record_reuse_forbidden",
    "same_object_region_pair_required",
    "expected_far_queries",
    "collection_status",
    "notes",
]


def load_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def finite(row: dict[str, str], key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise RuntimeError(
            f"Nonfinite {key} for {row.get('query_image_name', '<unknown>')}"
        )
    return value


def query_split(row: dict[str, str]) -> str:
    prefix = row["query_image_name"].split("_", 1)[0]
    if not prefix:
        raise RuntimeError(
            f"Cannot infer split from {row['query_image_name']}"
        )
    return prefix


def assert_source_integrity(
    rows: list[dict[str, str]],
    report: dict,
    cfg: dict,
) -> None:
    expected = int(cfg["expected_false_negative_queries"])
    if len(rows) != expected:
        raise RuntimeError(
            "Phase4I.6 source query count changed: "
            f"expected={expected}, actual={len(rows)}"
        )
    names = [row["query_image_name"] for row in rows]
    if len(names) != len(set(names)):
        raise RuntimeError("Phase4I.6 requires unique query image names")
    expected_probes = {str(value) for value in cfg["probe_focus"]}
    unexpected = sorted(
        {
            row["query_probe"]
            for row in rows
            if row["query_probe"] not in expected_probes
        }
    )
    if unexpected:
        raise RuntimeError(
            f"Phase4I.6 received non-far probes: {unexpected}"
        )
    if int(report["false_negative_queries"]) != expected:
        raise RuntimeError("Phase4I.5b report/query count mismatch")
    if int(report["phase4i3_changed_cache_queries"]) != int(
        cfg["expected_changed_cache_queries"]
    ):
        raise RuntimeError("Phase4I.5b changed-cache count drifted")
    if int(report["identity_unchanged_queries"]) != int(
        cfg["expected_identity_unchanged_queries"]
    ):
        raise RuntimeError("Phase4I.5b identity count drifted")
    integrity = report.get("integrity", {})
    if (
        int(integrity.get("sealed_final_holdout_rows_read", -1)) != 0
        or int(integrity.get("development_validation_rows_read", -1)) != 0
    ):
        raise RuntimeError("Phase4I.6 refuses a non-development OOF audit")

    sealed = final_holdout_keys(project_path(cfg["final_partition_csv"]))
    leaked = [
        row["query_image_name"]
        for row in rows
        if (query_split(row), row["query_record_id"]) in sealed
    ]
    if leaked:
        raise RuntimeError(
            f"Phase4I.6 refuses sealed final-holdout rows: {leaked[:3]}"
        )


def classify_reference(
    row: dict[str, str],
    cfg: dict,
) -> dict[str, str]:
    changed_harm = (
        int(row["phase4i3_changed_cache"]) == 1
        and int(row["mae_or_ssim_harm"]) == 1
    )
    high_confidence = int(row["high_confidence_false_negative"]) == 1
    underestimate = max(
        finite(row, "true_ttc")
        - finite(row, "temporal_predicted_ttc"),
        0.0,
    )
    severe_underestimate = underestimate >= float(
        cfg["severe_ttc_underestimate_frames"]
    )
    probe100 = int(row["query_probe"]) == 100
    weights = cfg["priority_weights"]
    priority = (
        float(weights["changed_harm"]) * changed_harm
        + float(weights["high_confidence_false_negative"]) * high_confidence
        + float(weights["severe_ttc_underestimate"]) * severe_underestimate
        + float(weights["underestimate_per_frame"]) * underestimate
        + float(weights["probe100"]) * probe100
    )
    case_types = []
    reasons = []
    if changed_harm:
        case_types.append("changed_cache_mae_or_ssim_harm")
        reasons.append("reproduce_true_far_progress_error_that_harmed_retrieval")
    if high_confidence:
        case_types.append("high_confidence_far_false_negative")
        reasons.append("separate_confident_far_miss_from_near_mid")
    if severe_underestimate:
        case_types.append("severe_ttc_underestimate")
        reasons.append("contrast_visually_similar_states_with_different_ttc")
    if probe100:
        case_types.append("probe100")
    if not case_types:
        case_types.append("ordinary_far_false_negative")
        reasons.append("increase_record_disjoint_far_coverage")
    return {
        "reference_rank": "",
        "query_record_id": row["query_record_id"],
        "query_image_name": row["query_image_name"],
        "query_probe": row["query_probe"],
        "oof_fold": row["oof_fold"],
        "phase4i3_changed_cache": row["phase4i3_changed_cache"],
        "retrieval_outcome": row["retrieval_outcome"],
        "mae_or_ssim_harm": row["mae_or_ssim_harm"],
        "high_confidence_false_negative": row[
            "high_confidence_false_negative"
        ],
        "far_miss_margin": f"{finite(row, 'far_miss_margin'):.9f}",
        "true_ttc": f"{finite(row, 'true_ttc'):.9f}",
        "temporal_predicted_ttc": (
            f"{finite(row, 'temporal_predicted_ttc'):.9f}"
        ),
        "temporal_ttc_absolute_error": (
            f"{finite(row, 'temporal_ttc_absolute_error'):.9f}"
        ),
        "temporal_ttc_underestimate": f"{underestimate:.9f}",
        "base_predicted_ttc": f"{finite(row, 'base_predicted_ttc'):.9f}",
        "ttc_entropy": f"{finite(row, 'ttc_entropy'):.9f}",
        "trajectory_stability": (
            f"{finite(row, 'trajectory_stability'):.9f}"
        ),
        "visual_first_last_cosine_distance": (
            f"{finite(row, 'visual_first_last_cosine_distance'):.9f}"
        ),
        "corridor_approach_reduction": (
            f"{finite(row, 'corridor_approach_reduction'):.9f}"
        ),
        "priority_score": f"{priority:.9f}",
        "case_types": "|".join(case_types),
        "selected_as_collection_reference": "0",
        "acquisition_reason": "|".join(reasons),
    }


def build_reference_rows(
    rows: list[dict[str, str]],
    cfg: dict,
) -> list[dict[str, str]]:
    output = [classify_reference(row, cfg) for row in rows]
    output.sort(
        key=lambda row: (
            -float(row["priority_score"]),
            -float(row["temporal_ttc_underestimate"]),
            row["query_image_name"],
        )
    )
    selected_records = set()
    target = int(cfg["target_reference_records"])
    for rank, row in enumerate(output, start=1):
        row["reference_rank"] = str(rank)
        record_id = row["query_record_id"]
        if record_id not in selected_records and len(selected_records) < target:
            selected_records.add(record_id)
            row["selected_as_collection_reference"] = "1"
    if len(selected_records) != target:
        raise RuntimeError(
            "Phase4I.6 could not select the configured number of "
            f"reference records: {len(selected_records)} != {target}"
        )
    return output


def build_collection_plan(
    reference_rows: list[dict[str, str]],
    cfg: dict,
) -> list[dict[str, str]]:
    references = [
        row
        for row in reference_rows
        if row["selected_as_collection_reference"] == "1"
    ]
    target_records = int(cfg["target_new_records"])
    if target_records % 2:
        raise RuntimeError(
            "Phase4I.6 target_new_records must be even for paired collection"
        )
    pair_designs = list(cfg["pair_designs"])
    motion_profiles = [
        str(value) for value in cfg["shared_motion_profiles"]
    ]
    if not references or not pair_designs or not motion_profiles:
        raise RuntimeError("Phase4I.6 collection recipes cannot be empty")

    output = []
    for index in range(target_records):
        pair_index = index // 2
        variant_index = index % 2
        pair = pair_designs[pair_index % len(pair_designs)]
        reference = references[pair_index % len(references)]
        variant = "A" if variant_index == 0 else "B"
        output.append(
            {
                "collection_slot": f"{index + 1:03d}",
                "new_record_id": "",
                "new_split": "",
                "object_id": "",
                "contact_region_id": "",
                "planned_pair_id": f"pair_{pair_index + 1:03d}",
                "pair_variant": variant,
                "pair_design": str(pair["name"]),
                "reference_record_id": reference["query_record_id"],
                "reference_query_image_name": reference["query_image_name"],
                "probe_focus": "|".join(
                    str(value) for value in cfg["probe_focus"]
                ),
                "shared_motion_profile": motion_profiles[
                    pair_index % len(motion_profiles)
                ],
                "paired_variant": str(
                    pair["variant_a"]
                    if variant_index == 0
                    else pair["variant_b"]
                ),
                "target_failure_mode": (
                    "true_far_predicted_as_near_or_mid"
                ),
                "minimum_real_point_count": str(
                    cfg["minimum_real_point_count"]
                ),
                "minimum_history_span": str(cfg["minimum_history_span"]),
                "maximum_padding_ratio": str(
                    cfg["maximum_padding_ratio"]
                ),
                "maximum_frame_gap": str(cfg["maximum_frame_gap"]),
                "record_disjoint_required": "1",
                "reference_record_reuse_forbidden": "1",
                "same_object_region_pair_required": "1",
                "expected_far_queries": str(len(cfg["probe_focus"])),
                "collection_status": "planned",
                "notes": "",
            }
        )
    validate_collection_plan(output)
    return output


def validate_collection_plan(rows: list[dict[str, str]]) -> None:
    pairs: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        pairs.setdefault(row["planned_pair_id"], []).append(row)
    for pair_id, pair_rows in pairs.items():
        if len(pair_rows) != 2:
            raise RuntimeError(
                f"Phase4I.6 {pair_id} must contain exactly two records"
            )
        if {row["pair_variant"] for row in pair_rows} != {"A", "B"}:
            raise RuntimeError(
                f"Phase4I.6 {pair_id} must contain A and B variants"
            )
        for key in (
            "pair_design",
            "reference_record_id",
            "reference_query_image_name",
            "shared_motion_profile",
            "probe_focus",
        ):
            if len({row[key] for row in pair_rows}) != 1:
                raise RuntimeError(
                    f"Phase4I.6 {pair_id} does not share {key}"
                )
        if len({row["paired_variant"] for row in pair_rows}) != 2:
            raise RuntimeError(
                f"Phase4I.6 {pair_id} must change paired_variant"
            )


def plan(config_path: str, section: str) -> dict:
    cfg = load_config(config_path)[section]
    rows = read_csv_rows(project_path(cfg["audit_query_csv"]))
    report = load_json(project_path(cfg["audit_metrics_json"]))
    assert_source_integrity(rows, report, cfg)
    references = build_reference_rows(rows, cfg)
    collection = build_collection_plan(references, cfg)

    planned_far_queries = sum(
        int(row["expected_far_queries"]) for row in collection
    )
    if planned_far_queries < int(cfg["target_new_far_queries"]):
        raise RuntimeError(
            "Phase4I.6 collection plan misses its far-query target: "
            f"{planned_far_queries} < {cfg['target_new_far_queries']}"
        )
    write_csv_rows(
        project_path(cfg["reference_output_csv"]),
        references,
        REFERENCE_FIELDS,
    )
    write_csv_rows(
        project_path(cfg["collection_plan_csv"]),
        collection,
        COLLECTION_FIELDS,
    )

    selected = [
        row
        for row in references
        if row["selected_as_collection_reference"] == "1"
    ]
    summary = {
        "mode": "phase4i6_temporal_far_collection_plan_v1",
        "source_false_negative_queries": len(rows),
        "source_false_negative_records": len(
            {row["query_record_id"] for row in rows}
        ),
        "selected_reference_records": len(selected),
        "selected_reference_queries": len(selected),
        "source_case_counts": {
            "changed_cache_harm": sum(
                int(row["phase4i3_changed_cache"]) == 1
                and int(row["mae_or_ssim_harm"]) == 1
                for row in references
            ),
            "high_confidence_false_negative": sum(
                int(row["high_confidence_false_negative"])
                for row in references
            ),
            "severe_ttc_underestimate": sum(
                float(row["temporal_ttc_underestimate"])
                >= float(cfg["severe_ttc_underestimate_frames"])
                for row in references
            ),
        },
        "collection_contract": {
            "target_new_record_disjoint_records": int(
                cfg["target_new_records"]
            ),
            "target_new_far_queries": int(cfg["target_new_far_queries"]),
            "collection_slots_written": len(collection),
            "planned_far_queries": planned_far_queries,
            "planned_pairs": len(collection) // 2,
            "probe_focus": [int(value) for value in cfg["probe_focus"]],
            "shared_motion_profile_counts": dict(
                Counter(
                    row["shared_motion_profile"] for row in collection
                )
            ),
            "pair_design_counts": dict(
                Counter(row["pair_design"] for row in collection)
            ),
            "minimum_real_point_count": int(
                cfg["minimum_real_point_count"]
            ),
            "minimum_history_span": int(cfg["minimum_history_span"]),
            "maximum_padding_ratio": float(cfg["maximum_padding_ratio"]),
            "maximum_frame_gap": int(cfg["maximum_frame_gap"]),
            "new_record_ids_must_be_unique": True,
            "new_record_ids_must_not_reuse_reference_ids": True,
        },
        "integrity": {
            "source": "frozen strict Phase4I.5b development OOF only",
            "model_retrained": False,
            "threshold_retuned": False,
            "query_tactile_input": False,
            "sealed_final_holdout_rows_read": 0,
            "development_validation_rows_read": 0,
        },
        "outputs": {
            "reference_csv": str(project_path(cfg["reference_output_csv"])),
            "collection_plan_csv": str(
                project_path(cfg["collection_plan_csv"])
            ),
        },
        "next_action": (
            "fill new_record_id and collect/select record-disjoint paired "
            "temporal far sequences; validate the 32-frame trajectory "
            "contract before retraining TTC"
        ),
    }
    write_json(project_path(cfg["metrics_json"]), summary)
    print(summary)
    for row in selected[: int(cfg["print_reference_limit"])]:
        print(
            row["query_record_id"],
            row["query_image_name"],
            "probe=",
            row["query_probe"],
            "underestimate=",
            row["temporal_ttc_underestimate"],
            "cases=",
            row["case_types"],
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plan record-disjoint Phase4I.6 temporal-far data collection."
        )
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--section",
        default="phase4i6_temporal_far_collection_plan_v1",
    )
    args = parser.parse_args()
    plan(args.config, args.section)


if __name__ == "__main__":
    main()
