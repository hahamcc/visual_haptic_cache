"""Build the reviewed Phase4I.6 TTC increment with pair-grouped OOF folds."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .config import load_config, project_path
from .phase4h_dino_adaptation import final_holdout_keys
from .select_phase4i6e_visual_motion_pairs import (
    BIN_FIELDS,
    MOTION_FIELDS,
    merge_passed_records,
    quantile_bin_assignments,
    sample_index,
    standardized_motion,
)
from .temporal_progress import (
    masked_trajectory_features,
    read_trajectory_tracks,
)
from .utils import read_csv_rows, write_csv_rows, write_json


RECORD_FIELDS = [
    "selection_rank",
    "split",
    "record_id",
    "source_pool",
    "selection_source",
    "approved_pair_id",
    "pair_variant",
    "group_id",
    "oof_fold",
    "robust_motion_profile",
    *MOTION_FIELDS,
    *BIN_FIELDS.keys(),
]

APPROVED_PAIR_FIELDS = [
    "planned_pair_id",
    "record_a",
    "record_b",
    "image_a",
    "image_b",
    "detail_similarity",
    "context_similarity",
    "wide_similarity",
    "visual_similarity",
    "motion_distance",
    "reciprocal_visual_rank",
    "pair_design",
    "oof_fold",
    "approval_source",
    "approved",
]

SAMPLE_EXTRA_FIELDS = [
    "selection_source",
    "approved_pair_id",
    "pair_variant",
    "group_id",
    "oof_fold",
]


def approved_pair_records(
    pair_rows: list[dict[str, str]],
    approved_ids: list[str],
) -> tuple[list[dict[str, str]], dict[str, tuple[str, str]]]:
    by_id = {row["planned_pair_id"]: row for row in pair_rows}
    if len(by_id) != len(pair_rows):
        raise RuntimeError("Phase4I.6F V2 pair IDs are not unique")
    missing = [pair_id for pair_id in approved_ids if pair_id not in by_id]
    if missing:
        raise RuntimeError(
            f"Phase4I.6F approved pair IDs are missing: {missing[:3]}"
        )
    selected = [by_id[pair_id] for pair_id in approved_ids]
    pairs = {
        row["planned_pair_id"]: (row["record_a"], row["record_b"])
        for row in selected
    }
    records = [
        record_id
        for pair in pairs.values()
        for record_id in pair
    ]
    if len(records) != len(set(records)):
        raise RuntimeError(
            "Phase4I.6F approved pairs reuse a record"
        )
    return selected, pairs


def _coverage_bonus(
    index: int,
    assignments: dict[str, np.ndarray],
    counts: dict[str, Counter],
    target_per_bin: float,
) -> float:
    terms = []
    for name, values in assignments.items():
        bin_index = int(values[index])
        deficit = max(target_per_bin - counts[name][bin_index], 0.0)
        terms.append(deficit / max(target_per_bin, 1.0))
    return float(np.mean(terms)) if terms else 0.0


def select_diverse_fill(
    rows: list[dict[str, str]],
    seed_record_ids: set[str],
    target_records: int,
    assignments: dict[str, np.ndarray],
    standardized: np.ndarray,
    quantile_bins: int,
    distance_weight: float,
) -> list[int]:
    by_record = {row["record_id"]: index for index, row in enumerate(rows)}
    missing = sorted(seed_record_ids - set(by_record))
    if missing:
        raise RuntimeError(
            f"Phase4I.6F approved records are not robust: {missing[:3]}"
        )
    if target_records > len(rows):
        raise RuntimeError(
            f"Phase4I.6F requests {target_records} of only {len(rows)} records"
        )
    selected = {by_record[record_id] for record_id in seed_record_ids}
    counts = {name: Counter() for name in assignments}
    for index in selected:
        for name, values in assignments.items():
            counts[name][int(values[index])] += 1
    target_per_bin = target_records / float(quantile_bins)
    while len(selected) < target_records:
        best: tuple[float, float, str, int] | None = None
        for index, row in enumerate(rows):
            if index in selected:
                continue
            coverage = _coverage_bonus(
                index,
                assignments,
                counts,
                target_per_bin,
            )
            if selected:
                selected_indices = np.asarray(sorted(selected), dtype=np.int64)
                distances = np.linalg.norm(
                    standardized[selected_indices] - standardized[index],
                    axis=1,
                ) / math.sqrt(max(standardized.shape[1], 1))
                nearest_distance = float(distances.min())
            else:
                nearest_distance = 0.0
            score = coverage + distance_weight * math.tanh(nearest_distance)
            candidate = (
                score,
                nearest_distance,
                row["record_id"],
                index,
            )
            if best is None or candidate > best:
                best = candidate
        if best is None:
            raise RuntimeError("Phase4I.6F could not complete diverse fill")
        index = best[-1]
        selected.add(index)
        for name, values in assignments.items():
            counts[name][int(values[index])] += 1
    return sorted(selected)


def assign_grouped_folds(
    rows: list[dict[str, str]],
    group_by_record: dict[str, str],
    assignments: dict[str, np.ndarray],
    fold_count: int,
) -> dict[str, int]:
    if fold_count < 2:
        raise ValueError("Phase4I.6F needs at least two OOF folds")
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[group_by_record[row["record_id"]]].append(index)
    total = len(rows)
    maximum_fold_size = int(math.ceil(total / fold_count))
    overall_bins = {
        name: Counter(int(value) for value in values)
        for name, values in assignments.items()
    }
    fold_sizes = [0] * fold_count
    fold_bins = [
        {name: Counter() for name in assignments}
        for _ in range(fold_count)
    ]
    group_fold: dict[str, int] = {}
    ordered_groups = sorted(
        groups.items(),
        key=lambda item: (-len(item[1]), item[0]),
    )
    for group_id, indices in ordered_groups:
        best: tuple[float, int, int] | None = None
        for fold in range(fold_count):
            if fold_sizes[fold] + len(indices) > maximum_fold_size:
                continue
            bin_penalty = 0.0
            for name, values in assignments.items():
                added = Counter(int(values[index]) for index in indices)
                for bin_index, total_bin in overall_bins[name].items():
                    target = total_bin / float(fold_count)
                    after = (
                        fold_bins[fold][name][bin_index]
                        + added[bin_index]
                    )
                    bin_penalty += (after - target) ** 2 / max(target, 1.0)
            candidate = (
                fold_sizes[fold],
                float(bin_penalty),
                fold,
            )
            if best is None or candidate < best:
                best = candidate
        if best is None:
            raise RuntimeError(
                f"Phase4I.6F cannot place group {group_id} into a fold"
            )
        fold = best[-1]
        group_fold[group_id] = fold
        fold_sizes[fold] += len(indices)
        for name, values in assignments.items():
            fold_bins[fold][name].update(
                int(values[index]) for index in indices
            )
    if max(fold_sizes) - min(fold_sizes) > 1:
        raise RuntimeError(
            f"Phase4I.6F fold sizes are imbalanced: {fold_sizes}"
        )
    return {
        row["record_id"]: group_fold[group_by_record[row["record_id"]]]
        for row in rows
    }


def merge_tracks(
    paths: list[Path],
) -> tuple[
    dict[tuple[str, str], list[dict[str, float]]],
    list[dict[str, str]],
]:
    merged: dict[tuple[str, str], list[dict[str, float]]] = {}
    raw_rows = []
    for path in paths:
        parsed = read_trajectory_tracks(path)
        overlap = sorted(set(merged) & set(parsed))
        if overlap:
            raise RuntimeError(
                f"Phase4I.6F track pools overlap: {overlap[:3]}"
            )
        merged.update(parsed)
        raw_rows.extend(read_csv_rows(path))
    return merged, raw_rows


def validate_sample_quality(
    sample: dict[str, str],
    tracks: dict[tuple[str, str], list[dict[str, float]]],
    cfg: dict,
) -> dict[str, float]:
    _, _, quality = masked_trajectory_features(
        sample,
        tracks,
        history_frames=int(cfg["history_frames"]),
        spatial_scale_px=float(cfg["spatial_scale_px"]),
        speed_scale_px=float(cfg["speed_scale_px"]),
    )
    observed_ttc = (
        int(sample["contact_frame_detected"]) - int(sample["frame_id"])
    )
    failures = []
    if observed_ttc != int(sample["probe"]):
        failures.append("observed_ttc_mismatch")
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
    if bool(cfg["require_paths_exist"]) and not Path(
        sample["vision_path"]
    ).exists():
        failures.append("vision_path_missing")
    if bool(cfg["require_paths_exist"]) and not Path(
        sample["touch_path"]
    ).exists():
        failures.append("touch_path_missing")
    if failures:
        raise RuntimeError(
            f"Phase4I.6F sample {sample['image_name']} failed: "
            f"{'|'.join(failures)}"
        )
    return quality


def build(config_path: str, section: str) -> dict:
    cfg = load_config(config_path)[section]
    pool_names = [str(value) for value in cfg["pool_names"]]
    robust_paths = [project_path(path) for path in cfg["robust_record_csvs"]]
    sample_paths = [project_path(path) for path in cfg["candidate_sample_csvs"]]
    track_paths = [project_path(path) for path in cfg["candidate_track_csvs"]]
    partition_paths = [
        project_path(path) for path in cfg["candidate_partition_csvs"]
    ]
    if not (
        len(pool_names)
        == len(robust_paths)
        == len(sample_paths)
        == len(track_paths)
        == len(partition_paths)
    ):
        raise RuntimeError("Phase4I.6F pool configuration lengths differ")
    robust = merge_passed_records(
        [
            (name, read_csv_rows(path))
            for name, path in zip(pool_names, robust_paths, strict=True)
        ]
    )
    approved_rows, approved_pairs = approved_pair_records(
        read_csv_rows(project_path(cfg["v2_pair_csv"])),
        [str(value) for value in cfg["approved_pair_ids"]],
    )
    minimum_pair_motion = float(
        cfg["minimum_approved_pair_motion_distance"]
    )
    weak_pairs = [
        row["planned_pair_id"]
        for row in approved_rows
        if float(row["motion_distance"]) < minimum_pair_motion
    ]
    if weak_pairs:
        raise RuntimeError(
            f"Phase4I.6F approved pairs violate the motion contract: "
            f"{weak_pairs[:3]}"
        )
    approved_record_ids = {
        record_id
        for pair in approved_pairs.values()
        for record_id in pair
    }
    target_records = int(cfg["target_records"])
    quantile_bins = int(cfg["quantile_bins"])
    assignments_all, bin_edges = quantile_bin_assignments(
        robust,
        BIN_FIELDS,
        quantile_bins,
    )
    standardized = standardized_motion(robust)
    selected_indices = select_diverse_fill(
        robust,
        approved_record_ids,
        target_records,
        assignments_all,
        standardized,
        quantile_bins,
        float(cfg["fill_distance_weight"]),
    )
    selected = [robust[index] for index in selected_indices]
    selected_keys = {
        (row["split"], row["record_id"]) for row in selected
    }
    partition_rows = [
        row
        for path in partition_paths
        for row in read_csv_rows(path)
    ]
    partition_keys = {
        (row["split"], row["record_id"])
        for row in partition_rows
        if row["partition"] == "development_candidate"
    }
    outside_partition = sorted(selected_keys - partition_keys)
    if outside_partition:
        raise RuntimeError(
            "Phase4I.6F selected records outside frozen development "
            f"candidate partitions: {outside_partition[:3]}"
        )
    sealed = final_holdout_keys(project_path(cfg["final_partition_csv"]))
    overlap = sorted(selected_keys & sealed)
    if overlap:
        raise RuntimeError(
            f"Phase4I.6F selected final-holdout records: {overlap[:3]}"
        )
    selected_assignments = {
        name: values[np.asarray(selected_indices, dtype=np.int64)]
        for name, values in assignments_all.items()
    }
    pair_by_record = {}
    pair_variant = {}
    for pair_id, (record_a, record_b) in approved_pairs.items():
        pair_by_record[record_a] = pair_id
        pair_by_record[record_b] = pair_id
        pair_variant[record_a] = "A"
        pair_variant[record_b] = "B"
    group_by_record = {
        row["record_id"]: pair_by_record.get(
            row["record_id"],
            f"single_{row['record_id']}",
        )
        for row in selected
    }
    fold_by_record = assign_grouped_folds(
        selected,
        group_by_record,
        selected_assignments,
        int(cfg["fold_count"]),
    )
    probes = {int(value) for value in cfg["probe_focus"]}
    samples, all_sample_rows = sample_index(
        [read_csv_rows(path) for path in sample_paths],
        selected_keys,
        probes,
    )
    tracks, all_track_rows = merge_tracks(track_paths)
    record_rows = []
    selected_sample_rows = []
    for record_index, row in enumerate(selected):
        rank = record_index + 1
        record_id = row["record_id"]
        pair_id = pair_by_record.get(record_id, "")
        selection_source = (
            "approved_visual_motion_pair"
            if pair_id
            else "continuous_motion_diversity_fill"
        )
        record_rows.append(
            {
                "selection_rank": str(rank),
                "split": row["split"],
                "record_id": record_id,
                "source_pool": row["source_pool"],
                "selection_source": selection_source,
                "approved_pair_id": pair_id,
                "pair_variant": pair_variant.get(record_id, ""),
                "group_id": group_by_record[record_id],
                "oof_fold": str(fold_by_record[record_id]),
                "robust_motion_profile": row["robust_motion_profile"],
                **{field: row[field] for field in MOTION_FIELDS},
                **{
                    name: str(int(values[record_index]))
                    for name, values in selected_assignments.items()
                },
            }
        )
        for probe in sorted(probes):
            sample = dict(samples[(row["split"], record_id, probe)])
            validate_sample_quality(sample, tracks, cfg)
            sample.update(
                {
                    "selection_source": selection_source,
                    "approved_pair_id": pair_id,
                    "pair_variant": pair_variant.get(record_id, ""),
                    "group_id": group_by_record[record_id],
                    "oof_fold": str(fold_by_record[record_id]),
                }
            )
            selected_sample_rows.append(sample)
    approved_pair_output = []
    for row in approved_rows:
        pair_id = row["planned_pair_id"]
        record_a, record_b = approved_pairs[pair_id]
        if fold_by_record[record_a] != fold_by_record[record_b]:
            raise RuntimeError(
                f"Phase4I.6F pair {pair_id} crosses OOF folds"
            )
        approved_pair_output.append(
            {
                **{
                    field: row.get(field, "")
                    for field in APPROVED_PAIR_FIELDS
                    if field
                    not in {"oof_fold", "approval_source", "approved"}
                },
                "oof_fold": str(fold_by_record[record_a]),
                "approval_source": str(cfg["approval_source"]),
                "approved": "1",
            }
        )
    fold_pair_counts = {
        str(fold): sum(
            int(row["oof_fold"]) == fold
            for row in approved_pair_output
        )
        for fold in range(int(cfg["fold_count"]))
    }
    if max(fold_pair_counts.values()) - min(
        fold_pair_counts.values()
    ) > 1:
        raise RuntimeError(
            "Phase4I.6F approved-pair folds are imbalanced: "
            f"{fold_pair_counts}"
        )
    selected_track_rows = [
        row
        for row in all_track_rows
        if (row["split"], row["record_id"]) in selected_keys
    ]
    if not selected_track_rows:
        raise RuntimeError("Phase4I.6F selected track output is empty")
    write_csv_rows(
        project_path(cfg["record_output_csv"]),
        record_rows,
        RECORD_FIELDS,
    )
    write_csv_rows(
        project_path(cfg["approved_pair_output_csv"]),
        approved_pair_output,
        APPROVED_PAIR_FIELDS,
    )
    sample_fields = list(all_sample_rows[0].keys()) + [
        field
        for field in SAMPLE_EXTRA_FIELDS
        if field not in all_sample_rows[0]
    ]
    write_csv_rows(
        project_path(cfg["sample_output_csv"]),
        selected_sample_rows,
        sample_fields,
    )
    write_csv_rows(
        project_path(cfg["track_output_csv"]),
        selected_track_rows,
        list(all_track_rows[0].keys()),
    )
    identity = {
        "approved_pair_ids": list(cfg["approved_pair_ids"]),
        "selected_records": [
            [row["split"], row["record_id"], fold_by_record[row["record_id"]]]
            for row in selected
        ],
        "probe_focus": sorted(probes),
        "quality_contract": {
            key: cfg[key]
            for key in (
                "history_frames",
                "minimum_real_point_count",
                "minimum_history_span",
                "maximum_padding_ratio",
                "maximum_frame_gap",
            )
        },
    }
    fingerprint = hashlib.sha256(
        json.dumps(
            identity,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    summary = {
        "mode": "phase4i6f_ttc_increment_v1",
        "robust_available_records": len(robust),
        "approved_pairs": len(approved_pair_output),
        "approved_pair_records": len(approved_record_ids),
        "diversity_fill_records": target_records - len(approved_record_ids),
        "selected_records": len(record_rows),
        "selected_far_queries": len(selected_sample_rows),
        "selected_track_rows": len(selected_track_rows),
        "fold_record_counts": dict(
            Counter(row["oof_fold"] for row in record_rows)
        ),
        "fold_pair_counts": fold_pair_counts,
        "source_pool_counts": dict(
            Counter(row["source_pool"] for row in record_rows)
        ),
        "selection_source_counts": dict(
            Counter(row["selection_source"] for row in record_rows)
        ),
        "motion_profile_counts": dict(
            Counter(row["robust_motion_profile"] for row in record_rows)
        ),
        "selected_quantile_bin_counts": {
            name: dict(Counter(int(row[name]) for row in record_rows))
            for name in BIN_FIELDS
        },
        "quantile_edges": bin_edges,
        "fingerprint": fingerprint,
        "quality_contract": identity["quality_contract"],
        "approved_pair_motion_contract": {
            "minimum_distance": minimum_pair_motion,
            "violations": 0,
        },
        "integrity": {
            "selected_records_in_frozen_candidate_partitions": True,
            "approved_pair_variants_share_oof_fold": True,
            "query_tactile_input": False,
            "tactile_images_read": 0,
            "future_visual_frames_used": False,
            "sealed_final_holdout_rows_read": 0,
            "development_validation_outcomes_read": 0,
        },
        "outputs": {
            "record_csv": str(project_path(cfg["record_output_csv"])),
            "approved_pair_csv": str(
                project_path(cfg["approved_pair_output_csv"])
            ),
            "sample_csv": str(project_path(cfg["sample_output_csv"])),
            "track_csv": str(project_path(cfg["track_output_csv"])),
        },
        "next_action": (
            "freeze this increment and train the next TTC model with all "
            "records as primary supervision and approved pairs as auxiliary "
            "pairwise supervision"
        ),
    }
    write_json(project_path(cfg["metrics_json"]), summary)
    print(summary)
    for fold in range(int(cfg["fold_count"])):
        fold_rows = [
            row for row in record_rows if int(row["oof_fold"]) == fold
        ]
        print(
            "fold",
            fold,
            "records=",
            len(fold_rows),
            "pairs=",
            sum(
                1
                for row in approved_pair_output
                if int(row["oof_fold"]) == fold
            ),
            "profiles=",
            dict(
                Counter(row["robust_motion_profile"] for row in fold_rows)
            ),
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the reviewed Phase4I.6F TTC increment."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--section",
        default="phase4i6f_ttc_increment_v1",
    )
    args = parser.parse_args()
    build(args.config, args.section)


if __name__ == "__main__":
    main()
