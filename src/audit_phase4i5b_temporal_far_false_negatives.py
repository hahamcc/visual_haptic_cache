"""Audit frozen Phase4I.5a temporal false-negative far queries."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .config import load_config, project_path
from .phase4h_dino_adaptation import assert_development_only
from .train_phase4i4_far_risk_gate_oof import assert_csv_schema
from .train_phase4i5_temporal_progress_oof import load_temporal_cache
from .train_phase4i_factorized_residual_cascade import METRICS
from .utils import read_csv_rows, write_csv_rows, write_json


QUERY_FIELDS = [
    "query_record_id",
    "query_image_name",
    "query_probe",
    "oof_fold",
    "far_risk_probability",
    "far_risk_threshold",
    "far_miss_margin",
    "miss_margin_bin",
    "temporal_predicted_ttc",
    "true_ttc",
    "temporal_ttc_absolute_error",
    "temporal_ttc_error_bin",
    "base_predicted_ttc",
    "base_ttc_absolute_error",
    "temporal_minus_base_ttc_absolute_error",
    "ttc_entropy",
    "trajectory_stability",
    "motion_speed",
    "motion_cumulative",
    "visual_first_last_cosine_distance",
    "visual_mean_step_cosine_distance",
    "visual_max_step_cosine_distance",
    "visual_change_quartile",
    "corridor_start_distance",
    "corridor_end_distance",
    "corridor_approach_reduction",
    "approach_direction",
    "temporal_padding_ratio",
    "v1_selected_cache_image_name",
    "phase4i3_selected_cache_image_name",
    "phase4i3_changed_cache",
    "v1_oracle_rank",
    "phase4i3_oracle_rank",
    "oracle_top1_delta",
    "v1_tactile_diff_mae",
    "phase4i3_tactile_diff_mae",
    "mae_delta",
    "v1_tactile_ssim",
    "phase4i3_tactile_ssim",
    "ssim_delta",
    "v1_tactile_mask_iou",
    "phase4i3_tactile_mask_iou",
    "iou_delta",
    "retrieval_outcome",
    "strict_triple_win",
    "spatial_only_gain",
    "mae_or_ssim_harm",
    "triple_harm",
    "high_confidence_false_negative",
]

SLICE_FIELDS = [
    "dimension",
    "bucket",
    "queries",
    "records",
    "changed_cache_rate",
    "strict_triple_win_rate",
    "mae_or_ssim_harm_rate",
    "triple_harm_rate",
    "mean_mae_delta",
    "mean_ssim_delta",
    "mean_iou_delta",
    "mean_far_risk_probability",
    "mean_far_miss_margin",
    "mean_temporal_ttc_absolute_error",
    "mean_base_ttc_absolute_error",
    "mean_temporal_minus_base_ttc_absolute_error",
    "mean_visual_first_last_cosine_distance",
    "mean_corridor_approach_reduction",
]

RECORD_FIELDS = [
    "query_record_id",
    "queries",
    "probe75_queries",
    "probe100_queries",
    "changed_cache_queries",
    "strict_triple_win_queries",
    "spatial_only_gain_queries",
    "mae_or_ssim_harm_queries",
    "triple_harm_queries",
    "high_confidence_false_negatives",
    "mean_mae_delta",
    "mean_ssim_delta",
    "mean_iou_delta",
    "mean_far_miss_margin",
    "mean_temporal_ttc_absolute_error",
    "mean_base_ttc_absolute_error",
    "mean_temporal_minus_base_ttc_absolute_error",
    "mean_visual_first_last_cosine_distance",
    "mean_corridor_approach_reduction",
    "priority_score",
    "representative_query_image_names",
]


def unique_by_query(
    rows: list[dict[str, str]],
    label: str,
) -> dict[str, dict[str, str]]:
    output = {}
    for row in rows:
        name = row["query_image_name"]
        if name in output:
            raise RuntimeError(f"Duplicate {label} query: {name}")
        output[name] = row
    return output


def finite(row: dict[str, str], key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise RuntimeError(
            f"Nonfinite {key} for {row.get('query_image_name', '<unknown>')}"
        )
    return value


def retrieval_outcome(
    mae_delta: float,
    ssim_delta: float,
    iou_delta: float,
    epsilon: float,
) -> str:
    mae_harm = mae_delta > epsilon
    ssim_harm = ssim_delta < -epsilon
    iou_harm = iou_delta < -epsilon
    if (
        mae_delta < -epsilon
        and ssim_delta >= -epsilon
        and iou_delta >= -epsilon
    ):
        return "strict_triple_win"
    if mae_harm and ssim_harm and iou_harm:
        return "triple_harm"
    if iou_delta > epsilon and (mae_harm or ssim_harm):
        return "spatial_only_gain"
    if mae_harm or ssim_harm:
        return "mae_or_ssim_harm"
    return "mixed_or_neutral"


def cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-12:
        return 0.0
    similarity = float(np.dot(left, right) / denominator)
    return float(1.0 - np.clip(similarity, -1.0, 1.0))


def temporal_visual_diagnostics(
    visual: np.ndarray,
    geometry: np.ndarray,
    valid: np.ndarray,
    padding: np.ndarray,
) -> dict[str, float]:
    indices = np.flatnonzero(valid > 0.5)
    if len(indices) < 2:
        raise RuntimeError("Phase4I.5b requires at least two temporal frames")
    frame_values = visual[indices].astype(np.float32)
    step = np.asarray(
        [
            cosine_distance(left, right)
            for left, right in zip(
                frame_values[:-1],
                frame_values[1:],
                strict=True,
            )
        ],
        dtype=np.float32,
    )
    start_distance = float(geometry[indices[0], 6])
    end_distance = float(geometry[indices[-1], 6])
    return {
        "visual_first_last_cosine_distance": cosine_distance(
            frame_values[0],
            frame_values[-1],
        ),
        "visual_mean_step_cosine_distance": float(step.mean()),
        "visual_max_step_cosine_distance": float(step.max()),
        "corridor_start_distance": start_distance,
        "corridor_end_distance": end_distance,
        "corridor_approach_reduction": start_distance - end_distance,
        "temporal_padding_ratio": float(padding[indices].mean()),
    }


def numeric_bin(value: float, first: float, second: float) -> str:
    if value <= first:
        return f"le_{first:g}"
    if value <= second:
        return f"{first:g}_to_{second:g}"
    return f"gt_{second:g}"


def quartile_labels(values: np.ndarray) -> tuple[np.ndarray, list[float]]:
    edges = np.unique(np.quantile(values, [0.25, 0.5, 0.75]))
    labels = np.digitize(values, edges)
    return labels, [float(value) for value in edges]


def slice_row(
    rows: list[dict[str, str]],
    dimension: str,
    bucket: str,
) -> dict[str, str]:
    if not rows:
        raise ValueError("Phase4I.5b cannot summarize an empty slice")

    def mean(key: str) -> float:
        return float(np.mean([float(row[key]) for row in rows]))

    return {
        "dimension": dimension,
        "bucket": bucket,
        "queries": str(len(rows)),
        "records": str(len({row["query_record_id"] for row in rows})),
        "changed_cache_rate": f"{mean('phase4i3_changed_cache'):.9f}",
        "strict_triple_win_rate": f"{mean('strict_triple_win'):.9f}",
        "mae_or_ssim_harm_rate": f"{mean('mae_or_ssim_harm'):.9f}",
        "triple_harm_rate": f"{mean('triple_harm'):.9f}",
        "mean_mae_delta": f"{mean('mae_delta'):.9f}",
        "mean_ssim_delta": f"{mean('ssim_delta'):.9f}",
        "mean_iou_delta": f"{mean('iou_delta'):.9f}",
        "mean_far_risk_probability": (
            f"{mean('far_risk_probability'):.9f}"
        ),
        "mean_far_miss_margin": f"{mean('far_miss_margin'):.9f}",
        "mean_temporal_ttc_absolute_error": (
            f"{mean('temporal_ttc_absolute_error'):.9f}"
        ),
        "mean_base_ttc_absolute_error": (
            f"{mean('base_ttc_absolute_error'):.9f}"
        ),
        "mean_temporal_minus_base_ttc_absolute_error": (
            f"{mean('temporal_minus_base_ttc_absolute_error'):.9f}"
        ),
        "mean_visual_first_last_cosine_distance": (
            f"{mean('visual_first_last_cosine_distance'):.9f}"
        ),
        "mean_corridor_approach_reduction": (
            f"{mean('corridor_approach_reduction'):.9f}"
        ),
    }


def build_slices(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    output = [slice_row(rows, "all", "all")]
    dimensions = (
        ("probe", "query_probe"),
        ("oof_fold", "oof_fold"),
        ("retrieval_outcome", "retrieval_outcome"),
        ("miss_margin_bin", "miss_margin_bin"),
        ("temporal_ttc_error_bin", "temporal_ttc_error_bin"),
        ("visual_change_quartile", "visual_change_quartile"),
        ("approach_direction", "approach_direction"),
    )
    for dimension, key in dimensions:
        groups: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            groups[row[key]].append(row)
        for bucket in sorted(groups):
            output.append(slice_row(groups[bucket], dimension, bucket))
    return output


def record_rows(
    rows: list[dict[str, str]],
    maximum_records: int,
) -> list[dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["query_record_id"]].append(row)
    output = []
    for record_id, items in grouped.items():
        harmful = sum(int(row["mae_or_ssim_harm"]) for row in items)
        triple = sum(int(row["triple_harm"]) for row in items)
        high_confidence = sum(
            int(row["high_confidence_false_negative"]) for row in items
        )
        priority = 3 * triple + 2 * harmful + high_confidence

        def mean(key: str) -> float:
            return float(np.mean([float(row[key]) for row in items]))

        ranked = sorted(
            items,
            key=lambda row: (
                -int(row["triple_harm"]),
                -int(row["mae_or_ssim_harm"]),
                -float(row["mae_delta"]),
                row["query_image_name"],
            ),
        )
        output.append(
            {
                "query_record_id": record_id,
                "queries": str(len(items)),
                "probe75_queries": str(
                    sum(int(row["query_probe"]) == 75 for row in items)
                ),
                "probe100_queries": str(
                    sum(int(row["query_probe"]) == 100 for row in items)
                ),
                "changed_cache_queries": str(
                    sum(int(row["phase4i3_changed_cache"]) for row in items)
                ),
                "strict_triple_win_queries": str(
                    sum(int(row["strict_triple_win"]) for row in items)
                ),
                "spatial_only_gain_queries": str(
                    sum(int(row["spatial_only_gain"]) for row in items)
                ),
                "mae_or_ssim_harm_queries": str(harmful),
                "triple_harm_queries": str(triple),
                "high_confidence_false_negatives": str(high_confidence),
                "mean_mae_delta": f"{mean('mae_delta'):.9f}",
                "mean_ssim_delta": f"{mean('ssim_delta'):.9f}",
                "mean_iou_delta": f"{mean('iou_delta'):.9f}",
                "mean_far_miss_margin": f"{mean('far_miss_margin'):.9f}",
                "mean_temporal_ttc_absolute_error": (
                    f"{mean('temporal_ttc_absolute_error'):.9f}"
                ),
                "mean_base_ttc_absolute_error": (
                    f"{mean('base_ttc_absolute_error'):.9f}"
                ),
                "mean_temporal_minus_base_ttc_absolute_error": (
                    f"{mean('temporal_minus_base_ttc_absolute_error'):.9f}"
                ),
                "mean_visual_first_last_cosine_distance": (
                    f"{mean('visual_first_last_cosine_distance'):.9f}"
                ),
                "mean_corridor_approach_reduction": (
                    f"{mean('corridor_approach_reduction'):.9f}"
                ),
                "priority_score": str(priority),
                "representative_query_image_names": "|".join(
                    row["query_image_name"] for row in ranked[:3]
                ),
            }
        )
    output.sort(
        key=lambda row: (
            -int(row["priority_score"]),
            -int(row["triple_harm_queries"]),
            -int(row["mae_or_ssim_harm_queries"]),
            -float(row["mean_mae_delta"]),
            row["query_record_id"],
        )
    )
    return output[:maximum_records]


def safe_correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 2 or float(left.std()) <= 1e-12:
        return None
    if float(right.std()) <= 1e-12:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def load_feature_metadata(prefix: Path) -> dict:
    with Path(f"{prefix}.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    for key in (
        "query_true_probe_feature_used",
        "query_tactile_input",
        "future_visual_frames_used",
    ):
        if metadata.get(key) is not False:
            raise RuntimeError(
                f"Phase4I.5b feature cache violates online contract: {key}"
            )
    return metadata


def audit(config_path: str, section: str) -> dict:
    cfg = load_config(config_path)[section]
    samples = read_csv_rows(project_path(cfg["samples_csv"]))
    assert_development_only(samples, project_path(cfg["final_partition_csv"]))
    development_names = {
        row["image_name"]
        for row in samples
        if row["dataset_split"] == "train"
    }
    gate_rows = read_csv_rows(project_path(cfg["phase4i5a_query_csv"]))
    gate_by_name = unique_by_query(gate_rows, "Phase4I.5a")
    if set(gate_by_name) != development_names:
        raise RuntimeError("Phase4I.5b gate queries do not match development")
    phase4i3_rows = read_csv_rows(project_path(cfg["phase4i3_query_csv"]))
    phase4i3_by_name = unique_by_query(phase4i3_rows, "Phase4I.3")
    if set(phase4i3_by_name) != development_names:
        raise RuntimeError("Phase4I.5b retrieval queries do not match development")

    false_negative_names = [
        row["query_image_name"]
        for row in gate_rows
        if row["true_far_label"] == "1"
        and row["safety_predicted_far_risk"] == "0"
    ]
    expected = int(cfg["expected_false_negative_queries"])
    if len(false_negative_names) != expected:
        raise RuntimeError(
            "Phase4I.5b false-negative identity mismatch: "
            f"expected={expected}, actual={len(false_negative_names)}"
        )
    feature_prefix = project_path(cfg["feature_cache_prefix"])
    feature_metadata = load_feature_metadata(feature_prefix)
    visual, geometry, valid, padding = load_temporal_cache(
        feature_prefix,
        false_negative_names,
    )
    epsilon = float(cfg["metric_epsilon"])
    high_confidence_margin = float(cfg["high_confidence_miss_margin"])
    raw_rows = []
    for index, name in enumerate(false_negative_names):
        gate = gate_by_name[name]
        retrieval = phase4i3_by_name[name]
        probability = finite(gate, "temporal_far_risk_probability")
        threshold = finite(gate, "safety_far_risk_threshold")
        miss_margin = threshold - probability
        temporal_ttc = finite(gate, "predicted_ttc")
        true_ttc = finite(gate, "true_ttc")
        base_ttc = finite(retrieval, "predicted_ttc")
        temporal_ttc_error = abs(temporal_ttc - true_ttc)
        base_ttc_error = abs(base_ttc - true_ttc)
        diagnostics = temporal_visual_diagnostics(
            visual[index],
            geometry[index],
            valid[index],
            padding[index],
        )
        deltas = {
            metric: finite(retrieval, f"phase4i3_{metric}")
            - finite(retrieval, f"v1_{metric}")
            for metric in METRICS
        }
        outcome = retrieval_outcome(
            deltas["tactile_diff_mae"],
            deltas["tactile_ssim"],
            deltas["tactile_mask_iou"],
            epsilon,
        )
        changed = (
            retrieval["phase4i3_selected_cache_image_name"]
            != retrieval["v1_selected_cache_image_name"]
        )
        raw_rows.append(
            {
                "query_record_id": gate["query_record_id"],
                "query_image_name": name,
                "query_probe": gate["query_probe"],
                "oof_fold": gate["oof_fold"],
                "far_risk_probability": probability,
                "far_risk_threshold": threshold,
                "far_miss_margin": miss_margin,
                "temporal_predicted_ttc": temporal_ttc,
                "true_ttc": true_ttc,
                "temporal_ttc_absolute_error": temporal_ttc_error,
                "base_predicted_ttc": base_ttc,
                "base_ttc_absolute_error": base_ttc_error,
                "temporal_minus_base_ttc_absolute_error": (
                    temporal_ttc_error - base_ttc_error
                ),
                "ttc_entropy": finite(retrieval, "ttc_entropy"),
                "trajectory_stability": finite(
                    retrieval,
                    "trajectory_stability",
                ),
                "motion_speed": finite(retrieval, "motion_speed"),
                "motion_cumulative": finite(
                    retrieval,
                    "motion_cumulative",
                ),
                **diagnostics,
                "v1_selected_cache_image_name": retrieval[
                    "v1_selected_cache_image_name"
                ],
                "phase4i3_selected_cache_image_name": retrieval[
                    "phase4i3_selected_cache_image_name"
                ],
                "phase4i3_changed_cache": int(changed),
                "v1_oracle_rank": int(
                    retrieval["v1_ranker_oracle_embedding_rank"]
                ),
                "phase4i3_oracle_rank": int(
                    retrieval["phase4i3_ranker_oracle_embedding_rank"]
                ),
                "oracle_top1_delta": (
                    int(
                        retrieval[
                            "phase4i3_ranker_oracle_embedding_rank"
                        ]
                    )
                    == 1
                )
                - (
                    int(retrieval["v1_ranker_oracle_embedding_rank"]) == 1
                ),
                "v1_tactile_diff_mae": finite(
                    retrieval,
                    "v1_tactile_diff_mae",
                ),
                "phase4i3_tactile_diff_mae": finite(
                    retrieval,
                    "phase4i3_tactile_diff_mae",
                ),
                "mae_delta": deltas["tactile_diff_mae"],
                "v1_tactile_ssim": finite(
                    retrieval,
                    "v1_tactile_ssim",
                ),
                "phase4i3_tactile_ssim": finite(
                    retrieval,
                    "phase4i3_tactile_ssim",
                ),
                "ssim_delta": deltas["tactile_ssim"],
                "v1_tactile_mask_iou": finite(
                    retrieval,
                    "v1_tactile_mask_iou",
                ),
                "phase4i3_tactile_mask_iou": finite(
                    retrieval,
                    "phase4i3_tactile_mask_iou",
                ),
                "iou_delta": deltas["tactile_mask_iou"],
                "retrieval_outcome": outcome,
                "strict_triple_win": int(outcome == "strict_triple_win"),
                "spatial_only_gain": int(outcome == "spatial_only_gain"),
                "mae_or_ssim_harm": int(
                    outcome
                    in (
                        "spatial_only_gain",
                        "mae_or_ssim_harm",
                        "triple_harm",
                    )
                ),
                "triple_harm": int(outcome == "triple_harm"),
                "high_confidence_false_negative": int(
                    miss_margin >= high_confidence_margin
                ),
            }
        )

    visual_values = np.asarray(
        [
            row["visual_first_last_cosine_distance"]
            for row in raw_rows
        ],
        dtype=np.float64,
    )
    visual_labels, visual_edges = quartile_labels(visual_values)
    output_rows = []
    for index, raw in enumerate(raw_rows):
        raw["miss_margin_bin"] = numeric_bin(
            float(raw["far_miss_margin"]),
            float(cfg["miss_margin_first_boundary"]),
            float(cfg["miss_margin_second_boundary"]),
        )
        raw["temporal_ttc_error_bin"] = numeric_bin(
            float(raw["temporal_ttc_absolute_error"]),
            float(cfg["ttc_error_first_boundary"]),
            float(cfg["ttc_error_second_boundary"]),
        )
        raw["visual_change_quartile"] = f"q{int(visual_labels[index]) + 1}"
        raw["approach_direction"] = (
            "approaching"
            if float(raw["corridor_approach_reduction"]) > 0
            else "not_approaching"
        )
        output_rows.append(
            {
                key: (
                    f"{float(raw[key]):.9f}"
                    if isinstance(raw[key], (float, np.floating))
                    else str(int(raw[key]))
                    if isinstance(raw[key], (bool, int, np.integer))
                    else str(raw[key])
                )
                for key in QUERY_FIELDS
            }
        )
    slices = build_slices(output_rows)
    records = record_rows(
        output_rows,
        int(cfg["maximum_record_rows"]),
    )
    assert_csv_schema(output_rows, QUERY_FIELDS, "Phase4I.5b queries")
    assert_csv_schema(slices, SLICE_FIELDS, "Phase4I.5b slices")
    assert_csv_schema(records, RECORD_FIELDS, "Phase4I.5b records")
    write_csv_rows(
        project_path(cfg["query_output_csv"]),
        output_rows,
        QUERY_FIELDS,
    )
    write_csv_rows(
        project_path(cfg["slice_output_csv"]),
        slices,
        SLICE_FIELDS,
    )
    write_csv_rows(
        project_path(cfg["record_output_csv"]),
        records,
        RECORD_FIELDS,
    )

    changed = [
        row for row in output_rows if row["phase4i3_changed_cache"] == "1"
    ]
    harmful = [
        row for row in changed if row["mae_or_ssim_harm"] == "1"
    ]
    harmful_records = Counter(row["query_record_id"] for row in harmful)
    top_record_harm = sum(
        count
        for _, count in harmful_records.most_common(
            int(cfg["concentration_top_records"])
        )
    )
    correlations = {}
    mae = np.asarray(
        [float(row["mae_delta"]) for row in output_rows],
        dtype=np.float64,
    )
    for name, values in (
        (
            "far_miss_margin_vs_mae_delta",
            np.asarray(
                [float(row["far_miss_margin"]) for row in output_rows]
            ),
        ),
        (
            "temporal_ttc_error_vs_mae_delta",
            np.asarray(
                [
                    float(row["temporal_ttc_absolute_error"])
                    for row in output_rows
                ]
            ),
        ),
        (
            "visual_change_vs_mae_delta",
            np.asarray(
                [
                    float(row["visual_first_last_cosine_distance"])
                    for row in output_rows
                ]
            ),
        ),
        (
            "approach_reduction_vs_mae_delta",
            np.asarray(
                [
                    float(row["corridor_approach_reduction"])
                    for row in output_rows
                ]
            ),
        ),
    ):
        correlations[name] = safe_correlation(values, mae)

    outcome_counts = Counter(
        row["retrieval_outcome"] for row in output_rows
    )
    report = {
        "mode": "phase4i5b_temporal_far_false_negative_audit_v1",
        "false_negative_queries": len(output_rows),
        "false_negative_records": len(
            {row["query_record_id"] for row in output_rows}
        ),
        "probe_counts": dict(Counter(row["query_probe"] for row in output_rows)),
        "fold_counts": dict(Counter(row["oof_fold"] for row in output_rows)),
        "outcome_counts": dict(outcome_counts),
        "phase4i3_changed_cache_queries": len(changed),
        "phase4i3_changed_cache_rate": len(changed) / len(output_rows),
        "mae_or_ssim_harm_changed_queries": len(harmful),
        "mae_or_ssim_harm_rate_among_changed": (
            len(harmful) / max(len(changed), 1)
        ),
        "high_confidence_false_negatives": sum(
            int(row["high_confidence_false_negative"])
            for row in output_rows
        ),
        "visual_change_quartile_edges": visual_edges,
        "correlations": correlations,
        "harm_concentration": {
            "top_records": int(cfg["concentration_top_records"]),
            "harmful_queries_in_top_records": top_record_harm,
            "share": top_record_harm / max(len(harmful), 1),
        },
        "feature_cache_identity": feature_metadata[
            "query_identity_sha256"
        ],
        "outputs": {
            "query_csv": str(project_path(cfg["query_output_csv"])),
            "slice_csv": str(project_path(cfg["slice_output_csv"])),
            "record_csv": str(project_path(cfg["record_output_csv"])),
        },
        "integrity": {
            "source": "frozen strict Phase4I.5a OOF false negatives only",
            "model_retrained": False,
            "threshold_retuned": False,
            "query_tactile_input": False,
            "query_tactile_usage": "offline retrieval audit labels only",
            "future_visual_frames_used": False,
            "c2_contact_box": "unchanged",
            "top32_candidates": "unchanged Phase4I.3",
            "sealed_final_holdout_rows_read": 0,
            "development_validation_rows_read": 0,
        },
        "next_action": (
            "inspect the ranked record CSV; use structural abstention only "
            "if harm is concentrated and online-identifiable, otherwise "
            "collect record-disjoint temporal far examples"
        ),
    }
    write_json(project_path(cfg["metrics_json"]), report)
    print(report)
    for row in records[:20]:
        print(
            row["query_record_id"],
            "queries=",
            row["queries"],
            "harm=",
            row["mae_or_ssim_harm_queries"],
            "triple_harm=",
            row["triple_harm_queries"],
            "priority=",
            row["priority_score"],
            "representatives=",
            row["representative_query_image_names"],
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit Phase4I.5a temporal false-negative far queries."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--section",
        default="phase4i5b_temporal_far_false_negative_audit_v1",
    )
    args = parser.parse_args()
    audit(args.config, args.section)


if __name__ == "__main__":
    main()
