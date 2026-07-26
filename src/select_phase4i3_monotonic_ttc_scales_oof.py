"""Strict-OOF monotonic predicted-TTC scaling over the Phase4I.2 residual."""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import torch

from .config import load_config, project_path
from .evaluate_phase4h_factorized_intensity_oof import (
    fast_bootstrap_comparison,
)
from .phase4h_dino_adaptation import (
    assert_candidate_identity,
    assert_development_only,
    candidate_groups,
    candidate_set_fingerprint,
    record_hash_split,
)
from .train_phase4b_predicted_box_cache_ranker import set_seed
from .train_phase4h_dino_gate import metric_summary
from .train_phase4i2_ttc_robust_intensity_residual import (
    predicted_ttc_groups,
    predict_ranker,
    query_features,
    train_ranker,
    uncertainty_attenuation,
)
from .train_phase4i_factorized_residual_cascade import (
    METRICS,
    ONLINE_PROGRESS_FIELDS,
    build_cascade_query_rows,
    consistent_reference_rows,
    grouped,
    online_progress_features,
    query_standardize,
    rank_positions,
    required_float,
)
from .utils import ensure_dir, read_csv_rows, write_csv_rows, write_json


QUERY_FIELDS = [
    "query_record_id",
    "query_image_name",
    "query_probe",
    "oof_fold",
    "predicted_ttc_group",
    "selected_group_scale",
    "phase4i2_reference_effective_weight",
    "phase4i3_base_effective_weight",
    "phase4i3_effective_weight",
    "phase4i3_selected_cache_image_name",
    "phase4i3_ranker_oracle_embedding_rank",
    *[f"phase4i3_{metric}" for metric in METRICS],
    "v1_selected_cache_image_name",
    "v1_ranker_oracle_embedding_rank",
    *[f"v1_{metric}" for metric in METRICS],
    *ONLINE_PROGRESS_FIELDS,
    "deployment_accepted",
    "final_selection_source",
    "selected_cache_record_id",
    "selected_cache_image_name",
    "ranker_oracle_embedding_rank",
    *METRICS,
]
CANDIDATE_FIELDS = [
    "query_record_id",
    "query_image_name",
    "query_probe",
    "oof_fold",
    "predicted_ttc_group",
    "selected_group_scale",
    "phase4i2_reference_effective_weight",
    "phase4i3_base_effective_weight",
    "phase4i3_effective_weight",
    "candidate_rank",
    "phase4i3_candidate_rank",
    "final_candidate_rank",
    "v1_score",
    "predicted_intensity_distance",
    "phase4i3_score",
    "candidate_record_id",
    "candidate_image_name",
    "candidate_tactile_embedding_distance",
    "candidate_oracle_embedding_rank",
    "deployment_accepted",
    "final_selection_source",
]
SLICE_FIELDS = [
    "dimension",
    "slice",
    "queries",
    "mean_mae_delta",
    "mean_ssim_delta",
    "mean_iou_delta",
    "mean_top1_delta",
    "mae_worse_rate",
    "strict_triple_win_rate",
    "changed_cache_rate",
]
RECORD_FIELDS = [
    "query_record_id",
    "queries",
    "mean_mae_delta",
    "mean_ssim_delta",
    "mean_iou_delta",
    "mean_top1_delta",
    "mae_worse_rate",
    "strict_triple_win_rate",
    "changed_cache_rate",
]
SCALE_FIELDS = [
    "held_out_fold",
    "near_scale",
    "mid_scale",
    "far_scale",
    "safe",
    "objective",
    "all_tactile_diff_mae_delta",
    "all_tactile_ssim_delta",
    "all_tactile_mask_iou_delta",
    "all_oracle_top1_delta",
    "predicted_far_tactile_diff_mae_delta",
    "predicted_far_tactile_ssim_delta",
    "predicted_far_tactile_mask_iou_delta",
    "predicted_far_oracle_top1_delta",
    "selected",
]


def monotonic_scale_grid(values: list[float]) -> list[tuple[float, float, float]]:
    """Return (near, mid, far) scales satisfying far <= mid <= near."""
    unique = sorted({float(value) for value in values})
    if not unique or unique[0] < 0 or unique[-1] > 1:
        raise ValueError("Phase4I.3 scale grid must stay inside [0, 1]")
    return [
        scales
        for scales in itertools.product(unique, repeat=3)
        if scales[2] <= scales[1] <= scales[0]
    ]


def apply_group_scales(
    v1_scores: np.ndarray,
    intensity_scores: np.ndarray,
    effective_weights: np.ndarray,
    groups: np.ndarray,
    scales: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lengths = {
        len(v1_scores),
        len(intensity_scores),
        len(effective_weights),
        len(groups),
    }
    if len(lengths) != 1:
        raise ValueError("Phase4I.3 score inputs have inconsistent lengths")
    selected_scales = np.asarray(scales, dtype=np.float32)[groups]
    scaled_weights = effective_weights * selected_scales
    scores = v1_scores + scaled_weights[:, None] * intensity_scores
    return scores.astype(np.float32), scaled_weights, selected_scales


def delta_summary(
    v1_rows: list[dict[str, str]],
    current_rows: list[dict[str, str]],
) -> dict[str, float]:
    if len(v1_rows) != len(current_rows) or not v1_rows:
        raise ValueError("Phase4I.3 delta summary requires paired nonempty rows")
    v1 = {row["query_image_name"]: row for row in v1_rows}
    current = {row["query_image_name"]: row for row in current_rows}
    if set(v1) != set(current):
        raise RuntimeError("Phase4I.3 comparison query sets differ")
    names = list(v1)
    deltas = {
        metric: np.asarray(
            [
                required_float(current[name], metric)
                - required_float(v1[name], metric)
                for name in names
            ],
            dtype=np.float64,
        )
        for metric in METRICS
    }
    top1_delta = np.asarray(
        [
            float(
                int(current[name]["ranker_oracle_embedding_rank"]) == 1
            )
            - float(int(v1[name]["ranker_oracle_embedding_rank"]) == 1)
            for name in names
        ],
        dtype=np.float64,
    )
    return {
        "queries": float(len(names)),
        "tactile_diff_mae": float(deltas["tactile_diff_mae"].mean()),
        "tactile_ssim": float(deltas["tactile_ssim"].mean()),
        "tactile_mask_iou": float(deltas["tactile_mask_iou"].mean()),
        "oracle_top1": float(top1_delta.mean()),
    }


def scale_is_safe(
    all_delta: dict[str, float],
    predicted_far_delta: dict[str, float],
    epsilon: float,
) -> bool:
    for values in (all_delta, predicted_far_delta):
        if (
            values["tactile_diff_mae"] > epsilon
            or values["tactile_ssim"] < -epsilon
            or values["tactile_mask_iou"] < -epsilon
            or values["oracle_top1"] < -epsilon
        ):
            return False
    return True


def choose_scale(
    candidates: list[dict],
) -> dict:
    safe = [row for row in candidates if row["safe"]]
    if not safe:
        raise RuntimeError("Phase4I.3 grid must contain the V1 identity scale")
    return min(
        safe,
        key=lambda row: (
            float(row["objective"]),
            float(row["far_scale"]),
            float(row["mid_scale"]),
            float(row["near_scale"]),
        ),
    )


def diagnostic_row(
    rows: list[dict[str, str]],
    dimension: str,
    name: str,
) -> dict[str, str]:
    mae = np.asarray(
        [
            required_float(row, "robust_tactile_diff_mae")
            - required_float(row, "v1_tactile_diff_mae")
            for row in rows
        ],
        dtype=np.float64,
    )
    ssim = np.asarray(
        [
            required_float(row, "robust_tactile_ssim")
            - required_float(row, "v1_tactile_ssim")
            for row in rows
        ],
        dtype=np.float64,
    )
    iou = np.asarray(
        [
            required_float(row, "robust_tactile_mask_iou")
            - required_float(row, "v1_tactile_mask_iou")
            for row in rows
        ],
        dtype=np.float64,
    )
    top1 = np.asarray(
        [
            float(int(row["robust_ranker_oracle_embedding_rank"]) == 1)
            - float(int(row["v1_ranker_oracle_embedding_rank"]) == 1)
            for row in rows
        ],
        dtype=np.float64,
    )
    triple = (mae < 0) & (ssim >= 0) & (iou >= 0)
    changed = np.asarray(
        [
            row["robust_selected_cache_image_name"]
            != row["v1_selected_cache_image_name"]
            for row in rows
        ],
        dtype=bool,
    )
    return {
        "dimension": dimension,
        "slice": name,
        "queries": str(len(rows)),
        "mean_mae_delta": f"{mae.mean():.9f}",
        "mean_ssim_delta": f"{ssim.mean():.9f}",
        "mean_iou_delta": f"{iou.mean():.9f}",
        "mean_top1_delta": f"{top1.mean():.9f}",
        "mae_worse_rate": f"{(mae > 0).mean():.9f}",
        "strict_triple_win_rate": f"{triple.mean():.9f}",
        "changed_cache_rate": f"{changed.mean():.9f}",
    }


def phase4i2_diagnostics(
    rows: list[dict[str, str]],
    thresholds: tuple[float, float],
    maximum_records: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict]:
    if not rows:
        raise RuntimeError("Phase4I.3 requires Phase4I.2 query rows")
    predicted_ttc = np.asarray(
        [required_float(row, "predicted_ttc") for row in rows],
        dtype=np.float32,
    )
    attenuation = np.asarray(
        [required_float(row, "robust_attenuation") for row in rows],
        dtype=np.float32,
    )
    entropy = np.asarray(
        [required_float(row, "ttc_entropy") for row in rows],
        dtype=np.float32,
    )
    stability = np.asarray(
        [required_float(row, "trajectory_stability") for row in rows],
        dtype=np.float32,
    )
    groups = np.digitize(predicted_ttc, thresholds)
    slices: list[dict[str, str]] = [diagnostic_row(rows, "all", "all")]

    def add_categories(
        dimension: str,
        labels: np.ndarray,
        names: list[str] | None = None,
    ) -> None:
        for value in sorted(np.unique(labels).tolist()):
            subset = [
                row for row, label in zip(rows, labels, strict=True)
                if label == value
            ]
            label = names[int(value)] if names is not None else str(value)
            slices.append(diagnostic_row(subset, dimension, label))

    add_categories(
        "predicted_ttc_group",
        groups,
        ["near_lt30", "mid_30_60", "far_ge60"],
    )
    add_categories(
        "oof_fold",
        np.asarray([row["oof_fold"] for row in rows]),
    )
    for dimension, values in (
        ("attenuation_quartile", attenuation),
        ("ttc_entropy_quartile", entropy),
        ("trajectory_stability_quartile", stability),
    ):
        edges = np.unique(np.quantile(values, [0.25, 0.5, 0.75]))
        add_categories(
            dimension,
            np.digitize(values, edges),
        )

    by_record: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_record.setdefault(row["query_record_id"], []).append(row)
    records = [
        {
            "query_record_id": record_id,
            **{
                key: value
                for key, value in diagnostic_row(
                    record_rows,
                    "record",
                    record_id,
                ).items()
                if key not in ("dimension", "slice")
            },
        }
        for record_id, record_rows in by_record.items()
    ]
    records.sort(
        key=lambda row: (
            -float(row["mean_mae_delta"]),
            row["query_record_id"],
        )
    )
    records = records[:maximum_records]
    report = {
        "query_count": len(rows),
        "predicted_ttc_thresholds": list(thresholds),
        "attenuation_range": [
            float(attenuation.min()),
            float(attenuation.max()),
        ],
        "slice_count": len(slices),
        "record_count": len(by_record),
        "worst_record_count_written": len(records),
    }
    return slices, records, report


def metric_deltas(
    v1_rows: list[dict[str, str]],
    current_rows: list[dict[str, str]],
    selected: np.ndarray,
) -> tuple[dict[str, float], dict[str, float]]:
    all_delta = delta_summary(v1_rows, current_rows)
    far_v1 = [row for row, keep in zip(v1_rows, selected, strict=True) if keep]
    far_current = [
        row for row, keep in zip(current_rows, selected, strict=True) if keep
    ]
    if not far_v1:
        raise RuntimeError("Phase4I.3 inner validation has no predicted-far rows")
    return all_delta, delta_summary(far_v1, far_current)


def evaluate_scale_grid(
    fold: str,
    validation: np.ndarray,
    v1_reference: list[dict[str, str]],
    groups_by_name: dict[str, list[dict[str, str]]],
    arrays: dict[str, np.ndarray],
    predicted_groups: np.ndarray,
    effective_weights: np.ndarray,
    source_by_name: dict[str, dict[str, str]],
    cfg: dict,
    touch_cache: dict[str, np.ndarray],
) -> tuple[dict, list[dict]]:
    grid = monotonic_scale_grid(cfg["scale_grid"])
    local_v1 = [v1_reference[index] for index in validation]
    local_groups = predicted_groups[validation]
    predicted_far = local_groups == 2
    reports = []
    for near, mid, far in grid:
        scores, scaled_weights, _ = apply_group_scales(
            arrays["v1"][validation],
            arrays["intensity"][validation],
            effective_weights,
            local_groups,
            (near, mid, far),
        )
        rows = build_cascade_query_rows(
            local_v1,
            groups_by_name,
            scores.argmin(axis=1),
            scores,
            np.stack(
                (
                    np.zeros(len(validation), dtype=np.float32),
                    scaled_weights,
                ),
                axis=1,
            ),
            source_by_name,
            cfg,
            touch_cache,
            f"phase4i3_fold{fold}_scale_{near}_{mid}_{far}",
        )
        all_delta, far_delta = metric_deltas(
            local_v1,
            rows,
            predicted_far,
        )
        safe = scale_is_safe(
            all_delta,
            far_delta,
            float(cfg["selection_guard_epsilon"]),
        )
        reports.append(
            {
                "held_out_fold": fold,
                "near_scale": near,
                "mid_scale": mid,
                "far_scale": far,
                "safe": safe,
                "objective": (
                    all_delta["tactile_diff_mae"]
                    + far_delta["tactile_diff_mae"]
                ),
                "all_delta": all_delta,
                "predicted_far_delta": far_delta,
            }
        )
    selected = choose_scale(reports)
    for row in reports:
        row["selected"] = row is selected
    return selected, reports


def scale_csv_row(row: dict) -> dict[str, str]:
    return {
        "held_out_fold": row["held_out_fold"],
        "near_scale": f"{row['near_scale']:.6f}",
        "mid_scale": f"{row['mid_scale']:.6f}",
        "far_scale": f"{row['far_scale']:.6f}",
        "safe": str(int(row["safe"])),
        "objective": f"{row['objective']:.9f}",
        **{
            f"all_{metric}_delta": f"{row['all_delta'][metric]:.9f}"
            for metric in (
                "tactile_diff_mae",
                "tactile_ssim",
                "tactile_mask_iou",
                "oracle_top1",
            )
        },
        **{
            f"predicted_far_{metric}_delta": (
                f"{row['predicted_far_delta'][metric]:.9f}"
            )
            for metric in (
                "tactile_diff_mae",
                "tactile_ssim",
                "tactile_mask_iou",
                "oracle_top1",
            )
        },
        "selected": str(int(row["selected"])),
    }


def run(config_path: str, section: str) -> dict:
    cfg = load_config(config_path)[section]
    set_seed(int(cfg["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples = read_csv_rows(project_path(cfg["samples_csv"]))
    assert_development_only(samples, project_path(cfg["final_partition_csv"]))
    source_by_name = {
        row["image_name"]: row
        for row in samples
        if row["dataset_split"] == "train"
    }
    query_rows = read_csv_rows(project_path(cfg["v1_query_csv"]))
    query_names = [row["query_image_name"] for row in query_rows]
    query_by_name = {row["query_image_name"]: row for row in query_rows}
    if len(query_by_name) != len(query_rows) or set(query_names) != set(
        source_by_name
    ):
        raise RuntimeError("Phase4I.3 V1 query rows do not match development")
    phase4i2_rows = read_csv_rows(project_path(cfg["phase4i2_query_csv"]))
    phase4i2_by_name = {
        row["query_image_name"]: row for row in phase4i2_rows
    }
    if set(phase4i2_by_name) != set(query_names):
        raise RuntimeError("Phase4I.3 query set differs from Phase4I.2")

    thresholds = tuple(float(value) for value in cfg["predicted_ttc_groups"])
    slice_rows, record_rows, diagnostic_report = phase4i2_diagnostics(
        phase4i2_rows,
        thresholds,
        int(cfg["maximum_diagnostic_records"]),
    )
    write_csv_rows(
        project_path(cfg["diagnostic_slice_csv"]),
        slice_rows,
        SLICE_FIELDS,
    )
    write_csv_rows(
        project_path(cfg["diagnostic_record_csv"]),
        record_rows,
        RECORD_FIELDS,
    )

    top_k = int(cfg["geometry_filter_k"])
    v1_groups = candidate_groups(
        read_csv_rows(project_path(cfg["v1_candidate_csv"])),
        top_k,
    )
    factor_groups_raw = candidate_groups(
        read_csv_rows(project_path(cfg["factor_candidate_csv"])),
        top_k,
    )
    assert_candidate_identity(v1_groups, factor_groups_raw)
    if set(factor_groups_raw) != set(query_names):
        raise RuntimeError("Phase4I.3 candidate queries differ from V1")
    candidate_fingerprint = candidate_set_fingerprint(factor_groups_raw)
    groups_by_name = grouped(
        [row for group in factor_groups_raw.values() for row in group]
    )
    for name in query_names:
        if any(
            row["candidate_record_id"] == row["query_record_id"]
            for row in groups_by_name[name]
        ):
            raise RuntimeError(f"Phase4I.3 same-record candidate for {name}")
        if any(
            row["oof_fold"] != query_by_name[name]["oof_fold"]
            for row in groups_by_name[name]
        ):
            raise RuntimeError(f"Phase4I.3 fold mismatch for {name}")

    def matrix(field: str) -> np.ndarray:
        return np.asarray(
            [
                [
                    required_float(row, field)
                    for row in groups_by_name[name]
                ]
                for name in query_names
            ],
            dtype=np.float32,
        )

    arrays = {
        "v1": query_standardize(matrix("v1_score")),
        "intensity": query_standardize(
            matrix("predicted_intensity_distance_dino_motion")
        ),
        "target": matrix("candidate_tactile_embedding_distance"),
    }
    progress = online_progress_features(groups_by_name, query_names)
    raw_features = query_features(
        arrays["v1"],
        arrays["intensity"],
        progress,
    )
    attenuation = uncertainty_attenuation(progress, cfg)
    predicted_groups = predicted_ttc_groups(progress, thresholds)
    v1_reference = consistent_reference_rows(
        query_rows,
        groups_by_name,
        arrays["v1"],
    )
    records = np.asarray([row["query_record_id"] for row in query_rows])
    folds = sorted({row["oof_fold"] for row in query_rows})
    if len(folds) != 3:
        raise RuntimeError(f"Phase4I.3 requires three OOF folds, got {folds}")

    output_scores = np.zeros((len(query_rows), top_k), dtype=np.float32)
    output_base_weights = np.zeros(len(query_rows), dtype=np.float32)
    output_weights = np.zeros(len(query_rows), dtype=np.float32)
    output_scales = np.zeros(len(query_rows), dtype=np.float32)
    selected_by_fold = {}
    all_scale_reports = []
    training_reports = []
    touch_cache: dict[str, np.ndarray] = {}
    checkpoint_root = project_path(cfg["checkpoint_dir"])
    ensure_dir(checkpoint_root)
    for fold in folds:
        held_out = np.asarray(
            [
                index
                for index, row in enumerate(query_rows)
                if row["oof_fold"] == fold
            ],
            dtype=np.int32,
        )
        outer_fit = np.asarray(
            [
                index
                for index, row in enumerate(query_rows)
                if row["oof_fold"] != fold
            ],
            dtype=np.int32,
        )
        scale_calibration = np.asarray(
            [
                index
                for index in outer_fit
                if record_hash_split(
                    records[index],
                    float(cfg["scale_calibration_fraction"]),
                    int(cfg["scale_calibration_seed"]) + int(fold),
                )
            ],
            dtype=np.int32,
        )
        calibration_set = set(scale_calibration.tolist())
        fit_and_early_stop = np.asarray(
            [
                index
                for index in outer_fit
                if index not in calibration_set
            ],
            dtype=np.int32,
        )
        early_stop = np.asarray(
            [
                index
                for index in fit_and_early_stop
                if record_hash_split(
                    records[index],
                    float(cfg["inner_validation_fraction"]),
                    int(cfg["inner_split_seed"]) + int(fold),
                )
            ],
            dtype=np.int32,
        )
        early_stop_set = set(early_stop.tolist())
        fit = np.asarray(
            [
                index
                for index in fit_and_early_stop
                if index not in early_stop_set
            ],
            dtype=np.int32,
        )
        if not len(fit) or not len(early_stop) or not len(scale_calibration):
            raise RuntimeError(
                f"Phase4I.3 fold {fold} nested split contains an empty set"
            )
        split_records = [
            set(records[indices].tolist())
            for indices in (fit, early_stop, scale_calibration, held_out)
        ]
        if any(
            split_records[left] & split_records[right]
            for left in range(len(split_records))
            for right in range(left + 1, len(split_records))
        ):
            raise RuntimeError(
                f"Phase4I.3 fold {fold} nested record split overlaps"
            )
        calibration_effective_values = []
        held_effective_values = []
        for seed in [int(value) for value in cfg["seeds"]]:
            print(
                f"phase4i3 fold {fold} seed {seed}: training nested residual",
                flush=True,
            )
            set_seed(seed)
            checkpoint_path = (
                checkpoint_root / f"fold_{fold}_seed_{seed}.pt"
            )
            model, training_report = train_ranker(
                arrays,
                raw_features,
                progress,
                attenuation,
                fit,
                early_stop,
                records,
                True,
                cfg,
                device,
                checkpoint_path,
                {
                    "scope": (
                        "strict_oof_three_way_nested_ttc_robust_residual"
                    ),
                    "fold": fold,
                    "seed": seed,
                    "candidate_fingerprint": candidate_fingerprint,
                    "scale_calibration_records_seen": False,
                    "held_out_records_seen": False,
                    "query_true_probe_used": False,
                    "query_tactile_input": False,
                },
            )
            _, _, calibration_effective = predict_ranker(
                model,
                arrays,
                raw_features,
                attenuation,
                scale_calibration,
                device,
                checkpoint_path,
            )
            _, _, held_effective = predict_ranker(
                model,
                arrays,
                raw_features,
                attenuation,
                held_out,
                device,
                checkpoint_path,
            )
            calibration_effective_values.append(calibration_effective)
            held_effective_values.append(held_effective)
            training_reports.append(
                {
                    "fold": fold,
                    "seed": seed,
                    "fit_queries": len(fit),
                    "early_stop_queries": len(early_stop),
                    "scale_calibration_queries": len(scale_calibration),
                    "held_out_queries": len(held_out),
                    **training_report,
                }
            )
        calibration_effective = np.mean(
            calibration_effective_values,
            axis=0,
        ).astype(np.float32)
        selected, scale_reports = evaluate_scale_grid(
            fold,
            scale_calibration,
            v1_reference,
            groups_by_name,
            arrays,
            predicted_groups,
            calibration_effective,
            source_by_name,
            cfg,
            touch_cache,
        )
        selected_by_fold[fold] = {
            key: selected[key]
            for key in (
                "near_scale",
                "mid_scale",
                "far_scale",
                "objective",
            )
        }
        all_scale_reports.extend(scale_reports)
        held_effective = np.mean(
            held_effective_values,
            axis=0,
        ).astype(np.float32)
        scales = (
            float(selected["near_scale"]),
            float(selected["mid_scale"]),
            float(selected["far_scale"]),
        )
        scores, weights, selected_scales = apply_group_scales(
            arrays["v1"][held_out],
            arrays["intensity"][held_out],
            held_effective,
            predicted_groups[held_out],
            scales,
        )
        output_scores[held_out] = scores
        output_base_weights[held_out] = held_effective
        output_weights[held_out] = weights
        output_scales[held_out] = selected_scales
        print(
            f"phase4i3 fold {fold}: selected scales "
            f"near/mid/far={scales}",
            flush=True,
        )

    write_csv_rows(
        project_path(cfg["scale_grid_csv"]),
        [scale_csv_row(row) for row in all_scale_reports],
        SCALE_FIELDS,
    )
    phase4i3_rows = build_cascade_query_rows(
        v1_reference,
        groups_by_name,
        output_scores.argmin(axis=1),
        output_scores,
        np.stack(
            (
                np.zeros(len(query_rows), dtype=np.float32),
                output_weights,
            ),
            axis=1,
        ),
        source_by_name,
        cfg,
        touch_cache,
        "phase4i3_monotonic_ttc_scale",
    )
    comparison = fast_bootstrap_comparison(
        v1_reference,
        phase4i3_rows,
        {
            "bootstrap_iterations": int(cfg["bootstrap_iterations"]),
            "bootstrap_seed": int(cfg["bootstrap_seed"]),
        },
    )
    accepted = bool(comparison["accepted"])
    final_rows = phase4i3_rows if accepted else v1_reference

    query_output = []
    for index, (v1, current, final) in enumerate(
        zip(v1_reference, phase4i3_rows, final_rows, strict=True)
    ):
        phase4i2 = phase4i2_by_name[v1["query_image_name"]]
        query_output.append(
            {
                "query_record_id": v1["query_record_id"],
                "query_image_name": v1["query_image_name"],
                "query_probe": v1["query_probe"],
                "oof_fold": v1["oof_fold"],
                "predicted_ttc_group": str(int(predicted_groups[index])),
                "selected_group_scale": f"{output_scales[index]:.6f}",
                "phase4i2_reference_effective_weight": phase4i2[
                    "robust_effective_weight"
                ],
                "phase4i3_base_effective_weight": (
                    f"{output_base_weights[index]:.9f}"
                ),
                "phase4i3_effective_weight": f"{output_weights[index]:.9f}",
                "phase4i3_selected_cache_image_name": current[
                    "selected_cache_image_name"
                ],
                "phase4i3_ranker_oracle_embedding_rank": current[
                    "ranker_oracle_embedding_rank"
                ],
                **{
                    f"phase4i3_{metric}": current[metric]
                    for metric in METRICS
                },
                "v1_selected_cache_image_name": v1[
                    "selected_cache_image_name"
                ],
                "v1_ranker_oracle_embedding_rank": v1[
                    "ranker_oracle_embedding_rank"
                ],
                **{f"v1_{metric}": v1[metric] for metric in METRICS},
                **{
                    field: f"{progress[index, field_index]:.9f}"
                    for field_index, field in enumerate(
                        ONLINE_PROGRESS_FIELDS
                    )
                },
                "deployment_accepted": str(int(accepted)),
                "final_selection_source": "phase4i3" if accepted else "v1",
                "selected_cache_record_id": final[
                    "selected_cache_record_id"
                ],
                "selected_cache_image_name": final[
                    "selected_cache_image_name"
                ],
                "ranker_oracle_embedding_rank": final[
                    "ranker_oracle_embedding_rank"
                ],
                **{metric: final[metric] for metric in METRICS},
            }
        )

    candidate_output = []
    for query_index, query in enumerate(v1_reference):
        phase4i3_rank = rank_positions(output_scores[query_index])
        for candidate_index, row in enumerate(
            groups_by_name[query["query_image_name"]]
        ):
            final_rank = (
                int(phase4i3_rank[candidate_index])
                if accepted
                else int(row["candidate_rank"])
            )
            candidate_output.append(
                {
                    "query_record_id": query["query_record_id"],
                    "query_image_name": query["query_image_name"],
                    "query_probe": query["query_probe"],
                    "oof_fold": query["oof_fold"],
                    "predicted_ttc_group": str(
                        int(predicted_groups[query_index])
                    ),
                    "selected_group_scale": (
                        f"{output_scales[query_index]:.6f}"
                    ),
                    "phase4i2_reference_effective_weight": phase4i2_by_name[
                        query["query_image_name"]
                    ]["robust_effective_weight"],
                    "phase4i3_base_effective_weight": (
                        f"{output_base_weights[query_index]:.9f}"
                    ),
                    "phase4i3_effective_weight": (
                        f"{output_weights[query_index]:.9f}"
                    ),
                    "candidate_rank": row["candidate_rank"],
                    "phase4i3_candidate_rank": str(
                        int(phase4i3_rank[candidate_index])
                    ),
                    "final_candidate_rank": str(final_rank),
                    "v1_score": row["v1_score"],
                    "predicted_intensity_distance": row[
                        "predicted_intensity_distance_dino_motion"
                    ],
                    "phase4i3_score": (
                        f"{output_scores[query_index, candidate_index]:.9f}"
                    ),
                    "candidate_record_id": row["candidate_record_id"],
                    "candidate_image_name": row["candidate_image_name"],
                    "candidate_tactile_embedding_distance": row[
                        "candidate_tactile_embedding_distance"
                    ],
                    "candidate_oracle_embedding_rank": row[
                        "candidate_oracle_embedding_rank"
                    ],
                    "deployment_accepted": str(int(accepted)),
                    "final_selection_source": (
                        "phase4i3" if accepted else "v1"
                    ),
                }
            )
    write_csv_rows(
        project_path(cfg["query_output_csv"]),
        query_output,
        QUERY_FIELDS,
    )
    write_csv_rows(
        project_path(cfg["candidate_output_csv"]),
        candidate_output,
        CANDIDATE_FIELDS,
    )

    report = {
        "mode": "phase4i3_monotonic_predicted_ttc_scale_oof_v1",
        "device": str(device),
        "diagnostic": diagnostic_report,
        "scale_grid": [list(row) for row in monotonic_scale_grid(cfg["scale_grid"])],
        "selected_scales_by_fold": selected_by_fold,
        "training": training_reports,
        "phase4i3": {
            "summary": {
                "all": metric_summary(phase4i3_rows),
                "far_probe75_100": metric_summary(
                    phase4i3_rows,
                    lambda row: int(row["query_probe"]) >= 75,
                ),
            },
            "vs_v1": comparison,
        },
        "deployment_accepted": accepted,
        "deployed": {
            "source": "phase4i3" if accepted else "v1",
            "summary": {
                "all": metric_summary(final_rows),
                "far_probe75_100": metric_summary(
                    final_rows,
                    lambda row: int(row["query_probe"]) >= 75,
                ),
            },
        },
        "selection_contract": {
            "groups": "predicted TTC <30, 30-60, >=60",
            "constraint": "far_scale <= mid_scale <= near_scale",
            "grid": [float(value) for value in cfg["scale_grid"]],
            "selection_rows": (
                "outer-fit record-disjoint scale-calibration split only"
            ),
            "guards": (
                "overall and predicted-far MAE/SSIM/IoU/Top1 non-degradation"
            ),
            "objective": (
                "minimize overall plus predicted-far tactile MAE delta"
            ),
            "identity_fallback": [0.0, 0.0, 0.0],
        },
        "integrity": {
            "source": "strict development 3-fold record-level OOF only",
            "c2_contact_box": "unchanged",
            "top32_candidates": "frozen V1 identity",
            "candidate_fingerprint": candidate_fingerprint,
            "same_record_candidates": 0,
            "direct_dino_residual_used": False,
            "dino_used_inside_frozen_intensity_predictor": True,
            "query_true_probe_feature_used": False,
            "true_probe_model_selection_used": False,
            "query_tactile_input": False,
            "query_tactile_usage": (
                "offline record-disjoint scale calibration and evaluation only"
            ),
            "nested_split": (
                "fit / early-stop / scale-calibration / outer-held-out "
                "records are mutually exclusive"
            ),
            "sealed_final_holdout_rows_read": 0,
            "development_validation_rows_read": 0,
            "sample_level_gate_used": False,
        },
        "next_action": (
            "freeze Phase4I.3 and implement independent development validation"
            if accepted
            else "retain V1; inspect Phase4I.3 slice/record diagnostics and do not run final holdout"
        ),
    }
    write_json(project_path(cfg["metrics_json"]), report)
    print(
        {
            "mode": report["mode"],
            "selected_scales_by_fold": report["selected_scales_by_fold"],
            "phase4i3": report["phase4i3"],
            "deployment_accepted": report["deployment_accepted"],
            "deployed": report["deployed"],
            "integrity": report["integrity"],
            "next_action": report["next_action"],
        }
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select strict-OOF monotonic predicted-TTC residual scales."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--section",
        default="phase4i3_monotonic_ttc_scale_oof_v1",
    )
    args = parser.parse_args()
    run(args.config, args.section)


if __name__ == "__main__":
    main()
