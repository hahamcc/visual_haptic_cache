"""Threshold-only safety recalibration for frozen Phase4I.5 OOF outputs."""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np

from .config import load_config, project_path
from .evaluate_phase4h_factorized_intensity_oof import (
    fast_bootstrap_comparison,
)
from .phase4h_dino_adaptation import assert_development_only
from .train_phase4h_dino_gate import metric_summary
from .train_phase4i4_far_risk_gate_oof import (
    THRESHOLD_FIELDS,
    assert_csv_schema,
    retrieval_rows,
    select_by_risk,
    threshold_csv_row,
)
from .train_phase4i5_temporal_progress_oof import (
    PROGRESS_NAMES,
    confusion_report,
)
from .train_phase4i_factorized_residual_cascade import METRICS
from .utils import read_csv_rows, write_csv_rows, write_json


QUERY_FIELDS = [
    "query_record_id",
    "query_image_name",
    "query_probe",
    "oof_fold",
    "temporal_far_risk_probability",
    "original_far_risk_threshold",
    "safety_far_risk_threshold",
    "original_predicted_far_risk",
    "safety_predicted_far_risk",
    "true_far_label",
    "threshold_changed",
    "risk_decision_changed",
    "predicted_progress_class",
    "true_progress_class",
    "predicted_ttc",
    "true_ttc",
    "v1_selected_cache_image_name",
    "phase4i3_selected_cache_image_name",
    "safety_gated_selected_cache_image_name",
    "safety_gated_selection_source",
    "safety_gated_ranker_oracle_embedding_rank",
    *[f"safety_gated_{metric}" for metric in METRICS],
    "deployment_accepted",
    "final_selection_source",
    "selected_cache_image_name",
    "ranker_oracle_embedding_rank",
    *METRICS,
]


def parsed_threshold_row(row: dict[str, str]) -> dict[str, float | int | str]:
    return {
        "held_out_fold": row["held_out_fold"],
        "threshold": float(row["threshold"]),
        "recall": float(row["recall"]),
        "false_positive_rate": float(row["false_positive_rate"]),
        "precision": float(row["precision"]),
        "risk_rate": float(row["risk_rate"]),
        "true_positives": int(row["true_positives"]),
        "false_positives": int(row["false_positives"]),
        "true_negatives": int(row["true_negatives"]),
        "false_negatives": int(row["false_negatives"]),
        "selected": int(row["selected"]),
    }


def select_safety_threshold(
    rows: list[dict[str, str]],
    target_recall: float,
) -> dict[str, float | int | str]:
    if not 0 < target_recall <= 1:
        raise ValueError("Phase4I.5a target recall must be inside (0, 1]")
    options = [parsed_threshold_row(row) for row in rows]
    folds = {str(row["held_out_fold"]) for row in options}
    if len(folds) != 1:
        raise RuntimeError("Phase4I.5a threshold options mix OOF folds")
    valid = [
        row for row in options if float(row["recall"]) >= target_recall
    ]
    if not valid:
        raise RuntimeError(
            f"Phase4I.5a cannot reach calibration recall {target_recall}"
        )
    return min(
        valid,
        key=lambda row: (
            float(row["false_positive_rate"]),
            -float(row["recall"]),
            -float(row["threshold"]),
        ),
    )


def keyed_retrieval_rows(
    rows: list[dict[str, str]],
    prefix: str,
    names: list[str],
) -> list[dict[str, str]]:
    keyed = {
        row["query_image_name"]: row for row in retrieval_rows(rows, prefix)
    }
    if len(keyed) != len(rows) or set(keyed) != set(names):
        raise RuntimeError(
            f"Phase4I.5a {prefix} retrieval rows do not match queries"
        )
    return [keyed[name] for name in names]


def classification_for_indices(
    source_rows: list[dict[str, str]],
    predicted_risk: np.ndarray,
    indices: np.ndarray,
) -> dict:
    class_lookup = {name: index for index, name in enumerate(PROGRESS_NAMES)}
    target = np.asarray(
        [
            class_lookup[source_rows[index]["true_progress_class"]]
            for index in indices
        ],
        dtype=np.int64,
    )
    predicted = np.asarray(
        [
            class_lookup[source_rows[index]["predicted_progress_class"]]
            for index in indices
        ],
        dtype=np.int64,
    )
    far_target = np.asarray(
        [int(source_rows[index]["true_far_label"]) for index in indices],
        dtype=np.float32,
    )
    report = confusion_report(
        target,
        predicted,
        far_target,
        predicted_risk[indices],
    )
    report["ttc_mae_frames"] = float(
        np.mean(
            [
                abs(
                    float(source_rows[index]["predicted_ttc"])
                    - float(source_rows[index]["true_ttc"])
                )
                for index in indices
            ]
        )
    )
    return report


def run(config_path: str, section: str) -> dict:
    cfg = load_config(config_path)[section]
    samples = read_csv_rows(project_path(cfg["samples_csv"]))
    assert_development_only(samples, project_path(cfg["final_partition_csv"]))
    development_names = {
        row["image_name"]
        for row in samples
        if row["dataset_split"] == "train"
    }
    source_rows = read_csv_rows(project_path(cfg["phase4i5_query_csv"]))
    names = [row["query_image_name"] for row in source_rows]
    if len(set(names)) != len(source_rows) or set(names) != development_names:
        raise RuntimeError("Phase4I.5a queries do not match development OOF")
    if any(row["deployment_accepted"] != "0" for row in source_rows):
        raise RuntimeError(
            "Phase4I.5a expects the frozen rejected Phase4I.5 OOF output"
        )
    probabilities = np.asarray(
        [
            float(row["temporal_far_risk_probability"])
            for row in source_rows
        ],
        dtype=np.float32,
    )
    original_thresholds = np.asarray(
        [float(row["far_risk_threshold"]) for row in source_rows],
        dtype=np.float32,
    )
    if (
        not np.isfinite(probabilities).all()
        or not np.isfinite(original_thresholds).all()
        or np.any(probabilities < 0)
        or np.any(probabilities > 1)
    ):
        raise RuntimeError("Phase4I.5a source probabilities are invalid")

    raw_grid = read_csv_rows(project_path(cfg["phase4i5_threshold_csv"]))
    grid_by_fold: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in raw_grid:
        grid_by_fold[row["held_out_fold"]].append(row)
    folds = sorted({row["oof_fold"] for row in source_rows})
    if len(folds) != 3 or set(grid_by_fold) != set(folds):
        raise RuntimeError("Phase4I.5a requires three matching OOF grids")
    target_recall = float(cfg["calibration_far_recall_target"])
    selected_by_fold = {
        fold: select_safety_threshold(grid_by_fold[fold], target_recall)
        for fold in folds
    }
    original_by_fold = {}
    for fold in folds:
        values = {
            float(source_rows[index]["far_risk_threshold"])
            for index in range(len(source_rows))
            if source_rows[index]["oof_fold"] == fold
        }
        if len(values) != 1:
            raise RuntimeError(
                f"Phase4I.5a fold {fold} has inconsistent source thresholds"
            )
        original_by_fold[fold] = next(iter(values))
    safety_thresholds = np.asarray(
        [
            float(selected_by_fold[row["oof_fold"]]["threshold"])
            for row in source_rows
        ],
        dtype=np.float32,
    )
    if np.any(safety_thresholds > original_thresholds + 1e-8):
        raise RuntimeError(
            "Phase4I.5a safety threshold is stricter than Phase4I.5"
        )
    original_risk = probabilities >= original_thresholds
    safety_risk = probabilities >= safety_thresholds
    if np.any(original_risk & ~safety_risk):
        raise RuntimeError("Phase4I.5a unexpectedly removed original risk flags")

    classification = classification_for_indices(
        source_rows,
        safety_risk,
        np.arange(len(source_rows), dtype=np.int32),
    )
    classification["by_oof_fold"] = {}
    for fold in folds:
        indices = np.asarray(
            [
                index
                for index, row in enumerate(source_rows)
                if row["oof_fold"] == fold
            ],
            dtype=np.int32,
        )
        classification["by_oof_fold"][fold] = classification_for_indices(
            source_rows,
            safety_risk,
            indices,
        )
    classification_contract_pass = bool(
        classification["far_recall"] >= float(cfg["minimum_far_recall"])
        and classification["near_mid_retention"]
        >= float(cfg["minimum_near_mid_retention"])
    )

    phase4i3_source = read_csv_rows(project_path(cfg["phase4i3_query_csv"]))
    v1_rows = keyed_retrieval_rows(phase4i3_source, "v1", names)
    phase4i3_rows = keyed_retrieval_rows(
        phase4i3_source,
        "phase4i3",
        names,
    )
    gated_rows = select_by_risk(v1_rows, phase4i3_rows, safety_risk)
    comparison = fast_bootstrap_comparison(
        v1_rows,
        gated_rows,
        {
            "bootstrap_iterations": int(cfg["bootstrap_iterations"]),
            "bootstrap_seed": int(cfg["bootstrap_seed"]),
        },
    )
    accepted = bool(classification_contract_pass and comparison["accepted"])
    final_rows = gated_rows if accepted else v1_rows

    query_output = []
    for index, (source, v1, current, gated, final) in enumerate(
        zip(
            source_rows,
            v1_rows,
            phase4i3_rows,
            gated_rows,
            final_rows,
            strict=True,
        )
    ):
        gate_source = "v1_far_risk" if safety_risk[index] else "phase4i3"
        query_output.append(
            {
                "query_record_id": source["query_record_id"],
                "query_image_name": source["query_image_name"],
                "query_probe": source["query_probe"],
                "oof_fold": source["oof_fold"],
                "temporal_far_risk_probability": (
                    f"{probabilities[index]:.9f}"
                ),
                "original_far_risk_threshold": (
                    f"{original_thresholds[index]:.9f}"
                ),
                "safety_far_risk_threshold": (
                    f"{safety_thresholds[index]:.9f}"
                ),
                "original_predicted_far_risk": str(
                    int(original_risk[index])
                ),
                "safety_predicted_far_risk": str(int(safety_risk[index])),
                "true_far_label": source["true_far_label"],
                "threshold_changed": str(
                    int(
                        not np.isclose(
                            original_thresholds[index],
                            safety_thresholds[index],
                        )
                    )
                ),
                "risk_decision_changed": str(
                    int(original_risk[index] != safety_risk[index])
                ),
                "predicted_progress_class": source[
                    "predicted_progress_class"
                ],
                "true_progress_class": source["true_progress_class"],
                "predicted_ttc": source["predicted_ttc"],
                "true_ttc": source["true_ttc"],
                "v1_selected_cache_image_name": v1[
                    "selected_cache_image_name"
                ],
                "phase4i3_selected_cache_image_name": current[
                    "selected_cache_image_name"
                ],
                "safety_gated_selected_cache_image_name": gated[
                    "selected_cache_image_name"
                ],
                "safety_gated_selection_source": gate_source,
                "safety_gated_ranker_oracle_embedding_rank": gated[
                    "ranker_oracle_embedding_rank"
                ],
                **{
                    f"safety_gated_{metric}": gated[metric]
                    for metric in METRICS
                },
                "deployment_accepted": str(int(accepted)),
                "final_selection_source": (
                    gate_source if accepted else "v1"
                ),
                "selected_cache_image_name": final[
                    "selected_cache_image_name"
                ],
                "ranker_oracle_embedding_rank": final[
                    "ranker_oracle_embedding_rank"
                ],
                **{metric: final[metric] for metric in METRICS},
            }
        )
    threshold_output = []
    for fold in folds:
        selected = selected_by_fold[fold]
        selected_count = 0
        for raw in grid_by_fold[fold]:
            option = parsed_threshold_row(raw)
            option["selected"] = bool(
                np.isclose(
                    float(option["threshold"]),
                    float(selected["threshold"]),
                    rtol=0.0,
                    atol=1e-12,
                )
            )
            selected_count += int(option["selected"])
            threshold_output.append(threshold_csv_row(option))
        if selected_count != 1:
            raise RuntimeError(
                f"Phase4I.5a fold {fold} selected {selected_count} thresholds"
            )
    assert_csv_schema(
        query_output,
        QUERY_FIELDS,
        "Phase4I.5a query output",
    )
    assert_csv_schema(
        threshold_output,
        THRESHOLD_FIELDS,
        "Phase4I.5a threshold output",
    )
    write_csv_rows(
        project_path(cfg["query_output_csv"]),
        query_output,
        QUERY_FIELDS,
    )
    write_csv_rows(
        project_path(cfg["threshold_output_csv"]),
        threshold_output,
        THRESHOLD_FIELDS,
    )

    changed = safety_risk != original_risk
    report = {
        "mode": "phase4i5a_temporal_far_safety_recalibration_oof_v1",
        "policy": {
            "model_retrained": False,
            "dino_recomputed": False,
            "target_selection": (
                "single predeclared safety target; no recall-target sweep"
            ),
            "calibration_far_recall_target": target_recall,
            "original_threshold_by_fold": original_by_fold,
            "selected_by_fold": selected_by_fold,
            "new_risk_flags": int(changed.sum()),
            "new_risk_flag_rate": float(changed.mean()),
        },
        "classification_contract": {
            "minimum_far_recall": float(cfg["minimum_far_recall"]),
            "minimum_near_mid_retention": float(
                cfg["minimum_near_mid_retention"]
            ),
            "oof": classification,
            "passed": classification_contract_pass,
        },
        "gated": {
            "summary": {
                "all": metric_summary(gated_rows),
                "far_probe75_100": metric_summary(
                    gated_rows,
                    lambda row: int(row["query_probe"]) >= 75,
                ),
            },
            "vs_v1": comparison,
        },
        "deployment_accepted": accepted,
        "deployed": {
            "source": "phase4i5a" if accepted else "v1",
            "summary": {
                "all": metric_summary(final_rows),
                "far_probe75_100": metric_summary(
                    final_rows,
                    lambda row: int(row["query_probe"]) >= 75,
                ),
            },
        },
        "integrity": {
            "source": "frozen strict Phase4I.5 OOF probabilities only",
            "model_retrained": False,
            "dino_recomputed": False,
            "query_true_probe_feature_used": False,
            "true_probe_offline_calibration_target_used": True,
            "query_tactile_input": False,
            "c2_contact_box": "unchanged",
            "top32_candidates": "unchanged Phase4I.3",
            "sealed_final_holdout_rows_read": 0,
            "development_validation_rows_read": 0,
        },
        "next_action": (
            "freeze Phase4I.5a and run independent development validation"
            if accepted
            else (
                "retain V1; stop threshold tuning and inspect/collect "
                "temporal far false negatives"
            )
        ),
    }
    write_json(project_path(cfg["metrics_json"]), report)
    print(
        {
            "mode": report["mode"],
            "policy": report["policy"],
            "classification_contract": report["classification_contract"],
            "gated": report["gated"],
            "deployment_accepted": report["deployment_accepted"],
            "deployed": report["deployed"],
            "integrity": report["integrity"],
            "next_action": report["next_action"],
        }
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recalibrate frozen Phase4I.5 far-risk thresholds."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--section",
        default="phase4i5a_far_safety_recalibration_oof_v1",
    )
    args = parser.parse_args()
    run(args.config, args.section)


if __name__ == "__main__":
    main()
