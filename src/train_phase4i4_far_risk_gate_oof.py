"""Strict-OOF far-horizon risk calibration for Phase4I.3 retrieval."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .config import load_config, project_path
from .evaluate_phase4h_factorized_intensity_oof import (
    fast_bootstrap_comparison,
)
from .phase4h_dino_adaptation import (
    assert_development_only,
    record_hash_split,
)
from .train_phase4b_predicted_box_cache_ranker import set_seed
from .train_phase4h_dino_gate import metric_summary
from .train_phase4i_factorized_residual_cascade import (
    METRICS,
    ONLINE_PROGRESS_FIELDS,
    required_float,
)
from .utils import ensure_dir, read_csv_rows, write_csv_rows, write_json


DERIVED_FEATURES = (
    "predicted_ttc_lt30",
    "predicted_ttc_lt60",
    "predicted_ttc_x_entropy",
)
QUERY_FIELDS = [
    "query_record_id",
    "query_image_name",
    "query_probe",
    "oof_fold",
    "far_risk_probability",
    "far_risk_threshold",
    "predicted_far_risk",
    "true_far_label",
    "phase4i3_selected_cache_image_name",
    "v1_selected_cache_image_name",
    "gated_selected_cache_image_name",
    "gated_selection_source",
    "gated_ranker_oracle_embedding_rank",
    *[f"gated_{metric}" for metric in METRICS],
    *ONLINE_PROGRESS_FIELDS,
    "deployment_accepted",
    "final_selection_source",
    "selected_cache_image_name",
    "ranker_oracle_embedding_rank",
    *METRICS,
]
THRESHOLD_FIELDS = [
    "held_out_fold",
    "threshold",
    "recall",
    "false_positive_rate",
    "precision",
    "risk_rate",
    "true_positives",
    "false_positives",
    "true_negatives",
    "false_negatives",
    "selected",
]


def feature_names() -> list[str]:
    return [*ONLINE_PROGRESS_FIELDS, *DERIVED_FEATURES]


def online_features(rows: list[dict[str, str]]) -> np.ndarray:
    base = np.asarray(
        [
            [required_float(row, field) for field in ONLINE_PROGRESS_FIELDS]
            for row in rows
        ],
        dtype=np.float32,
    )
    ttc = base[:, ONLINE_PROGRESS_FIELDS.index("predicted_ttc")]
    entropy = base[:, ONLINE_PROGRESS_FIELDS.index("ttc_entropy")]
    derived = np.stack(
        (
            (ttc < 30.0).astype(np.float32),
            (ttc < 60.0).astype(np.float32),
            (ttc / 100.0) * entropy,
        ),
        axis=1,
    )
    values = np.concatenate((base, derived), axis=1).astype(np.float32)
    if not np.isfinite(values).all():
        raise RuntimeError("Phase4I.4 online features contain nonfinite values")
    return values


def far_labels(rows: list[dict[str, str]], far_probe_minimum: int) -> np.ndarray:
    return np.asarray(
        [int(row["query_probe"]) >= far_probe_minimum for row in rows],
        dtype=np.float32,
    )


def record_balanced_weights(
    indices: np.ndarray,
    records: np.ndarray,
) -> np.ndarray:
    unique, inverse = np.unique(records[indices], return_inverse=True)
    counts = np.bincount(inverse, minlength=len(unique)).astype(np.float32)
    values = 1.0 / counts[inverse]
    return (values / max(float(values.mean()), 1e-8)).astype(np.float32)


class FarRiskClassifier(nn.Module):
    """A low-capacity linear far-horizon risk calibrator."""

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(feature_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features).squeeze(1)


def weighted_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    sample_weights: torch.Tensor,
    positive_weight: torch.Tensor,
) -> torch.Tensor:
    losses = nn.functional.binary_cross_entropy_with_logits(
        logits,
        targets,
        pos_weight=positive_weight,
        reduction="none",
    )
    return (losses * sample_weights).sum() / sample_weights.sum().clamp_min(
        1e-8
    )


def train_model(
    raw_features: np.ndarray,
    targets: np.ndarray,
    records: np.ndarray,
    fit: np.ndarray,
    early_stop: np.ndarray,
    cfg: dict,
    device: torch.device,
    checkpoint_path: Path,
    metadata: dict,
) -> tuple[FarRiskClassifier, dict]:
    mean = raw_features[fit].mean(axis=0)
    std = raw_features[fit].std(axis=0)
    std[std < 1e-6] = 1.0
    features = ((raw_features - mean) / std).astype(np.float32)
    fit_weights = record_balanced_weights(fit, records)
    validation_weights = record_balanced_weights(early_stop, records)
    positive_weighted = float(
        (fit_weights * targets[fit]).sum()
    )
    negative_weighted = float(
        (fit_weights * (1.0 - targets[fit])).sum()
    )
    if positive_weighted <= 0 or negative_weighted <= 0:
        raise RuntimeError("Phase4I.4 fit split requires both far classes")
    positive_weight = torch.tensor(
        [negative_weighted / positive_weighted],
        dtype=torch.float32,
        device=device,
    )
    model = FarRiskClassifier(features.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    best_loss, best_epoch, stale, history = float("inf"), 0, 0, []
    for epoch in range(1, int(cfg["epochs"]) + 1):
        model.train()
        order = np.random.permutation(len(fit))
        losses = []
        for start in range(0, len(order), int(cfg["batch_size"])):
            local = order[start : start + int(cfg["batch_size"])]
            batch = fit[local]
            logits = model(torch.from_numpy(features[batch]).to(device))
            loss = weighted_bce(
                logits,
                torch.from_numpy(targets[batch]).to(device),
                torch.from_numpy(fit_weights[local]).to(device),
                positive_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(),
                float(cfg["gradient_clip"]),
            )
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            validation_logits = model(
                torch.from_numpy(features[early_stop]).to(device)
            )
            validation_loss = float(
                weighted_bce(
                    validation_logits,
                    torch.from_numpy(targets[early_stop]).to(device),
                    torch.from_numpy(validation_weights).to(device),
                    positive_weight,
                ).cpu()
            )
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "validation_loss": validation_loss,
            }
        )
        if validation_loss < best_loss - 1e-6:
            best_loss, best_epoch, stale = validation_loss, epoch, 0
            ensure_dir(checkpoint_path.parent)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "feature_names": feature_names(),
                    "feature_mean": mean,
                    "feature_std": std,
                    "metadata": metadata,
                },
                checkpoint_path,
            )
        else:
            stale += 1
        if stale >= int(cfg["early_stopping_patience"]):
            break
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state"])
    return model, {
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "epochs_ran": len(history),
        "history": history,
    }


def probabilities_for(
    model: FarRiskClassifier,
    raw_features: np.ndarray,
    indices: np.ndarray,
    device: torch.device,
    checkpoint_path: Path,
) -> np.ndarray:
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    features = (
        (raw_features - checkpoint["feature_mean"])
        / checkpoint["feature_std"]
    ).astype(np.float32)
    model.eval()
    with torch.no_grad():
        return (
            torch.sigmoid(
                model(torch.from_numpy(features[indices]).to(device))
            )
            .cpu()
            .numpy()
            .astype(np.float32)
        )


def threshold_metrics(
    probabilities: np.ndarray,
    targets: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    predicted = probabilities >= threshold
    actual = targets.astype(bool)
    tp = int((predicted & actual).sum())
    fp = int((predicted & ~actual).sum())
    tn = int((~predicted & ~actual).sum())
    fn = int((~predicted & actual).sum())
    return {
        "threshold": float(threshold),
        "recall": tp / max(tp + fn, 1),
        "false_positive_rate": fp / max(fp + tn, 1),
        "precision": tp / max(tp + fp, 1),
        "risk_rate": float(predicted.mean()),
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
    }


def choose_recall_threshold(
    probabilities: np.ndarray,
    targets: np.ndarray,
    minimum_recall: float,
) -> tuple[dict, list[dict]]:
    if not 0 < minimum_recall <= 1:
        raise ValueError("Phase4I.4 minimum recall must be inside (0, 1]")
    if targets.sum() <= 0 or targets.sum() >= len(targets):
        raise RuntimeError("Phase4I.4 calibration requires both far classes")
    thresholds = sorted(
        {float(value) for value in probabilities.tolist()},
        reverse=True,
    )
    options = [
        threshold_metrics(probabilities, targets, threshold)
        for threshold in thresholds
    ]
    valid = [row for row in options if row["recall"] >= minimum_recall]
    if not valid:
        raise RuntimeError("Phase4I.4 could not satisfy far recall target")
    selected = min(
        valid,
        key=lambda row: (
            float(row["false_positive_rate"]),
            -float(row["recall"]),
            -float(row["threshold"]),
        ),
    )
    return selected, options


def threshold_csv_row(row: dict) -> dict[str, str]:
    output = {}
    for key in THRESHOLD_FIELDS:
        value = row[key]
        if key == "held_out_fold":
            output[key] = str(value)
        elif key in (
            "true_positives",
            "false_positives",
            "true_negatives",
            "false_negatives",
            "selected",
        ):
            output[key] = str(int(value))
        else:
            output[key] = f"{float(value):.9f}"
    return output


def assert_csv_schema(
    rows: list[dict[str, str]],
    fieldnames: list[str],
    label: str,
) -> None:
    if len(fieldnames) != len(set(fieldnames)):
        raise RuntimeError(f"Phase4I.4 {label} schema contains duplicates")
    expected = set(fieldnames)
    for index, row in enumerate(rows):
        actual = set(row)
        if actual != expected:
            raise RuntimeError(
                f"Phase4I.4 {label} row {index} schema mismatch: "
                f"missing={sorted(expected - actual)}, "
                f"extra={sorted(actual - expected)}"
            )


def retrieval_rows(
    rows: list[dict[str, str]],
    prefix: str,
) -> list[dict[str, str]]:
    output = []
    for row in rows:
        output.append(
            {
                "query_record_id": row["query_record_id"],
                "query_image_name": row["query_image_name"],
                "query_probe": row["query_probe"],
                "oof_fold": row["oof_fold"],
                "selected_cache_image_name": row[
                    f"{prefix}_selected_cache_image_name"
                ],
                "ranker_oracle_embedding_rank": row[
                    f"{prefix}_ranker_oracle_embedding_rank"
                ],
                **{
                    metric: row[f"{prefix}_{metric}"]
                    for metric in METRICS
                },
            }
        )
    return output


def select_by_risk(
    v1_rows: list[dict[str, str]],
    phase4i3_rows: list[dict[str, str]],
    predicted_risk: np.ndarray,
) -> list[dict[str, str]]:
    if len(v1_rows) != len(phase4i3_rows) or len(v1_rows) != len(
        predicted_risk
    ):
        raise ValueError("Phase4I.4 selection inputs differ in length")
    return [
        v1 if predicted_risk[index] else current
        for index, (v1, current) in enumerate(
            zip(v1_rows, phase4i3_rows, strict=True)
        )
    ]


def run(config_path: str, section: str) -> dict:
    cfg = load_config(config_path)[section]
    set_seed(int(cfg["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples = read_csv_rows(project_path(cfg["samples_csv"]))
    assert_development_only(samples, project_path(cfg["final_partition_csv"]))
    development_names = {
        row["image_name"]
        for row in samples
        if row["dataset_split"] == "train"
    }
    rows = read_csv_rows(project_path(cfg["phase4i3_query_csv"]))
    names = [row["query_image_name"] for row in rows]
    if len(set(names)) != len(rows) or set(names) != development_names:
        raise RuntimeError("Phase4I.4 queries do not match development OOF")
    records = np.asarray([row["query_record_id"] for row in rows])
    features = online_features(rows)
    far_probe_minimum = int(cfg["far_probe_minimum"])
    targets = far_labels(rows, far_probe_minimum)
    v1_rows = retrieval_rows(rows, "v1")
    phase4i3_rows = retrieval_rows(rows, "phase4i3")
    folds = sorted({row["oof_fold"] for row in rows})
    if len(folds) != 3:
        raise RuntimeError(f"Phase4I.4 requires three OOF folds, got {folds}")

    probabilities = np.zeros(len(rows), dtype=np.float32)
    thresholds = np.zeros(len(rows), dtype=np.float32)
    threshold_reports = []
    training_reports = []
    checkpoint_root = project_path(cfg["checkpoint_dir"])
    ensure_dir(checkpoint_root)
    for fold in folds:
        held_out = np.asarray(
            [
                index
                for index, row in enumerate(rows)
                if row["oof_fold"] == fold
            ],
            dtype=np.int32,
        )
        outer_fit = np.asarray(
            [
                index
                for index, row in enumerate(rows)
                if row["oof_fold"] != fold
            ],
            dtype=np.int32,
        )
        threshold_calibration = np.asarray(
            [
                index
                for index in outer_fit
                if record_hash_split(
                    records[index],
                    float(cfg["threshold_calibration_fraction"]),
                    int(cfg["threshold_calibration_seed"]) + int(fold),
                )
            ],
            dtype=np.int32,
        )
        calibration_set = set(threshold_calibration.tolist())
        fit_and_early = np.asarray(
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
                for index in fit_and_early
                if record_hash_split(
                    records[index],
                    float(cfg["early_stop_fraction"]),
                    int(cfg["early_stop_seed"]) + int(fold),
                )
            ],
            dtype=np.int32,
        )
        early_set = set(early_stop.tolist())
        fit = np.asarray(
            [
                index
                for index in fit_and_early
                if index not in early_set
            ],
            dtype=np.int32,
        )
        if not len(fit) or not len(early_stop) or not len(
            threshold_calibration
        ):
            raise RuntimeError(
                f"Phase4I.4 fold {fold} nested split contains an empty set"
            )
        for split_name, indices in (
            ("fit", fit),
            ("early_stop", early_stop),
            ("threshold_calibration", threshold_calibration),
        ):
            classes = np.unique(targets[indices])
            if len(classes) != 2:
                raise RuntimeError(
                    f"Phase4I.4 fold {fold} {split_name} "
                    "requires both far classes"
                )
        record_sets = [
            set(records[indices].tolist())
            for indices in (
                fit,
                early_stop,
                threshold_calibration,
                held_out,
            )
        ]
        if any(
            record_sets[left] & record_sets[right]
            for left in range(len(record_sets))
            for right in range(left + 1, len(record_sets))
        ):
            raise RuntimeError(
                f"Phase4I.4 fold {fold} nested record split overlaps"
            )
        calibration_values = []
        held_values = []
        for seed in [int(value) for value in cfg["seeds"]]:
            print(
                f"phase4i4 fold {fold} seed {seed}: training far-risk model",
                flush=True,
            )
            set_seed(seed)
            checkpoint_path = (
                checkpoint_root / f"fold_{fold}_seed_{seed}.pt"
            )
            model, training_report = train_model(
                features,
                targets,
                records,
                fit,
                early_stop,
                cfg,
                device,
                checkpoint_path,
                {
                    "scope": "strict_oof_far_risk_calibration",
                    "fold": fold,
                    "seed": seed,
                    "far_probe_minimum": far_probe_minimum,
                    "query_true_probe_feature_used": False,
                    "true_probe_offline_target_used": True,
                    "query_tactile_input": False,
                },
            )
            calibration_values.append(
                probabilities_for(
                    model,
                    features,
                    threshold_calibration,
                    device,
                    checkpoint_path,
                )
            )
            held_values.append(
                probabilities_for(
                    model,
                    features,
                    held_out,
                    device,
                    checkpoint_path,
                )
            )
            training_reports.append(
                {
                    "fold": fold,
                    "seed": seed,
                    "fit_queries": len(fit),
                    "early_stop_queries": len(early_stop),
                    "threshold_calibration_queries": len(
                        threshold_calibration
                    ),
                    "held_out_queries": len(held_out),
                    **training_report,
                }
            )
        calibration_probabilities = np.mean(
            calibration_values,
            axis=0,
        ).astype(np.float32)
        selected, options = choose_recall_threshold(
            calibration_probabilities,
            targets[threshold_calibration],
            float(cfg["minimum_far_recall"]),
        )
        selected_threshold = float(selected["threshold"])
        probabilities[held_out] = np.mean(
            held_values,
            axis=0,
        ).astype(np.float32)
        thresholds[held_out] = selected_threshold
        for option in options:
            threshold_reports.append(
                {
                    "held_out_fold": fold,
                    **option,
                    "selected": option is selected,
                }
            )
        print(
            f"phase4i4 fold {fold}: threshold={selected_threshold:.6f} "
            f"calibration_recall={selected['recall']:.3f} "
            f"calibration_fpr={selected['false_positive_rate']:.3f}",
            flush=True,
        )

    predicted_risk = probabilities >= thresholds
    gated_rows = select_by_risk(v1_rows, phase4i3_rows, predicted_risk)
    comparison = fast_bootstrap_comparison(
        v1_rows,
        gated_rows,
        {
            "bootstrap_iterations": int(cfg["bootstrap_iterations"]),
            "bootstrap_seed": int(cfg["bootstrap_seed"]),
        },
    )
    actual = targets.astype(bool)
    tp = int((predicted_risk & actual).sum())
    fp = int((predicted_risk & ~actual).sum())
    tn = int((~predicted_risk & ~actual).sum())
    fn = int((~predicted_risk & actual).sum())
    confusion = {
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "far_recall": tp / max(tp + fn, 1),
        "far_precision": tp / max(tp + fp, 1),
        "near_mid_retention": tn / max(tn + fp, 1),
        "risk_rate": float(predicted_risk.mean()),
    }
    gate_contract_pass = bool(
        confusion["far_recall"] >= float(cfg["minimum_far_recall"])
        and confusion["near_mid_retention"]
        >= float(cfg["minimum_near_mid_retention"])
    )
    accepted = bool(comparison["accepted"] and gate_contract_pass)
    final_rows = gated_rows if accepted else v1_rows

    query_output = []
    for index, (source, v1, current, gated, final) in enumerate(
        zip(rows, v1_rows, phase4i3_rows, gated_rows, final_rows, strict=True)
    ):
        gate_source = "v1_far_risk" if predicted_risk[index] else "phase4i3"
        final_source = gate_source if accepted else "v1"
        query_output.append(
            {
                "query_record_id": source["query_record_id"],
                "query_image_name": source["query_image_name"],
                "query_probe": source["query_probe"],
                "oof_fold": source["oof_fold"],
                "far_risk_probability": f"{probabilities[index]:.9f}",
                "far_risk_threshold": f"{thresholds[index]:.9f}",
                "predicted_far_risk": str(int(predicted_risk[index])),
                "true_far_label": str(int(targets[index])),
                "phase4i3_selected_cache_image_name": current[
                    "selected_cache_image_name"
                ],
                "v1_selected_cache_image_name": v1[
                    "selected_cache_image_name"
                ],
                "gated_selected_cache_image_name": gated[
                    "selected_cache_image_name"
                ],
                "gated_selection_source": gate_source,
                "gated_ranker_oracle_embedding_rank": gated[
                    "ranker_oracle_embedding_rank"
                ],
                **{
                    f"gated_{metric}": gated[metric]
                    for metric in METRICS
                },
                **{field: source[field] for field in ONLINE_PROGRESS_FIELDS},
                "deployment_accepted": str(int(accepted)),
                "final_selection_source": final_source,
                "selected_cache_image_name": final[
                    "selected_cache_image_name"
                ],
                "ranker_oracle_embedding_rank": final[
                    "ranker_oracle_embedding_rank"
                ],
                **{metric: final[metric] for metric in METRICS},
            }
        )
    threshold_output = [
        threshold_csv_row(row) for row in threshold_reports
    ]
    assert_csv_schema(query_output, QUERY_FIELDS, "query output")
    assert_csv_schema(
        threshold_output,
        THRESHOLD_FIELDS,
        "threshold output",
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

    report = {
        "mode": "phase4i4_far_risk_gate_oof_v1",
        "device": str(device),
        "feature_names": feature_names(),
        "gate_contract": {
            "minimum_far_recall": float(cfg["minimum_far_recall"]),
            "minimum_near_mid_retention": float(
                cfg["minimum_near_mid_retention"]
            ),
            "oof": confusion,
            "passed": gate_contract_pass,
        },
        "gated": {
            "summary": {
                "all": metric_summary(gated_rows),
                "far_probe75_100": metric_summary(
                    gated_rows,
                    lambda row: int(row["query_probe"])
                    >= far_probe_minimum,
                ),
            },
            "vs_v1": comparison,
        },
        "deployment_accepted": accepted,
        "deployed": {
            "source": "phase4i4" if accepted else "v1",
            "summary": {
                "all": metric_summary(final_rows),
                "far_probe75_100": metric_summary(
                    final_rows,
                    lambda row: int(row["query_probe"])
                    >= far_probe_minimum,
                ),
            },
        },
        "training": training_reports,
        "integrity": {
            "source": "strict development 3-fold record-level OOF only",
            "nested_split": (
                "fit / early-stop / threshold-calibration / held-out "
                "records are mutually exclusive"
            ),
            "query_true_probe_feature_used": False,
            "true_probe_offline_target_used": True,
            "query_tactile_input": False,
            "query_tactile_usage": "offline retrieval evaluation only",
            "c2_contact_box": "unchanged",
            "top32_candidates": "unchanged Phase4I.3",
            "sealed_final_holdout_rows_read": 0,
            "development_validation_rows_read": 0,
        },
        "next_action": (
            "freeze Phase4I.4 and implement independent development validation"
            if accepted
            else "retain V1; if far-risk separation is weak, improve TTC supervision or collect targeted far records"
        ),
    }
    write_json(project_path(cfg["metrics_json"]), report)
    print(
        {
            "mode": report["mode"],
            "gate_contract": report["gate_contract"],
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
        description="Train strict-OOF deployable far-horizon risk gate."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--section",
        default="phase4i4_far_risk_gate_oof_v1",
    )
    args = parser.parse_args()
    run(args.config, args.section)


if __name__ == "__main__":
    main()
