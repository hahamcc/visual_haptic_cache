"""Strict-OOF bounded residual cascade over frozen V1 Top-32 candidates."""
from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .config import load_config, project_path
from .evaluate_oracle_tactile_retrieval import tactile_difference, tactile_metrics
from .evaluate_phase4h_factorized_intensity_oof import fast_bootstrap_comparison
from .phase4h_dino_adaptation import (
    assert_candidate_identity,
    assert_development_only,
    candidate_groups,
    candidate_set_fingerprint,
    record_hash_split,
)
from .train_phase4b_predicted_box_cache_ranker import set_seed
from .train_phase4h_dino_gate import (
    choose_threshold,
    metric_summary,
    selected_rows,
)
from .utils import ensure_dir, read_csv_rows, write_csv_rows, write_json


METRICS = ("tactile_diff_mae", "tactile_ssim", "tactile_mask_iou")
QUERY_FIELDS = [
    "query_record_id",
    "query_image_name",
    "query_probe",
    "oof_fold",
    "cascade_weight_dino",
    "cascade_weight_intensity",
    "cascade_best_score",
    "cascade_margin",
    "cascade_selected_cache_record_id",
    "cascade_selected_cache_image_name",
    "cascade_ranker_oracle_embedding_rank",
    "cascade_tactile_diff_mae",
    "cascade_tactile_ssim",
    "cascade_tactile_mask_iou",
    "strict_triple_win_label",
    "gate_probability",
    "gate_threshold",
    "gate_candidate_accepted",
    "deployment_accepted",
    "final_selection_source",
    "selected_cache_record_id",
    "selected_cache_image_name",
    "ranker_oracle_embedding_rank",
    "tactile_diff_mae",
    "tactile_ssim",
    "tactile_mask_iou",
    "v1_selected_cache_image_name",
    "v1_tactile_diff_mae",
    "v1_tactile_ssim",
    "v1_tactile_mask_iou",
]
CANDIDATE_FIELDS = [
    "query_record_id",
    "query_image_name",
    "query_probe",
    "oof_fold",
    "candidate_rank",
    "cascade_candidate_rank",
    "final_candidate_rank",
    "cascade_score",
    "v1_score",
    "dino_score",
    "predicted_intensity_distance",
    "cascade_weight_dino",
    "cascade_weight_intensity",
    "candidate_record_id",
    "candidate_image_name",
    "candidate_tactile_embedding_distance",
    "candidate_oracle_embedding_rank",
    "gate_probability",
    "gate_candidate_accepted",
    "deployment_accepted",
    "final_selection_source",
]


def finite(value: str | float, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def required_float(row: dict[str, str], field: str) -> float:
    value = row.get(field, "")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Phase4I requires numeric {field} for "
            f"{row.get('query_image_name')}/{row.get('candidate_image_name')}"
        ) from exc
    if not math.isfinite(parsed):
        raise RuntimeError(
            f"Phase4I found non-finite {field} for "
            f"{row.get('query_image_name')}/{row.get('candidate_image_name')}"
        )
    return parsed


def grouped(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    output: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        output[row["query_image_name"]].append(row)
    for values in output.values():
        values.sort(key=lambda row: int(row["candidate_rank"]))
    return output


def query_standardize(values: np.ndarray) -> np.ndarray:
    mean = values.mean(axis=1, keepdims=True)
    std = values.std(axis=1, keepdims=True)
    return ((values - mean) / np.maximum(std, 1e-6)).astype(np.float32)


def normalized_entropy(scores: np.ndarray) -> np.ndarray:
    logits = -scores
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    entropy = -np.sum(
        probabilities * np.log(np.maximum(probabilities, 1e-12)),
        axis=1,
    )
    return (entropy / math.log(scores.shape[1])).astype(np.float32)


def normalized_margin(scores: np.ndarray) -> np.ndarray:
    ordered = np.sort(scores, axis=1)
    return (ordered[:, 1] - ordered[:, 0]).astype(np.float32)


def cascade_feature_names() -> list[str]:
    return [
        "v1_margin",
        "dino_margin",
        "intensity_margin",
        "v1_entropy",
        "dino_entropy",
        "intensity_entropy",
        "v1_dino_top1_agreement",
        "v1_intensity_top1_agreement",
        "dino_intensity_top1_agreement",
        "v1_dino_score_correlation",
        "v1_intensity_score_correlation",
    ]


def cascade_query_features(
    v1_scores: np.ndarray,
    dino_scores: np.ndarray,
    intensity_scores: np.ndarray,
) -> np.ndarray:
    v1_choice = v1_scores.argmin(axis=1)
    dino_choice = dino_scores.argmin(axis=1)
    intensity_choice = intensity_scores.argmin(axis=1)
    return np.stack(
        (
            normalized_margin(v1_scores),
            normalized_margin(dino_scores),
            normalized_margin(intensity_scores),
            normalized_entropy(v1_scores),
            normalized_entropy(dino_scores),
            normalized_entropy(intensity_scores),
            (v1_choice == dino_choice).astype(np.float32),
            (v1_choice == intensity_choice).astype(np.float32),
            (dino_choice == intensity_choice).astype(np.float32),
            (v1_scores * dino_scores).mean(axis=1),
            (v1_scores * intensity_scores).mean(axis=1),
        ),
        axis=1,
    ).astype(np.float32)


class FactorizedResidualCascade(nn.Module):
    """Query-conditioned, bounded corrections that keep V1 as the anchor."""

    def __init__(
        self,
        feature_dim: int,
        maximum_dino_weight: float,
        maximum_intensity_weight: float,
        initial_weight_logit: float,
    ) -> None:
        super().__init__()
        self.weight_head = nn.Linear(feature_dim, 2)
        nn.init.zeros_(self.weight_head.weight)
        nn.init.constant_(self.weight_head.bias, initial_weight_logit)
        self.register_buffer(
            "maximum_weights",
            torch.tensor(
                [maximum_dino_weight, maximum_intensity_weight],
                dtype=torch.float32,
            ),
        )

    def forward(
        self,
        v1_scores: torch.Tensor,
        dino_scores: torch.Tensor,
        intensity_scores: torch.Tensor,
        query_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weights = torch.sigmoid(self.weight_head(query_features))
        weights = weights * self.maximum_weights
        scores = (
            v1_scores
            + weights[:, 0, None] * dino_scores
            + weights[:, 1, None] * intensity_scores
        )
        return scores, weights


def record_balanced_weights(
    indices: np.ndarray,
    records: np.ndarray,
) -> np.ndarray:
    unique, inverse = np.unique(records[indices], return_inverse=True)
    counts = np.bincount(inverse, minlength=len(unique)).astype(np.float32)
    values = 1.0 / counts[inverse]
    return (values / max(float(values.mean()), 1e-8)).astype(np.float32)


def cascade_loss(
    model: FactorizedResidualCascade,
    v1_scores: torch.Tensor,
    dino_scores: torch.Tensor,
    intensity_scores: torch.Tensor,
    query_features: torch.Tensor,
    target_distances: torch.Tensor,
    query_weights: torch.Tensor,
    cfg: dict,
) -> tuple[torch.Tensor, dict[str, float]]:
    scores, residual_weights = model(
        v1_scores,
        dino_scores,
        intensity_scores,
        query_features,
    )
    target = torch.softmax(
        -target_distances / max(float(cfg["target_temperature"]), 1e-6),
        dim=1,
    )
    per_query = -(
        target
        * torch.log_softmax(
            -scores / max(float(cfg["score_temperature"]), 1e-6),
            dim=1,
        )
    ).sum(dim=1)
    listwise = (per_query * query_weights).sum() / query_weights.sum().clamp_min(
        1e-8
    )
    regularization = residual_weights.square().mean()
    total = listwise + float(cfg["residual_regularization"]) * regularization
    return total, {
        "total": float(total.detach().cpu()),
        "listwise": float(listwise.detach().cpu()),
        "regularization": float(regularization.detach().cpu()),
    }


def evaluate_cascade_loss(
    model: FactorizedResidualCascade,
    arrays: dict[str, np.ndarray],
    features: np.ndarray,
    indices: np.ndarray,
    records: np.ndarray,
    cfg: dict,
    device: torch.device,
) -> float:
    model.eval()
    weights = record_balanced_weights(indices, records)
    batch_size = int(cfg["batch_size"])
    total, denominator = 0.0, 0.0
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch = indices[start : start + batch_size]
            batch_weights = weights[start : start + len(batch)]
            loss, _ = cascade_loss(
                model,
                torch.from_numpy(arrays["v1"][batch]).to(device),
                torch.from_numpy(arrays["dino"][batch]).to(device),
                torch.from_numpy(arrays["intensity"][batch]).to(device),
                torch.from_numpy(features[batch]).to(device),
                torch.from_numpy(arrays["target"][batch]).to(device),
                torch.from_numpy(batch_weights).to(device),
                cfg,
            )
            weight = float(batch_weights.sum())
            total += float(loss.cpu()) * weight
            denominator += weight
    return total / max(denominator, 1e-8)


def train_cascade(
    arrays: dict[str, np.ndarray],
    raw_features: np.ndarray,
    fit: np.ndarray,
    validation: np.ndarray,
    records: np.ndarray,
    cfg: dict,
    device: torch.device,
    checkpoint_path: Path,
    metadata: dict,
) -> tuple[FactorizedResidualCascade, dict]:
    feature_mean = raw_features[fit].mean(axis=0)
    feature_std = raw_features[fit].std(axis=0)
    feature_std[feature_std < 1e-6] = 1.0
    features = ((raw_features - feature_mean) / feature_std).astype(np.float32)
    metadata = {
        **metadata,
        "feature_names": cascade_feature_names(),
        "feature_mean": feature_mean,
        "feature_std": feature_std,
    }
    model = FactorizedResidualCascade(
        features.shape[1],
        float(cfg["maximum_dino_weight"]),
        float(cfg["maximum_intensity_weight"]),
        float(cfg["initial_weight_logit"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    fit_weights = record_balanced_weights(fit, records)
    best_loss, best_epoch, stale, history = float("inf"), 0, 0, []
    batch_size = int(cfg["batch_size"])
    for epoch in range(1, int(cfg["epochs"]) + 1):
        model.train()
        order = np.random.permutation(len(fit))
        epoch_losses = []
        for start in range(0, len(order), batch_size):
            local = order[start : start + batch_size]
            batch = fit[local]
            loss, parts = cascade_loss(
                model,
                torch.from_numpy(arrays["v1"][batch]).to(device),
                torch.from_numpy(arrays["dino"][batch]).to(device),
                torch.from_numpy(arrays["intensity"][batch]).to(device),
                torch.from_numpy(features[batch]).to(device),
                torch.from_numpy(arrays["target"][batch]).to(device),
                torch.from_numpy(fit_weights[local]).to(device),
                cfg,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(),
                float(cfg["gradient_clip"]),
            )
            optimizer.step()
            epoch_losses.append(parts)
        validation_loss = evaluate_cascade_loss(
            model,
            arrays,
            features,
            validation,
            records,
            cfg,
            device,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(
                    np.mean([item["total"] for item in epoch_losses])
                ),
                "validation_loss": validation_loss,
            }
        )
        if validation_loss < best_loss - 1e-6:
            best_loss, best_epoch, stale = validation_loss, epoch, 0
            ensure_dir(checkpoint_path.parent)
            torch.save(
                {
                    "model_state": model.state_dict(),
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


def predict_cascade(
    model: FactorizedResidualCascade,
    arrays: dict[str, np.ndarray],
    raw_features: np.ndarray,
    indices: np.ndarray,
    device: torch.device,
    checkpoint_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    metadata = checkpoint["metadata"]
    features = (
        (raw_features - metadata["feature_mean"]) / metadata["feature_std"]
    ).astype(np.float32)
    scores, weights = [], []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(indices), 256):
            batch = indices[start : start + 256]
            batch_scores, batch_weights = model(
                torch.from_numpy(arrays["v1"][batch]).to(device),
                torch.from_numpy(arrays["dino"][batch]).to(device),
                torch.from_numpy(arrays["intensity"][batch]).to(device),
                torch.from_numpy(features[batch]).to(device),
            )
            scores.append(batch_scores.cpu().numpy())
            weights.append(batch_weights.cpu().numpy())
    return (
        np.concatenate(scores).astype(np.float32),
        np.concatenate(weights).astype(np.float32),
    )


def gate_feature_names() -> list[str]:
    return [
        "cascade_margin",
        "cascade_entropy",
        "v1_margin",
        "dino_margin",
        "intensity_margin",
        "cascade_v1_top1_agreement",
        "cascade_dino_top1_agreement",
        "cascade_intensity_top1_agreement",
        "dino_weight",
        "intensity_weight",
    ]


def gate_features(
    arrays: dict[str, np.ndarray],
    cascade_scores: np.ndarray,
    cascade_weights: np.ndarray,
) -> np.ndarray:
    cascade_choice = cascade_scores.argmin(axis=1)
    return np.stack(
        (
            normalized_margin(cascade_scores),
            normalized_entropy(cascade_scores),
            normalized_margin(arrays["v1"]),
            normalized_margin(arrays["dino"]),
            normalized_margin(arrays["intensity"]),
            (cascade_choice == arrays["v1"].argmin(axis=1)).astype(np.float32),
            (cascade_choice == arrays["dino"].argmin(axis=1)).astype(np.float32),
            (cascade_choice == arrays["intensity"].argmin(axis=1)).astype(
                np.float32
            ),
            cascade_weights[:, 0],
            cascade_weights[:, 1],
        ),
        axis=1,
    ).astype(np.float32)


def build_gate_oof_splits(
    query_rows: list[dict[str, str]],
    records: np.ndarray,
    folds: list[str],
    targets: np.ndarray,
    cfg: dict,
) -> tuple[list[dict[str, np.ndarray | str]], str]:
    splits: list[dict[str, np.ndarray | str]] = []
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
        inner_validation = np.asarray(
            [
                index
                for index in outer_fit
                if record_hash_split(
                    records[index],
                    float(cfg["inner_validation_fraction"]),
                    int(cfg["gate_inner_split_seed"]) + int(fold),
                )
            ],
            dtype=np.int32,
        )
        validation_set = set(inner_validation.tolist())
        inner_fit = np.asarray(
            [index for index in outer_fit if index not in validation_set],
            dtype=np.int32,
        )
        for name, indices, minimum_per_class in (
            ("outer fit", outer_fit, 4),
            ("inner fit", inner_fit, 4),
            ("inner validation", inner_validation, 1),
        ):
            if not len(indices):
                return [], f"fold {fold} {name} is empty"
            positives = int(targets[indices].sum())
            negatives = len(indices) - positives
            if (
                positives < minimum_per_class
                or negatives < minimum_per_class
            ):
                return [], (
                    f"fold {fold} {name} strict-triple labels are too "
                    f"imbalanced: positive={positives}, negative={negatives}"
                )
        splits.append(
            {
                "fold": fold,
                "held_out": held_out,
                "inner_fit": inner_fit,
                "inner_validation": inner_validation,
            }
        )
    return splits, ""


def train_linear_gate(
    raw_features: np.ndarray,
    targets: np.ndarray,
    fit: np.ndarray,
    validation: np.ndarray,
    cfg: dict,
    device: torch.device,
    checkpoint_path: Path,
    metadata: dict,
) -> tuple[nn.Linear, dict]:
    positive = float(targets[fit].sum())
    if positive < 4 or positive >= len(fit) - 4:
        raise RuntimeError(
            f"Phase4I gate labels are too imbalanced: {positive}/{len(fit)}"
        )
    mean = raw_features[fit].mean(axis=0)
    std = raw_features[fit].std(axis=0)
    std[std < 1e-6] = 1.0
    features = ((raw_features - mean) / std).astype(np.float32)
    model = nn.Linear(features.shape[1], 1).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["gate_learning_rate"]),
        weight_decay=float(cfg["gate_weight_decay"]),
    )
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            [(len(fit) - positive) / positive],
            device=device,
        )
    )
    best_loss, best_epoch, stale, history = float("inf"), 0, 0, []
    for epoch in range(1, int(cfg["gate_epochs"]) + 1):
        model.train()
        order = np.random.permutation(fit)
        train_losses = []
        for start in range(0, len(order), int(cfg["batch_size"])):
            batch = order[start : start + int(cfg["batch_size"])]
            logits = model(torch.from_numpy(features[batch]).to(device)).squeeze(
                1
            )
            loss = criterion(
                logits,
                torch.from_numpy(targets[batch]).to(device),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            validation_logits = model(
                torch.from_numpy(features[validation]).to(device)
            ).squeeze(1)
            validation_loss = float(
                criterion(
                    validation_logits,
                    torch.from_numpy(targets[validation]).to(device),
                ).cpu()
            )
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(train_losses)),
                "validation_loss": validation_loss,
            }
        )
        if validation_loss < best_loss - 1e-6:
            best_loss, best_epoch, stale = validation_loss, epoch, 0
            ensure_dir(checkpoint_path.parent)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "metadata": {
                        **metadata,
                        "feature_names": gate_feature_names(),
                        "feature_mean": mean,
                        "feature_std": std,
                    },
                },
                checkpoint_path,
            )
        else:
            stale += 1
        if stale >= int(cfg["gate_early_stopping_patience"]):
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


def predict_gate(
    model: nn.Linear,
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
    metadata = checkpoint["metadata"]
    features = (
        (raw_features - metadata["feature_mean"]) / metadata["feature_std"]
    ).astype(np.float32)
    model.eval()
    with torch.no_grad():
        return (
            model(torch.from_numpy(features[indices]).to(device))
            .squeeze(1)
            .cpu()
            .numpy()
            .astype(np.float32)
        )


def build_cascade_query_rows(
    query_rows: list[dict[str, str]],
    groups: dict[str, list[dict[str, str]]],
    choices: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray,
    source_by_name: dict[str, dict[str, str]],
    cfg: dict,
) -> list[dict[str, str]]:
    touch_cache: dict[str, np.ndarray] = {}
    output = []
    for index, query in enumerate(query_rows):
        if index % 250 == 0:
            print(
                f"phase4i tactile evaluation: {index}/{len(query_rows)} queries",
                flush=True,
            )
        group = groups[query["query_image_name"]]
        choice = int(choices[index])
        candidate = group[choice]
        selected = source_by_name[candidate["candidate_image_name"]]
        metric = tactile_metrics(
            tactile_difference(
                source_by_name[query["query_image_name"]]["touch_path"],
                touch_cache,
                int(cfg["tactile_size"]),
            ),
            tactile_difference(
                selected["touch_path"],
                touch_cache,
                int(cfg["tactile_size"]),
            ),
            float(cfg["tactile_mask_threshold"]),
        )
        ordered = np.argsort(scores[index], kind="stable")
        output.append(
            {
                "query_record_id": query["query_record_id"],
                "query_image_name": query["query_image_name"],
                "query_probe": query["query_probe"],
                "oof_fold": query["oof_fold"],
                "selected_cache_record_id": candidate["candidate_record_id"],
                "selected_cache_image_name": candidate["candidate_image_name"],
                "ranker_oracle_embedding_rank": candidate[
                    "candidate_oracle_embedding_rank"
                ],
                "ranker_best_score": f"{scores[index, choice]:.9f}",
                "ranker_margin": (
                    f"{scores[index, int(ordered[1])] - scores[index, choice]:.9f}"
                ),
                "cascade_weight_dino": f"{weights[index, 0]:.9f}",
                "cascade_weight_intensity": f"{weights[index, 1]:.9f}",
                **{
                    metric_name: f"{metric[metric_name]:.9f}"
                    for metric_name in METRICS
                },
            }
        )
    print(
        f"phase4i tactile evaluation: {len(query_rows)}/{len(query_rows)} queries",
        flush=True,
    )
    return output


def strict_triple_labels(
    v1_rows: list[dict[str, str]],
    cascade_rows: list[dict[str, str]],
) -> np.ndarray:
    v1 = {row["query_image_name"]: row for row in v1_rows}
    return np.asarray(
        [
            float(
                finite(row["tactile_diff_mae"])
                < finite(v1[row["query_image_name"]]["tactile_diff_mae"])
                and finite(row["tactile_ssim"])
                >= finite(v1[row["query_image_name"]]["tactile_ssim"])
                and finite(row["tactile_mask_iou"])
                >= finite(v1[row["query_image_name"]]["tactile_mask_iou"])
            )
            for row in cascade_rows
        ],
        dtype=np.float32,
    )


def train(config_path: str, section: str) -> dict:
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
    v1_by_name = {
        row["query_image_name"]: row
        for row in read_csv_rows(project_path(cfg["v1_query_csv"]))
    }
    query_names = list(v1_by_name)
    query_rows = [v1_by_name[name] for name in query_names]
    if set(query_names) != set(source_by_name):
        raise RuntimeError("Phase4I V1 queries do not match development-train rows")

    top_k = int(cfg["geometry_filter_k"])
    v1_groups = candidate_groups(
        read_csv_rows(project_path(cfg["v1_candidate_csv"])),
        top_k,
    )
    factor_groups = candidate_groups(
        read_csv_rows(project_path(cfg["factor_candidate_csv"])),
        top_k,
    )
    assert_candidate_identity(v1_groups, factor_groups)
    if set(factor_groups) != set(query_names):
        raise RuntimeError("Phase4I candidate queries do not match V1 queries")
    candidate_fingerprint = candidate_set_fingerprint(factor_groups)
    factor_groups = grouped(
        [
            row
            for values in factor_groups.values()
            for row in values
        ]
    )
    for name in query_names:
        if any(
            row["candidate_record_id"] == row["query_record_id"]
            for row in factor_groups[name]
        ):
            raise RuntimeError(f"Phase4I same-record candidate found for {name}")
        if any(
            row["oof_fold"] != v1_by_name[name]["oof_fold"]
            for row in factor_groups[name]
        ):
            raise RuntimeError(f"Phase4I fold mismatch for {name}")

    def matrix(field: str) -> np.ndarray:
        return np.asarray(
            [
                [required_float(row, field) for row in factor_groups[name]]
                for name in query_names
            ],
            dtype=np.float32,
        )

    arrays = {
        "v1": query_standardize(matrix("v1_score")),
        "dino": query_standardize(matrix("dino_score")),
        "intensity": query_standardize(
            matrix("predicted_intensity_distance_dino_motion")
        ),
        "target": matrix("candidate_tactile_embedding_distance"),
    }
    raw_features = cascade_query_features(
        arrays["v1"],
        arrays["dino"],
        arrays["intensity"],
    )
    records = np.asarray(
        [row["query_record_id"] for row in query_rows]
    )
    folds = sorted({row["oof_fold"] for row in query_rows})
    if len(folds) != 3:
        raise RuntimeError(f"Phase4I requires three OOF folds, got {folds}")

    cascade_scores = np.zeros((len(query_rows), top_k), dtype=np.float32)
    cascade_weights = np.zeros((len(query_rows), 2), dtype=np.float32)
    cascade_reports = []
    cascade_checkpoint_dir = project_path(cfg["cascade_checkpoint_dir"])
    ensure_dir(cascade_checkpoint_dir)
    seeds = [int(value) for value in cfg["seeds"]]
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
        inner_validation = np.asarray(
            [
                index
                for index in outer_fit
                if record_hash_split(
                    records[index],
                    float(cfg["inner_validation_fraction"]),
                    int(cfg["inner_split_seed"]) + int(fold),
                )
            ],
            dtype=np.int32,
        )
        validation_set = set(inner_validation.tolist())
        inner_fit = np.asarray(
            [index for index in outer_fit if index not in validation_set],
            dtype=np.int32,
        )
        if not len(inner_fit) or not len(inner_validation):
            raise RuntimeError(f"Phase4I fold {fold} inner split is empty")
        seed_scores, seed_weights = [], []
        for seed in seeds:
            print(
                f"phase4i cascade fold {fold} seed {seed}: training",
                flush=True,
            )
            set_seed(seed)
            checkpoint_path = (
                cascade_checkpoint_dir / f"fold_{fold}_seed_{seed}.pt"
            )
            model, report = train_cascade(
                arrays,
                raw_features,
                inner_fit,
                inner_validation,
                records,
                cfg,
                device,
                checkpoint_path,
                {
                    "scope": "strict_oof_factorized_residual_cascade",
                    "fold": fold,
                    "seed": seed,
                    "candidate_fingerprint": candidate_fingerprint,
                    "query_tactile_input": False,
                    "candidate_tactile_usage": "offline ranking supervision only",
                },
            )
            scores, weights = predict_cascade(
                model,
                arrays,
                raw_features,
                held_out,
                device,
                checkpoint_path,
            )
            seed_scores.append(scores)
            seed_weights.append(weights)
            cascade_reports.append(
                {"fold": fold, "seed": seed, **report}
            )
            print(
                f"phase4i cascade fold {fold} seed {seed}: "
                f"best_epoch={report['best_epoch']} "
                f"validation_loss={report['best_validation_loss']:.6f}",
                flush=True,
            )
        cascade_scores[held_out] = np.mean(seed_scores, axis=0)
        cascade_weights[held_out] = np.mean(seed_weights, axis=0)

    cascade_choices = cascade_scores.argmin(axis=1)
    cascade_rows = build_cascade_query_rows(
        query_rows,
        factor_groups,
        cascade_choices,
        cascade_scores,
        cascade_weights,
        source_by_name,
        cfg,
    )
    targets = strict_triple_labels(query_rows, cascade_rows)

    raw_gate_features = gate_features(
        arrays,
        cascade_scores,
        cascade_weights,
    )
    gate_logits = np.zeros(len(query_rows), dtype=np.float32)
    gate_reports = []
    gate_checkpoint_dir = project_path(cfg["gate_checkpoint_dir"])
    ensure_dir(gate_checkpoint_dir)
    gate_splits, gate_block_reason = build_gate_oof_splits(
        query_rows,
        records,
        folds,
        targets,
        cfg,
    )
    if gate_splits:
        for split in gate_splits:
            fold = str(split["fold"])
            held_out = split["held_out"]
            inner_fit = split["inner_fit"]
            inner_validation = split["inner_validation"]
            print(f"phase4i gate fold {fold}: training", flush=True)
            checkpoint_path = gate_checkpoint_dir / f"fold_{fold}.pt"
            model, report = train_linear_gate(
                raw_gate_features,
                targets,
                inner_fit,
                inner_validation,
                cfg,
                device,
                checkpoint_path,
                {
                    "scope": "strict_oof_factorized_residual_gate",
                    "fold": fold,
                    "candidate_fingerprint": candidate_fingerprint,
                    "query_tactile_input": False,
                },
            )
            gate_logits[held_out] = predict_gate(
                model,
                raw_gate_features,
                held_out,
                device,
                checkpoint_path,
            )
            gate_reports.append({"fold": fold, **report})
            print(
                f"phase4i gate fold {fold}: "
                f"best_epoch={report['best_epoch']} "
                f"validation_loss={report['best_validation_loss']:.6f}",
                flush=True,
            )
        gate_probabilities = 1.0 / (
            1.0 + np.exp(-np.clip(gate_logits, -60.0, 60.0))
        )
        gate = choose_threshold(
            gate_probabilities,
            targets,
            query_rows,
            cascade_rows,
            float(cfg["minimum_gate_coverage"]),
            float(cfg["minimum_gate_precision"]),
        )
    else:
        print(
            f"phase4i gate disabled safely: {gate_block_reason}",
            flush=True,
        )
        gate_probabilities = np.zeros(len(query_rows), dtype=np.float32)
        gate = {
            "enabled": False,
            "reason": gate_block_reason,
            "options": [],
        }
    threshold = (
        float(gate["selected"]["threshold"]) if gate["enabled"] else None
    )
    gate_accepted = (
        gate_probabilities >= threshold
        if threshold is not None
        else np.zeros(len(query_rows), dtype=bool)
    )
    gated_rows = selected_rows(query_rows, cascade_rows, gate_accepted)
    comparison_cfg = {
        "bootstrap_iterations": int(cfg["bootstrap_iterations"]),
        "bootstrap_seed": int(cfg["bootstrap_seed"]),
    }
    print("phase4i: starting vectorized record bootstrap", flush=True)
    cascade_comparison = fast_bootstrap_comparison(
        query_rows,
        cascade_rows,
        comparison_cfg,
    )
    gated_comparison = fast_bootstrap_comparison(
        query_rows,
        gated_rows,
        comparison_cfg,
    )
    deployment_accepted = bool(
        gate["enabled"] and gated_comparison["accepted"]
    )
    final_rows = gated_rows if deployment_accepted else query_rows

    query_output = []
    for index, (v1, cascade, gated, final) in enumerate(
        zip(query_rows, cascade_rows, gated_rows, final_rows, strict=True)
    ):
        query_output.append(
            {
                "query_record_id": v1["query_record_id"],
                "query_image_name": v1["query_image_name"],
                "query_probe": v1["query_probe"],
                "oof_fold": v1["oof_fold"],
                "cascade_weight_dino": cascade["cascade_weight_dino"],
                "cascade_weight_intensity": cascade[
                    "cascade_weight_intensity"
                ],
                "cascade_best_score": cascade["ranker_best_score"],
                "cascade_margin": cascade["ranker_margin"],
                "cascade_selected_cache_record_id": cascade[
                    "selected_cache_record_id"
                ],
                "cascade_selected_cache_image_name": cascade[
                    "selected_cache_image_name"
                ],
                "cascade_ranker_oracle_embedding_rank": cascade[
                    "ranker_oracle_embedding_rank"
                ],
                **{
                    f"cascade_{metric}": cascade[metric]
                    for metric in METRICS
                },
                "strict_triple_win_label": str(int(targets[index])),
                "gate_probability": f"{gate_probabilities[index]:.9f}",
                "gate_threshold": (
                    "" if threshold is None else f"{threshold:.9f}"
                ),
                "gate_candidate_accepted": str(int(gate_accepted[index])),
                "deployment_accepted": str(int(deployment_accepted)),
                "final_selection_source": (
                    "factorized_residual_cascade"
                    if deployment_accepted and gate_accepted[index]
                    else "v1"
                ),
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
                "v1_selected_cache_image_name": v1[
                    "selected_cache_image_name"
                ],
                **{f"v1_{metric}": v1[metric] for metric in METRICS},
            }
        )

    candidate_output = []
    for query_index, query in enumerate(query_rows):
        group = factor_groups[query["query_image_name"]]
        cascade_rank = np.argsort(
            np.argsort(cascade_scores[query_index], kind="stable"),
            kind="stable",
        ) + 1
        for candidate_index, row in enumerate(group):
            use_cascade = bool(
                deployment_accepted and gate_accepted[query_index]
            )
            candidate_output.append(
                {
                    "query_record_id": query["query_record_id"],
                    "query_image_name": query["query_image_name"],
                    "query_probe": query["query_probe"],
                    "oof_fold": query["oof_fold"],
                    "candidate_rank": row["candidate_rank"],
                    "cascade_candidate_rank": str(
                        int(cascade_rank[candidate_index])
                    ),
                    "final_candidate_rank": (
                        str(int(cascade_rank[candidate_index]))
                        if use_cascade
                        else row["candidate_rank"]
                    ),
                    "cascade_score": (
                        f"{cascade_scores[query_index, candidate_index]:.9f}"
                    ),
                    "v1_score": row["v1_score"],
                    "dino_score": row["dino_score"],
                    "predicted_intensity_distance": row[
                        "predicted_intensity_distance_dino_motion"
                    ],
                    "cascade_weight_dino": (
                        f"{cascade_weights[query_index, 0]:.9f}"
                    ),
                    "cascade_weight_intensity": (
                        f"{cascade_weights[query_index, 1]:.9f}"
                    ),
                    "candidate_record_id": row["candidate_record_id"],
                    "candidate_image_name": row["candidate_image_name"],
                    "candidate_tactile_embedding_distance": row[
                        "candidate_tactile_embedding_distance"
                    ],
                    "candidate_oracle_embedding_rank": row[
                        "candidate_oracle_embedding_rank"
                    ],
                    "gate_probability": (
                        f"{gate_probabilities[query_index]:.9f}"
                    ),
                    "gate_candidate_accepted": str(
                        int(gate_accepted[query_index])
                    ),
                    "deployment_accepted": str(int(deployment_accepted)),
                    "final_selection_source": (
                        "factorized_residual_cascade"
                        if use_cascade
                        else "v1"
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
        "mode": "phase4i_strict_oof_factorized_residual_cascade_v1",
        "device": str(device),
        "cascade": {
            "summary": {
                "all": metric_summary(cascade_rows),
                "far_probe75_100": metric_summary(
                    cascade_rows,
                    lambda row: int(row["query_probe"]) >= 75,
                ),
            },
            "vs_v1": cascade_comparison,
            "mean_weights": {
                "dino": float(cascade_weights[:, 0].mean()),
                "intensity": float(cascade_weights[:, 1].mean()),
            },
        },
        "gate": {
            "enabled_by_point_guard": bool(gate["enabled"]),
            "threshold": threshold,
            "candidate_coverage": float(gate_accepted.mean()),
            "strict_triple_win_precision": (
                float(targets[gate_accepted].mean())
                if gate_accepted.any()
                else 0.0
            ),
            "selection": gate,
        },
        "gated_candidate": {
            "summary": {
                "all": metric_summary(gated_rows),
                "far_probe75_100": metric_summary(
                    gated_rows,
                    lambda row: int(row["query_probe"]) >= 75,
                ),
            },
            "vs_v1": gated_comparison,
        },
        "deployment_accepted": deployment_accepted,
        "deployed": {
            "source": (
                "gated_factorized_residual_cascade"
                if deployment_accepted
                else "v1"
            ),
            "summary": {
                "all": metric_summary(final_rows),
                "far_probe75_100": metric_summary(
                    final_rows,
                    lambda row: int(row["query_probe"]) >= 75,
                ),
            },
        },
        "training": {
            "cascade": cascade_reports,
            "gate": gate_reports,
        },
        "integrity": {
            "source": "strict development 3-fold record-level OOF only",
            "c2_contact_box": "unchanged",
            "top32_candidates": "frozen V1 identity",
            "candidate_fingerprint": candidate_fingerprint,
            "same_record_candidates": 0,
            "query_true_probe_used": False,
            "query_tactile_input": False,
            "query_tactile_usage": "offline ranking labels and evaluation only",
            "sealed_final_holdout_rows_read": 0,
            "development_validation_rows_read": 0,
            "hard_candidate_truncation": False,
        },
        "next_action": (
            "freeze full-development cascade and run independent development validation"
            if deployment_accepted
            else "retain V1; inspect cascade ranking correlation and gate errors without touching final holdout"
        ),
    }
    write_json(project_path(cfg["metrics_json"]), report)
    print(
        {
            key: value
            for key, value in report.items()
            if key != "training"
        }
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train strict-OOF Phase4I factorized residual cascade."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--section",
        default="phase4i_factorized_residual_cascade_oof_v1",
    )
    args = parser.parse_args()
    train(args.config, args.section)


if __name__ == "__main__":
    main()
