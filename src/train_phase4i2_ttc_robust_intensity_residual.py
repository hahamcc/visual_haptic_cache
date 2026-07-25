"""Strict-OOF intensity-only residual ranking with predicted-TTC robustness."""
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
    assert_candidate_identity,
    assert_development_only,
    candidate_groups,
    candidate_set_fingerprint,
    record_hash_split,
)
from .train_phase4b_predicted_box_cache_ranker import set_seed
from .train_phase4h_dino_gate import metric_summary
from .train_phase4i_factorized_residual_cascade import (
    METRICS,
    ONLINE_PROGRESS_FIELDS,
    build_cascade_query_rows,
    consistent_reference_rows,
    grouped,
    normalized_entropy,
    normalized_margin,
    online_progress_features,
    query_standardize,
    rank_positions,
    record_balanced_weights,
    required_float,
)
from .utils import ensure_dir, read_csv_rows, write_csv_rows, write_json


VARIANTS = ("plain_intensity", "ttc_robust_intensity")
QUERY_FIELDS = [
    "query_record_id",
    "query_image_name",
    "query_probe",
    "oof_fold",
    "v1_ranker_oracle_embedding_rank",
    "v1_selected_cache_image_name",
    *[f"v1_{metric}" for metric in METRICS],
    "plain_weight",
    "plain_selected_cache_image_name",
    "plain_ranker_oracle_embedding_rank",
    *[f"plain_{metric}" for metric in METRICS],
    "robust_raw_weight",
    "robust_attenuation",
    "robust_effective_weight",
    "robust_selected_cache_image_name",
    "robust_ranker_oracle_embedding_rank",
    *[f"robust_{metric}" for metric in METRICS],
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
    "candidate_rank",
    "plain_candidate_rank",
    "robust_candidate_rank",
    "final_candidate_rank",
    "v1_score",
    "predicted_intensity_distance",
    "plain_score",
    "robust_score",
    "plain_weight",
    "robust_raw_weight",
    "robust_attenuation",
    "robust_effective_weight",
    "candidate_record_id",
    "candidate_image_name",
    "candidate_tactile_embedding_distance",
    "candidate_oracle_embedding_rank",
    "deployment_accepted",
    "final_selection_source",
]


def feature_names() -> list[str]:
    return [
        "v1_margin",
        "intensity_margin",
        "v1_entropy",
        "intensity_entropy",
        "v1_intensity_top1_agreement",
        "v1_intensity_score_correlation",
        *ONLINE_PROGRESS_FIELDS,
    ]


def query_features(
    v1_scores: np.ndarray,
    intensity_scores: np.ndarray,
    progress: np.ndarray,
) -> np.ndarray:
    base = np.stack(
        (
            normalized_margin(v1_scores),
            normalized_margin(intensity_scores),
            normalized_entropy(v1_scores),
            normalized_entropy(intensity_scores),
            (
                v1_scores.argmin(axis=1)
                == intensity_scores.argmin(axis=1)
            ).astype(np.float32),
            (v1_scores * intensity_scores).mean(axis=1),
        ),
        axis=1,
    ).astype(np.float32)
    if progress.shape != (len(base), len(ONLINE_PROGRESS_FIELDS)):
        raise ValueError(
            "Phase4I.2 progress features do not match query feature contract"
        )
    return np.concatenate((base, progress), axis=1).astype(np.float32)


def predicted_ttc_groups(
    progress: np.ndarray,
    thresholds: tuple[float, float],
) -> np.ndarray:
    predicted_ttc = progress[:, ONLINE_PROGRESS_FIELDS.index("predicted_ttc")]
    return np.digitize(predicted_ttc, thresholds).astype(np.int32)


def robust_query_weights(
    indices: np.ndarray,
    records: np.ndarray,
    progress: np.ndarray,
    thresholds: tuple[float, float],
    robust: bool,
) -> np.ndarray:
    weights = record_balanced_weights(indices, records)
    if not robust:
        return weights
    groups = predicted_ttc_groups(progress, thresholds)[indices]
    unique, inverse = np.unique(groups, return_inverse=True)
    if len(unique) < 2:
        raise RuntimeError(
            "Phase4I.2 robust fit requires at least two predicted-TTC groups"
        )
    group_totals = np.bincount(
        inverse,
        weights=weights,
        minlength=len(unique),
    ).astype(np.float32)
    weights = weights / group_totals[inverse].clip(min=1e-8)
    return (weights / max(float(weights.mean()), 1e-8)).astype(np.float32)


def uncertainty_attenuation(
    progress: np.ndarray,
    cfg: dict,
) -> np.ndarray:
    predicted_ttc = np.clip(
        progress[:, ONLINE_PROGRESS_FIELDS.index("predicted_ttc")] / 100.0,
        0.0,
        1.0,
    )
    entropy = np.clip(
        progress[:, ONLINE_PROGRESS_FIELDS.index("ttc_entropy")],
        0.0,
        1.0,
    )
    trajectory_padding = np.clip(
        progress[:, ONLINE_PROGRESS_FIELDS.index("trajectory_padding_ratio")],
        0.0,
        1.0,
    )
    crop_padding = np.clip(
        progress[:, ONLINE_PROGRESS_FIELDS.index("query_padding_ratio")],
        0.0,
        1.0,
    )
    padding = np.maximum(trajectory_padding, crop_padding)
    attenuation = (
        1.0 - float(cfg["entropy_attenuation"]) * entropy
    ) * (
        1.0 - float(cfg["padding_attenuation"]) * padding
    ) * (
        1.0
        - float(cfg["far_entropy_attenuation"]) * predicted_ttc * entropy
    )
    return np.clip(
        attenuation,
        float(cfg["minimum_attenuation"]),
        1.0,
    ).astype(np.float32)


class IntensityResidualRanker(nn.Module):
    """One bounded query-conditioned intensity correction over frozen V1."""

    def __init__(
        self,
        feature_dim: int,
        maximum_weight: float,
        initial_weight_logit: float,
    ) -> None:
        super().__init__()
        self.weight_head = nn.Linear(feature_dim, 1)
        nn.init.zeros_(self.weight_head.weight)
        nn.init.constant_(self.weight_head.bias, initial_weight_logit)
        self.maximum_weight = float(maximum_weight)

    def forward(
        self,
        v1_scores: torch.Tensor,
        intensity_scores: torch.Tensor,
        features: torch.Tensor,
        attenuation: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw_weight = (
            torch.sigmoid(self.weight_head(features)).squeeze(1)
            * self.maximum_weight
        )
        effective_weight = raw_weight * attenuation
        scores = v1_scores + effective_weight[:, None] * intensity_scores
        return scores, raw_weight, effective_weight


def listwise_loss(
    model: IntensityResidualRanker,
    v1_scores: torch.Tensor,
    intensity_scores: torch.Tensor,
    features: torch.Tensor,
    attenuation: torch.Tensor,
    target_distances: torch.Tensor,
    query_weights: torch.Tensor,
    cfg: dict,
) -> tuple[torch.Tensor, dict[str, float]]:
    scores, raw_weight, effective_weight = model(
        v1_scores,
        intensity_scores,
        features,
        attenuation,
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
    regularization = raw_weight.square().mean()
    total = listwise + float(cfg["residual_regularization"]) * regularization
    return total, {
        "total": float(total.detach().cpu()),
        "listwise": float(listwise.detach().cpu()),
        "regularization": float(regularization.detach().cpu()),
        "mean_effective_weight": float(effective_weight.mean().detach().cpu()),
    }


def evaluate_loss(
    model: IntensityResidualRanker,
    arrays: dict[str, np.ndarray],
    features: np.ndarray,
    attenuation: np.ndarray,
    indices: np.ndarray,
    records: np.ndarray,
    progress: np.ndarray,
    robust: bool,
    cfg: dict,
    device: torch.device,
) -> float:
    thresholds = tuple(float(value) for value in cfg["predicted_ttc_groups"])
    weights = robust_query_weights(
        indices,
        records,
        progress,
        thresholds,
        robust,
    )
    total, denominator = 0.0, 0.0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(indices), int(cfg["batch_size"])):
            batch = indices[start : start + int(cfg["batch_size"])]
            local_weights = weights[start : start + len(batch)]
            loss, _ = listwise_loss(
                model,
                torch.from_numpy(arrays["v1"][batch]).to(device),
                torch.from_numpy(arrays["intensity"][batch]).to(device),
                torch.from_numpy(features[batch]).to(device),
                torch.from_numpy(attenuation[batch]).to(device),
                torch.from_numpy(arrays["target"][batch]).to(device),
                torch.from_numpy(local_weights).to(device),
                cfg,
            )
            weight = float(local_weights.sum())
            total += float(loss.cpu()) * weight
            denominator += weight
    return total / max(denominator, 1e-8)


def train_ranker(
    arrays: dict[str, np.ndarray],
    raw_features: np.ndarray,
    progress: np.ndarray,
    attenuation: np.ndarray,
    fit: np.ndarray,
    validation: np.ndarray,
    records: np.ndarray,
    robust: bool,
    cfg: dict,
    device: torch.device,
    checkpoint_path: Path,
    metadata: dict,
) -> tuple[IntensityResidualRanker, dict]:
    mean = raw_features[fit].mean(axis=0)
    std = raw_features[fit].std(axis=0)
    std[std < 1e-6] = 1.0
    features = ((raw_features - mean) / std).astype(np.float32)
    thresholds = tuple(float(value) for value in cfg["predicted_ttc_groups"])
    fit_weights = robust_query_weights(
        fit,
        records,
        progress,
        thresholds,
        robust,
    )
    model = IntensityResidualRanker(
        features.shape[1],
        float(cfg["maximum_intensity_weight"]),
        float(cfg["initial_weight_logit"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    best_loss, best_epoch, stale, history = float("inf"), 0, 0, []
    for epoch in range(1, int(cfg["epochs"]) + 1):
        model.train()
        order = np.random.permutation(len(fit))
        train_losses = []
        for start in range(0, len(order), int(cfg["batch_size"])):
            local = order[start : start + int(cfg["batch_size"])]
            batch = fit[local]
            loss, parts = listwise_loss(
                model,
                torch.from_numpy(arrays["v1"][batch]).to(device),
                torch.from_numpy(arrays["intensity"][batch]).to(device),
                torch.from_numpy(features[batch]).to(device),
                torch.from_numpy(attenuation[batch]).to(device),
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
            train_losses.append(parts)
        validation_loss = evaluate_loss(
            model,
            arrays,
            features,
            attenuation,
            validation,
            records,
            progress,
            robust,
            cfg,
            device,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(
                    np.mean([item["total"] for item in train_losses])
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
                    "metadata": {
                        **metadata,
                        "feature_names": feature_names(),
                        "feature_mean": mean,
                        "feature_std": std,
                        "predicted_ttc_groups": thresholds,
                    },
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


def predict_ranker(
    model: IntensityResidualRanker,
    arrays: dict[str, np.ndarray],
    raw_features: np.ndarray,
    attenuation: np.ndarray,
    indices: np.ndarray,
    device: torch.device,
    checkpoint_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    metadata = checkpoint["metadata"]
    features = (
        (raw_features - metadata["feature_mean"]) / metadata["feature_std"]
    ).astype(np.float32)
    score_values, raw_values, effective_values = [], [], []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(indices), 256):
            batch = indices[start : start + 256]
            scores, raw_weight, effective_weight = model(
                torch.from_numpy(arrays["v1"][batch]).to(device),
                torch.from_numpy(arrays["intensity"][batch]).to(device),
                torch.from_numpy(features[batch]).to(device),
                torch.from_numpy(attenuation[batch]).to(device),
            )
            score_values.append(scores.cpu().numpy())
            raw_values.append(raw_weight.cpu().numpy())
            effective_values.append(effective_weight.cpu().numpy())
    return (
        np.concatenate(score_values).astype(np.float32),
        np.concatenate(raw_values).astype(np.float32),
        np.concatenate(effective_values).astype(np.float32),
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
    query_rows = read_csv_rows(project_path(cfg["v1_query_csv"]))
    query_by_name = {row["query_image_name"]: row for row in query_rows}
    if len(query_by_name) != len(query_rows) or set(query_by_name) != set(
        source_by_name
    ):
        raise RuntimeError("Phase4I.2 V1 query rows do not match development")
    query_names = [row["query_image_name"] for row in query_rows]
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
        raise RuntimeError(
            "Phase4I.2 factor candidate queries do not match V1 queries"
        )
    candidate_fingerprint = candidate_set_fingerprint(factor_groups)
    factor_groups = grouped(
        [row for group in factor_groups.values() for row in group]
    )
    for name in query_names:
        if any(
            row["candidate_record_id"] == row["query_record_id"]
            for row in factor_groups[name]
        ):
            raise RuntimeError(f"Phase4I.2 same-record candidate for {name}")
        if any(
            row["oof_fold"] != query_by_name[name]["oof_fold"]
            for row in factor_groups[name]
        ):
            raise RuntimeError(f"Phase4I.2 fold mismatch for {name}")

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
        "intensity": query_standardize(
            matrix("predicted_intensity_distance_dino_motion")
        ),
        "target": matrix("candidate_tactile_embedding_distance"),
    }
    progress = online_progress_features(factor_groups, query_names)
    raw_features = query_features(
        arrays["v1"],
        arrays["intensity"],
        progress,
    )
    plain_attenuation = np.ones(len(query_rows), dtype=np.float32)
    robust_attenuation = uncertainty_attenuation(progress, cfg)
    attenuations = {
        "plain_intensity": plain_attenuation,
        "ttc_robust_intensity": robust_attenuation,
    }
    v1_reference = consistent_reference_rows(
        query_rows,
        factor_groups,
        arrays["v1"],
    )
    records = np.asarray([row["query_record_id"] for row in query_rows])
    folds = sorted({row["oof_fold"] for row in query_rows})
    if len(folds) != 3:
        raise RuntimeError(f"Phase4I.2 requires three OOF folds, got {folds}")

    scores_by_variant = {
        variant: np.zeros((len(query_rows), top_k), dtype=np.float32)
        for variant in VARIANTS
    }
    raw_weight_by_variant = {
        variant: np.zeros(len(query_rows), dtype=np.float32)
        for variant in VARIANTS
    }
    effective_weight_by_variant = {
        variant: np.zeros(len(query_rows), dtype=np.float32)
        for variant in VARIANTS
    }
    training_reports = []
    checkpoint_root = project_path(cfg["checkpoint_dir"])
    ensure_dir(checkpoint_root)
    seeds = [int(value) for value in cfg["seeds"]]
    for variant in VARIANTS:
        robust = variant == "ttc_robust_intensity"
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
                [
                    index
                    for index in outer_fit
                    if index not in validation_set
                ],
                dtype=np.int32,
            )
            if not len(inner_fit) or not len(inner_validation):
                raise RuntimeError(
                    f"Phase4I.2 {variant} fold {fold} inner split is empty"
                )
            seed_scores, seed_raw, seed_effective = [], [], []
            for seed in seeds:
                print(
                    f"phase4i2 {variant} fold {fold} seed {seed}: training",
                    flush=True,
                )
                set_seed(seed)
                checkpoint_path = (
                    checkpoint_root / variant / f"fold_{fold}_seed_{seed}.pt"
                )
                model, report = train_ranker(
                    arrays,
                    raw_features,
                    progress,
                    attenuations[variant],
                    inner_fit,
                    inner_validation,
                    records,
                    robust,
                    cfg,
                    device,
                    checkpoint_path,
                    {
                        "scope": "strict_oof_intensity_only_residual",
                        "variant": variant,
                        "fold": fold,
                        "seed": seed,
                        "candidate_fingerprint": candidate_fingerprint,
                        "query_true_probe_used": False,
                        "query_tactile_input": False,
                    },
                )
                scores, raw_weight, effective_weight = predict_ranker(
                    model,
                    arrays,
                    raw_features,
                    attenuations[variant],
                    held_out,
                    device,
                    checkpoint_path,
                )
                seed_scores.append(scores)
                seed_raw.append(raw_weight)
                seed_effective.append(effective_weight)
                training_reports.append(
                    {
                        "variant": variant,
                        "fold": fold,
                        "seed": seed,
                        **report,
                    }
                )
                print(
                    f"phase4i2 {variant} fold {fold} seed {seed}: "
                    f"best_epoch={report['best_epoch']} "
                    f"validation_loss={report['best_validation_loss']:.6f}",
                    flush=True,
                )
            scores_by_variant[variant][held_out] = np.mean(
                seed_scores,
                axis=0,
            )
            raw_weight_by_variant[variant][held_out] = np.mean(
                seed_raw,
                axis=0,
            )
            effective_weight_by_variant[variant][held_out] = np.mean(
                seed_effective,
                axis=0,
            )

    touch_cache: dict[str, np.ndarray] = {}
    rows_by_variant = {}
    comparison_cfg = {
        "bootstrap_iterations": int(cfg["bootstrap_iterations"]),
        "bootstrap_seed": int(cfg["bootstrap_seed"]),
    }
    comparisons = {}
    for variant in VARIANTS:
        two_weights = np.stack(
            (
                np.zeros(len(query_rows), dtype=np.float32),
                effective_weight_by_variant[variant],
            ),
            axis=1,
        )
        rows_by_variant[variant] = build_cascade_query_rows(
            v1_reference,
            factor_groups,
            scores_by_variant[variant].argmin(axis=1),
            scores_by_variant[variant],
            two_weights,
            source_by_name,
            cfg,
            touch_cache,
            variant,
        )
        comparisons[variant] = fast_bootstrap_comparison(
            v1_reference,
            rows_by_variant[variant],
            comparison_cfg,
        )

    accepted = [
        variant
        for variant in (
            "ttc_robust_intensity",
            "plain_intensity",
        )
        if comparisons[variant]["accepted"]
    ]
    selected_variant = accepted[0] if accepted else None
    deployment_accepted = selected_variant is not None
    final_rows = (
        rows_by_variant[selected_variant]
        if selected_variant is not None
        else v1_reference
    )

    query_output = []
    for index, (v1, plain, robust, final) in enumerate(
        zip(
            v1_reference,
            rows_by_variant["plain_intensity"],
            rows_by_variant["ttc_robust_intensity"],
            final_rows,
            strict=True,
        )
    ):
        query_output.append(
            {
                "query_record_id": v1["query_record_id"],
                "query_image_name": v1["query_image_name"],
                "query_probe": v1["query_probe"],
                "oof_fold": v1["oof_fold"],
                "v1_ranker_oracle_embedding_rank": v1[
                    "ranker_oracle_embedding_rank"
                ],
                "v1_selected_cache_image_name": v1[
                    "selected_cache_image_name"
                ],
                **{f"v1_{metric}": v1[metric] for metric in METRICS},
                "plain_weight": (
                    f"{effective_weight_by_variant['plain_intensity'][index]:.9f}"
                ),
                "plain_selected_cache_image_name": plain[
                    "selected_cache_image_name"
                ],
                "plain_ranker_oracle_embedding_rank": plain[
                    "ranker_oracle_embedding_rank"
                ],
                **{f"plain_{metric}": plain[metric] for metric in METRICS},
                "robust_raw_weight": (
                    f"{raw_weight_by_variant['ttc_robust_intensity'][index]:.9f}"
                ),
                "robust_attenuation": f"{robust_attenuation[index]:.9f}",
                "robust_effective_weight": (
                    f"{effective_weight_by_variant['ttc_robust_intensity'][index]:.9f}"
                ),
                "robust_selected_cache_image_name": robust[
                    "selected_cache_image_name"
                ],
                "robust_ranker_oracle_embedding_rank": robust[
                    "ranker_oracle_embedding_rank"
                ],
                **{f"robust_{metric}": robust[metric] for metric in METRICS},
                **{
                    field: f"{progress[index, field_index]:.9f}"
                    for field_index, field in enumerate(ONLINE_PROGRESS_FIELDS)
                },
                "deployment_accepted": str(int(deployment_accepted)),
                "final_selection_source": selected_variant or "v1",
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
        group = factor_groups[query["query_image_name"]]
        plain_rank = rank_positions(
            scores_by_variant["plain_intensity"][query_index]
        )
        robust_rank = rank_positions(
            scores_by_variant["ttc_robust_intensity"][query_index]
        )
        for candidate_index, row in enumerate(group):
            if selected_variant == "plain_intensity":
                final_rank = plain_rank[candidate_index]
            elif selected_variant == "ttc_robust_intensity":
                final_rank = robust_rank[candidate_index]
            else:
                final_rank = int(row["candidate_rank"])
            candidate_output.append(
                {
                    "query_record_id": query["query_record_id"],
                    "query_image_name": query["query_image_name"],
                    "query_probe": query["query_probe"],
                    "oof_fold": query["oof_fold"],
                    "candidate_rank": row["candidate_rank"],
                    "plain_candidate_rank": str(
                        int(plain_rank[candidate_index])
                    ),
                    "robust_candidate_rank": str(
                        int(robust_rank[candidate_index])
                    ),
                    "final_candidate_rank": str(int(final_rank)),
                    "v1_score": row["v1_score"],
                    "predicted_intensity_distance": row[
                        "predicted_intensity_distance_dino_motion"
                    ],
                    "plain_score": (
                        f"{scores_by_variant['plain_intensity'][query_index, candidate_index]:.9f}"
                    ),
                    "robust_score": (
                        f"{scores_by_variant['ttc_robust_intensity'][query_index, candidate_index]:.9f}"
                    ),
                    "plain_weight": (
                        f"{effective_weight_by_variant['plain_intensity'][query_index]:.9f}"
                    ),
                    "robust_raw_weight": (
                        f"{raw_weight_by_variant['ttc_robust_intensity'][query_index]:.9f}"
                    ),
                    "robust_attenuation": (
                        f"{robust_attenuation[query_index]:.9f}"
                    ),
                    "robust_effective_weight": (
                        f"{effective_weight_by_variant['ttc_robust_intensity'][query_index]:.9f}"
                    ),
                    "candidate_record_id": row["candidate_record_id"],
                    "candidate_image_name": row["candidate_image_name"],
                    "candidate_tactile_embedding_distance": row[
                        "candidate_tactile_embedding_distance"
                    ],
                    "candidate_oracle_embedding_rank": row[
                        "candidate_oracle_embedding_rank"
                    ],
                    "deployment_accepted": str(int(deployment_accepted)),
                    "final_selection_source": selected_variant or "v1",
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
        "mode": "phase4i2_ttc_robust_intensity_only_residual_oof_v1",
        "device": str(device),
        "variants": {
            variant: {
                "summary": {
                    "all": metric_summary(rows_by_variant[variant]),
                    "far_probe75_100": metric_summary(
                        rows_by_variant[variant],
                        lambda row: int(row["query_probe"]) >= 75,
                    ),
                },
                "vs_v1": comparisons[variant],
                "mean_raw_weight": float(
                    raw_weight_by_variant[variant].mean()
                ),
                "mean_effective_weight": float(
                    effective_weight_by_variant[variant].mean()
                ),
            }
            for variant in VARIANTS
        },
        "selection_rule": (
            "prefer accepted ttc_robust_intensity; otherwise accepted "
            "plain_intensity; otherwise V1"
        ),
        "deployment_accepted": deployment_accepted,
        "selected_variant": selected_variant,
        "deployed": {
            "source": selected_variant or "v1",
            "summary": {
                "all": metric_summary(final_rows),
                "far_probe75_100": metric_summary(
                    final_rows,
                    lambda row: int(row["query_probe"]) >= 75,
                ),
            },
        },
        "robustness": {
            "group_definition": (
                "predicted TTC only; true probe is evaluation-only"
            ),
            "predicted_ttc_thresholds": [
                float(value) for value in cfg["predicted_ttc_groups"]
            ],
            "mean_attenuation": float(robust_attenuation.mean()),
            "minimum_attenuation": float(robust_attenuation.min()),
            "maximum_attenuation": float(robust_attenuation.max()),
        },
        "training": training_reports,
        "integrity": {
            "source": "strict development 3-fold record-level OOF only",
            "c2_contact_box": "unchanged",
            "top32_candidates": "frozen V1 identity",
            "candidate_fingerprint": candidate_fingerprint,
            "same_record_candidates": 0,
            "direct_dino_residual_used": False,
            "dino_used_inside_frozen_intensity_predictor": True,
            "query_true_probe_used": False,
            "query_tactile_input": False,
            "query_tactile_usage": (
                "offline listwise labels and evaluation only"
            ),
            "sealed_final_holdout_rows_read": 0,
            "development_validation_rows_read": 0,
            "sample_level_gate_used": False,
        },
        "next_action": (
            "freeze selected intensity-only recipe and run independent development validation"
            if deployment_accepted
            else "retain V1; do not run development validation or final holdout"
        ),
    }
    write_json(project_path(cfg["metrics_json"]), report)
    print(
        {
            "mode": report["mode"],
            "variants": report["variants"],
            "selection_rule": report["selection_rule"],
            "deployment_accepted": report["deployment_accepted"],
            "selected_variant": report["selected_variant"],
            "deployed": report["deployed"],
            "robustness": report["robustness"],
            "integrity": report["integrity"],
            "next_action": report["next_action"],
        }
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train strict-OOF Phase4I.2 TTC-robust intensity residual."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--section",
        default="phase4i2_ttc_robust_intensity_residual_oof_v1",
    )
    args = parser.parse_args()
    train(args.config, args.section)


if __name__ == "__main__":
    main()
