"""Strict-OOF Phase4H.2 shape/intensity predictability and cascade diagnostic."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .build_phase4g_dino_v1_fusion import bootstrap_comparison
from .config import load_config, project_path
from .evaluate_oracle_tactile_retrieval import tactile_difference, tactile_metrics
from .phase4h_dino_adaptation import (
    assert_candidate_identity,
    assert_development_only,
    candidate_groups,
    deployable_motion_feature,
    record_hash_split,
)
from .phase4h_factorized_tactile import (
    INTENSITY_DIM,
    INTENSITY_FIELDS,
    SHAPE_DIM,
    LowCapacityIntensityRegressor,
    factorized_tactile_latents,
    standardized_distance,
)
from .temporal_progress import (
    DEFAULT_TTC_VALUES,
    masked_trajectory_features,
    read_trajectory_tracks,
)
from .train_phase4b_predicted_box_cache_ranker import prediction_map, set_seed
from .train_phase4h_dino_tactile_alignment import (
    load_feature_cache,
    load_frontier,
    standardize,
    summarize,
)
from .utils import ensure_dir, read_csv_rows, write_csv_rows, write_json


QUERY_FIELDS = [
    "strategy",
    "predictor",
    "shortlist_k",
    "query_record_id",
    "query_image_name",
    "query_probe",
    "oof_fold",
    "selected_cache_record_id",
    "selected_cache_image_name",
    "ranker_oracle_embedding_rank",
    "shape_distance",
    "true_intensity_distance",
    "predicted_intensity_distance",
    "query_intensity_prediction_mae",
    "global_median_intensity_mae",
    "tactile_diff_mae",
    "tactile_ssim",
    "tactile_mask_iou",
]


def factor_index_fingerprint(
    rows: list[dict[str, str]],
    tactile_size: int,
    threshold: float,
) -> str:
    digest = hashlib.sha256()
    digest.update(f"size={tactile_size}|threshold={threshold:.9f}\n".encode())
    for row in rows:
        digest.update(
            f"{row['split']}|{row['record_id']}|{row['image_name']}|{row['touch_path']}\n".encode(
                "utf-8"
            )
        )
    return digest.hexdigest()


def build_or_load_factor_index(
    rows: list[dict[str, str]],
    cfg: dict,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    output_path = project_path(cfg["factor_index_npz"])
    fingerprint = factor_index_fingerprint(
        rows,
        int(cfg["tactile_size"]),
        float(cfg["tactile_mask_threshold"]),
    )
    if output_path.is_file():
        data = np.load(output_path, allow_pickle=False)
        if (
            str(data["fingerprint"][0]) != fingerprint
            or list(data["image_names"].astype(str))
            != [row["image_name"] for row in rows]
        ):
            raise RuntimeError("Existing Phase4H.2 factor index fingerprint mismatch")
        print(f"phase4h.2: reusing {output_path}", flush=True)
        return (
            data["shape_latents"].astype(np.float32),
            data["intensity_latents"].astype(np.float32),
            {},
        )
    touch_cache: dict[str, np.ndarray] = {}
    shape_values, intensity_values = [], []
    for index, row in enumerate(rows, start=1):
        diff = tactile_difference(
            row["touch_path"],
            touch_cache,
            int(cfg["tactile_size"]),
        )
        shape, intensity = factorized_tactile_latents(
            diff,
            float(cfg["tactile_mask_threshold"]),
        )
        shape_values.append(shape)
        intensity_values.append(intensity)
        if index % 200 == 0 or index == len(rows):
            print(f"phase4h.2 tactile factor index: {index}/{len(rows)}", flush=True)
    shape = np.stack(shape_values).astype(np.float32)
    intensity = np.stack(intensity_values).astype(np.float32)
    ensure_dir(output_path.parent)
    np.savez_compressed(
        output_path,
        image_names=np.asarray([row["image_name"] for row in rows]),
        shape_latents=shape,
        intensity_latents=intensity,
        intensity_fields=np.asarray(INTENSITY_FIELDS),
        fingerprint=np.asarray([fingerprint]),
    )
    return shape, intensity, touch_cache


def record_balanced_weights(
    indices: np.ndarray,
    rows: list[dict[str, str]],
) -> np.ndarray:
    counts: dict[str, int] = defaultdict(int)
    for index in indices:
        counts[rows[int(index)]["record_id"]] += 1
    values = np.asarray(
        [1.0 / counts[rows[int(index)]["record_id"]] for index in indices],
        dtype=np.float32,
    )
    return values / max(float(values.mean()), 1e-8)


def weighted_loss(
    model: LowCapacityIntensityRegressor,
    features: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    prediction = model(features)
    per_query = F.smooth_l1_loss(prediction, target, reduction="none").mean(dim=1)
    return (per_query * weights).sum() / weights.sum().clamp_min(1e-8)


def evaluate_model_loss(
    model: LowCapacityIntensityRegressor,
    features: np.ndarray,
    targets: np.ndarray,
    indices: np.ndarray,
    weights: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> float:
    model.eval()
    total, denominator = 0.0, 0.0
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch = indices[start : start + batch_size]
            batch_weights = weights[start : start + len(batch)]
            prediction = model(torch.from_numpy(features[batch]).to(device))
            loss = F.smooth_l1_loss(
                prediction,
                torch.from_numpy(targets[batch]).to(device),
                reduction="none",
            ).mean(dim=1)
            total += float(
                (loss * torch.from_numpy(batch_weights).to(device)).sum().cpu()
            )
            denominator += float(batch_weights.sum())
    return total / max(denominator, 1e-8)


def train_regressor(
    features: np.ndarray,
    targets: np.ndarray,
    fit_indices: np.ndarray,
    validation_indices: np.ndarray,
    rows: list[dict[str, str]],
    cfg: dict,
    device: torch.device,
    checkpoint_path: Path,
    metadata: dict,
) -> tuple[LowCapacityIntensityRegressor, dict]:
    model = LowCapacityIntensityRegressor(
        features.shape[1],
        int(cfg["hidden_dim"]),
        INTENSITY_DIM,
        float(cfg["dropout"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    fit_weights = record_balanced_weights(fit_indices, rows)
    validation_weights = record_balanced_weights(validation_indices, rows)
    best_loss, best_epoch, stale, history = float("inf"), 0, 0, []
    batch_size = int(cfg["batch_size"])
    for epoch in range(1, int(cfg["epochs"]) + 1):
        model.train()
        permutation = np.random.permutation(len(fit_indices))
        train_total, train_weight = 0.0, 0.0
        for start in range(0, len(permutation), batch_size):
            local = permutation[start : start + batch_size]
            batch = fit_indices[local]
            weights = fit_weights[local]
            loss = weighted_loss(
                model,
                torch.from_numpy(features[batch]).to(device),
                torch.from_numpy(targets[batch]).to(device),
                torch.from_numpy(weights).to(device),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(),
                float(cfg["gradient_clip"]),
            )
            optimizer.step()
            train_total += float(loss.detach().cpu()) * float(weights.sum())
            train_weight += float(weights.sum())
        validation_loss = evaluate_model_loss(
            model,
            features,
            targets,
            validation_indices,
            validation_weights,
            batch_size,
            device,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_total / max(train_weight, 1e-8),
                "validation_loss": validation_loss,
            }
        )
        if validation_loss < best_loss - 1e-6:
            best_loss, best_epoch, stale = validation_loss, epoch, 0
            ensure_dir(checkpoint_path.parent)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "input_dim": features.shape[1],
                    "metadata": metadata,
                },
                checkpoint_path,
            )
        else:
            stale += 1
        if stale >= int(cfg["early_stopping_patience"]):
            break
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    return model, {
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "epochs_ran": len(history),
        "history": history,
    }


def predict_regressor(
    model: LowCapacityIntensityRegressor,
    features: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    output = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch = indices[start : start + batch_size]
            output.append(
                model(torch.from_numpy(features[batch]).to(device))
                .cpu()
                .numpy()
            )
    return np.concatenate(output).astype(np.float32)


def build_online_motion(
    rows: list[dict[str, str]],
    predictions: dict[str, dict[str, str]],
    cfg: dict,
) -> np.ndarray:
    tracks = read_trajectory_tracks(project_path(cfg["motion_tracks_csv"]))
    ttc_by_name = {}
    ttc_value = str(cfg.get("ttc_predictions_csv", "")).strip()
    ttc_path = project_path(ttc_value) if ttc_value else None
    if ttc_path is not None and ttc_path.is_file():
        ttc_by_name = {
            row["image_name"]: row
            for row in read_csv_rows(ttc_path)
            if row["image_name"] in predictions
        }
    features = []
    for row in rows:
        trajectory, mask, quality = masked_trajectory_features(
            row,
            tracks,
            int(cfg["trajectory_history_frames"]),
            float(cfg["trajectory_spatial_scale_px"]),
            float(cfg["trajectory_speed_scale_px"]),
        )
        prediction = predictions[row["image_name"]]
        features.append(
            deployable_motion_feature(
                row,
                float(prediction["pred_x"]),
                float(prediction["pred_y"]),
                trajectory,
                mask,
                quality,
                ttc_by_name.get(row["image_name"], prediction),
                cfg.get("ttc_values", DEFAULT_TTC_VALUES),
            )
        )
    return np.stack(features).astype(np.float32)


def normalized_features(
    raw: np.ndarray,
    fit_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = raw[fit_indices].mean(axis=0)
    std = raw[fit_indices].std(axis=0)
    std[std < 1e-6] = 1.0
    return standardize(raw, mean, std).astype(np.float32), mean, std


def select_from_shortlist(
    distances: np.ndarray,
    dino_ranks: np.ndarray,
    top_k: int,
) -> np.ndarray:
    masked = np.where(dino_ranks <= top_k, distances, np.inf)
    if not np.isfinite(masked).any(axis=1).all():
        raise RuntimeError(f"DINO Top-{top_k} shortlist is empty for a query")
    return masked.argmin(axis=1).astype(np.int32)


def prediction_metrics(
    predicted: np.ndarray,
    target: np.ndarray,
) -> dict[str, float]:
    residual = predicted - target
    denominator = float(np.square(target - target.mean(axis=0)).sum())
    return {
        "latent_mae": float(np.abs(residual).mean()),
        "latent_mse": float(np.square(residual).mean()),
        "latent_r2": float(
            1.0 - np.square(residual).sum() / max(denominator, 1e-8)
        ),
    }


def bootstrap_prediction_error(
    rows: list[dict[str, str]],
    prediction_error: np.ndarray,
    baseline_error: np.ndarray,
    iterations: int,
    seed: int,
) -> dict:
    names_by_record: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        names_by_record[row["record_id"]].append(index)
    records = sorted(names_by_record)
    rng = np.random.default_rng(seed)
    output = {}
    for regime, selected in (
        ("all", np.arange(len(rows), dtype=np.int32)),
        (
            "far_probe75_100",
            np.asarray(
                [index for index, row in enumerate(rows) if int(row["probe"]) >= 75],
                dtype=np.int32,
            ),
        ),
    ):
        selected_set = set(selected.tolist())
        eligible_records = [
            record
            for record in records
            if any(index in selected_set for index in names_by_record[record])
        ]
        if not eligible_records:
            raise RuntimeError(f"No records are available for prediction regime {regime}")
        deltas = prediction_error[selected] - baseline_error[selected]
        samples = []
        for _ in range(iterations):
            draw = rng.choice(
                eligible_records,
                len(eligible_records),
                replace=True,
            )
            indices = [
                index
                for record in draw
                for index in names_by_record[record]
                if index in selected_set
            ]
            samples.append(
                float(
                    np.mean(prediction_error[indices] - baseline_error[indices])
                )
            )
        output[regime] = {
            "queries": len(selected),
            "mean_error_delta_vs_global_median": float(deltas.mean()),
            "bootstrap_95_ci": [
                float(np.quantile(samples, 0.025)),
                float(np.quantile(samples, 0.975)),
            ],
        }
    output["accepted"] = bool(
        output["all"]["mean_error_delta_vs_global_median"] < 0
        and output["all"]["bootstrap_95_ci"][1] < 0
        and output["far_probe75_100"]["mean_error_delta_vs_global_median"] < 0
        and output["far_probe75_100"]["bootstrap_95_ci"][1] < 0
    )
    return output


def choices_from_distance(distances: np.ndarray) -> np.ndarray:
    return distances.argmin(axis=1).astype(np.int32)


def build_query_output(
    strategy: str,
    predictor: str,
    shortlist_k: int | None,
    choices: np.ndarray,
    rows: list[dict[str, str]],
    candidates: np.ndarray,
    oracle_embedding_ranks: np.ndarray,
    shape_distances: dict[str, np.ndarray],
    intensity_distances: dict[str, np.ndarray],
    predicted_distances: dict[str, np.ndarray],
    prediction_errors: dict[str, np.ndarray],
    global_errors: np.ndarray,
    fold_by_name: dict[str, str],
    touch_cache: dict[str, np.ndarray],
    metric_cache: dict[tuple[int, int], dict[str, float]],
    cfg: dict,
) -> list[dict[str, str]]:
    output = []
    for index, choice in enumerate(choices):
        candidate_index = int(candidates[index, int(choice)])
        selected = rows[candidate_index]
        pair_key = (index, candidate_index)
        if pair_key not in metric_cache:
            query_touch = tactile_difference(
                rows[index]["touch_path"],
                touch_cache,
                int(cfg["tactile_size"]),
            )
            candidate_touch = tactile_difference(
                selected["touch_path"],
                touch_cache,
                int(cfg["tactile_size"]),
            )
            metric_cache[pair_key] = tactile_metrics(
                query_touch,
                candidate_touch,
                float(cfg["tactile_mask_threshold"]),
            )
        tactile = metric_cache[pair_key]
        shape_key = fold_by_name[rows[index]["image_name"]]
        output.append(
            {
                "strategy": strategy,
                "predictor": predictor,
                "shortlist_k": "" if shortlist_k is None else str(shortlist_k),
                "query_record_id": rows[index]["record_id"],
                "query_image_name": rows[index]["image_name"],
                "query_probe": rows[index]["probe"],
                "oof_fold": shape_key,
                "selected_cache_record_id": selected["record_id"],
                "selected_cache_image_name": selected["image_name"],
                "ranker_oracle_embedding_rank": str(
                    int(oracle_embedding_ranks[index, int(choice)])
                ),
                "shape_distance": f"{shape_distances[shape_key][index, int(choice)]:.9f}",
                "true_intensity_distance": f"{intensity_distances[shape_key][index, int(choice)]:.9f}",
                "predicted_intensity_distance": (
                    ""
                    if predictor not in predicted_distances
                    else f"{predicted_distances[predictor][index, int(choice)]:.9f}"
                ),
                "query_intensity_prediction_mae": (
                    ""
                    if predictor not in prediction_errors
                    else f"{prediction_errors[predictor][index]:.9f}"
                ),
                "global_median_intensity_mae": f"{global_errors[index]:.9f}",
                **{
                    key: f"{tactile[key]:.9f}"
                    for key in (
                        "tactile_diff_mae",
                        "tactile_ssim",
                        "tactile_mask_iou",
                    )
                },
            }
        )
    return output


def evaluate(config_path: str, section: str) -> dict:
    cfg = load_config(config_path)[section]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with project_path(cfg["oof_evaluation_json"]).open("r", encoding="utf-8") as handle:
        phase4h_oof = json.load(handle)
    if phase4h_oof.get("ready_for_development_validation", False):
        raise RuntimeError(
            "Phase4H.2 is a failed-OOF diagnostic and must not replace an accepted Phase4H run"
        )
    samples = read_csv_rows(project_path(cfg["samples_csv"]))
    assert_development_only(samples, project_path(cfg["final_partition_csv"]))
    rows = [row for row in samples if row["dataset_split"] == "train"]
    names = [row["image_name"] for row in rows]
    row_index = {name: index for index, name in enumerate(names)}
    predictions = prediction_map(
        read_csv_rows(project_path(cfg["oof_predictions_csv"])),
        rows,
        "train",
        "Phase4H.2 factorized intensity OOF",
    )
    fold_by_name = {
        name: prediction["oof_fold"] for name, prediction in predictions.items()
    }
    folds = sorted(set(fold_by_name.values()))
    if len(folds) != 3:
        raise RuntimeError(f"Phase4H.2 requires exactly three OOF folds, got {folds}")

    recipe_name, feature_path = load_frontier(project_path(cfg["frontier_json"]))
    visual, _ = load_feature_cache(feature_path, names)
    motion = build_online_motion(rows, predictions, cfg)
    shape_raw, intensity_raw, touch_cache = build_or_load_factor_index(rows, cfg)
    if shape_raw.shape != (len(rows), SHAPE_DIM):
        raise RuntimeError(f"Unexpected shape latent matrix: {shape_raw.shape}")
    if intensity_raw.shape != (len(rows), INTENSITY_DIM):
        raise RuntimeError(f"Unexpected intensity latent matrix: {intensity_raw.shape}")

    top_k = int(cfg["geometry_filter_k"])
    v1_groups = candidate_groups(
        read_csv_rows(project_path(cfg["v1_candidate_csv"])), top_k
    )
    if set(v1_groups) != set(names):
        raise RuntimeError("Phase4H.2 V1 candidate queries do not match development OOF")
    for group in v1_groups.values():
        group.sort(key=lambda row: int(row["candidate_rank"]))
    if any(
        item["candidate_record_id"] == item["query_record_id"]
        for group in v1_groups.values()
        for item in group
    ):
        raise RuntimeError("Phase4H.2 found a same-record cache candidate")
    recipe_groups = candidate_groups(
        [
            row
            for row in read_csv_rows(project_path(cfg["ablation_candidate_csv"]))
            if row["recipe_name"] == recipe_name
        ],
        top_k,
    )
    assert_candidate_identity(v1_groups, recipe_groups)
    recipe_map = {
        (row["query_image_name"], row["candidate_image_name"]): row
        for group in recipe_groups.values()
        for row in group
    }
    candidates = np.stack(
        [
            np.asarray(
                [row_index[item["candidate_image_name"]] for item in v1_groups[name]],
                dtype=np.int32,
            )
            for name in names
        ]
    )
    v1_scores = np.asarray(
        [
            [float(item["candidate_score"]) for item in v1_groups[name]]
            for name in names
        ],
        dtype=np.float32,
    )
    v1_reference_rows = read_csv_rows(project_path(cfg["v1_query_csv"]))
    v1_reference_by_name = {
        row["query_image_name"]: row for row in v1_reference_rows
    }
    if len(v1_reference_by_name) != len(v1_reference_rows) or set(
        v1_reference_by_name
    ) != set(names):
        raise RuntimeError("Phase4H.2 V1 query baseline does not match development OOF")
    v1_choices = choices_from_distance(v1_scores)
    for index, choice in enumerate(v1_choices):
        selected_name = v1_groups[names[index]][int(choice)]["candidate_image_name"]
        if (
            selected_name
            != v1_reference_by_name[names[index]]["selected_cache_image_name"]
        ):
            raise RuntimeError(
                f"Phase4H.2 V1 Top-1 candidate mismatch for {names[index]}"
            )
    dino_ranks = np.asarray(
        [
            [
                int(recipe_map[(name, item["candidate_image_name"])]["candidate_rank"])
                for item in v1_groups[name]
            ]
            for name in names
        ],
        dtype=np.int32,
    )
    oracle_embedding_ranks = np.asarray(
        [
            [
                int(
                    recipe_map[(name, item["candidate_image_name"])][
                        "candidate_oracle_embedding_rank"
                    ]
                )
                for item in v1_groups[name]
            ]
            for name in names
        ],
        dtype=np.int32,
    )

    predictors = ("global_median", "motion_only", "dino_only", "dino_motion")
    learned_predictors = predictors[1:]
    predicted = {
        predictor: np.zeros((len(rows), INTENSITY_DIM), dtype=np.float32)
        for predictor in predictors
    }
    standardized_targets = np.zeros_like(intensity_raw)
    shape_distances: dict[str, np.ndarray] = {}
    intensity_distances: dict[str, np.ndarray] = {}
    training_reports = []
    checkpoint_dir = project_path(cfg["checkpoint_dir"])
    ensure_dir(checkpoint_dir)
    seeds = [int(value) for value in cfg["seeds"]]

    for fold in folds:
        print(f"phase4h.2: starting outer fold {fold}", flush=True)
        held_out = np.asarray(
            [
                index
                for index, name in enumerate(names)
                if fold_by_name[name] == fold
            ],
            dtype=np.int32,
        )
        outer_fit = np.asarray(
            [
                index
                for index, name in enumerate(names)
                if fold_by_name[name] != fold
            ],
            dtype=np.int32,
        )
        inner_validation = np.asarray(
            [
                index
                for index in outer_fit
                if record_hash_split(
                    rows[int(index)]["record_id"],
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
            raise RuntimeError(f"Fold {fold} has an empty inner record split")

        shape_mean = shape_raw[outer_fit].mean(axis=0)
        shape_std = shape_raw[outer_fit].std(axis=0)
        shape_std[shape_std < 1e-6] = 1.0
        intensity_mean = intensity_raw[outer_fit].mean(axis=0)
        intensity_std = intensity_raw[outer_fit].std(axis=0)
        intensity_std[intensity_std < 1e-6] = 1.0
        shape_standardized = standardize(shape_raw, shape_mean, shape_std).astype(
            np.float32
        )
        intensity_standardized = standardize(
            intensity_raw,
            intensity_mean,
            intensity_std,
        ).astype(np.float32)
        standardized_targets[held_out] = intensity_standardized[held_out]
        shape_distances[fold] = standardized_distance(
            shape_standardized,
            shape_standardized[candidates],
        )
        intensity_distances[fold] = standardized_distance(
            intensity_standardized,
            intensity_standardized[candidates],
        )
        median_raw = np.median(intensity_raw[outer_fit], axis=0)
        predicted["global_median"][held_out] = standardize(
            median_raw[None],
            intensity_mean,
            intensity_std,
        )[0]

        feature_sources = {
            "motion_only": motion,
            "dino_only": visual,
            "dino_motion": np.concatenate((visual, motion), axis=1).astype(
                np.float32
            ),
        }
        for predictor in learned_predictors:
            features, feature_mean, feature_std = normalized_features(
                feature_sources[predictor],
                inner_fit,
            )
            seed_predictions = []
            for seed in seeds:
                set_seed(seed)
                model, report = train_regressor(
                    features,
                    intensity_standardized,
                    inner_fit,
                    inner_validation,
                    rows,
                    cfg,
                    device,
                    checkpoint_dir / f"{predictor}_fold_{fold}_seed_{seed}.pt",
                    {
                        "scope": "strict_oof_factorized_intensity",
                        "predictor": predictor,
                        "fold": fold,
                        "seed": seed,
                        "primary_recipe": recipe_name,
                        "feature_mean": feature_mean,
                        "feature_std": feature_std,
                        "intensity_mean": intensity_mean,
                        "intensity_std": intensity_std,
                        "query_true_probe_used": False,
                        "query_tactile_input": False,
                    },
                )
                seed_predictions.append(
                    predict_regressor(
                        model,
                        features,
                        held_out,
                        int(cfg["batch_size"]),
                        device,
                    )
                )
                training_reports.append(
                    {
                        "predictor": predictor,
                        "fold": fold,
                        "seed": seed,
                        **report,
                    }
                )
                print(
                    "phase4h.2 "
                    f"predictor={predictor} fold={fold} seed={seed} "
                    f"best_epoch={report['best_epoch']} "
                    f"best_val={report['best_validation_loss']:.6f}",
                    flush=True,
                )
            predicted[predictor][held_out] = np.mean(
                seed_predictions,
                axis=0,
            )

    prediction_errors = {
        name: np.abs(values - standardized_targets).mean(axis=1)
        for name, values in predicted.items()
    }
    predicted_distances = {}
    for predictor in predictors:
        matrix = np.zeros((len(rows), top_k), dtype=np.float32)
        for fold in folds:
            held_out = np.asarray(
                [
                    index
                    for index, name in enumerate(names)
                    if fold_by_name[name] == fold
                ],
                dtype=np.int32,
            )
            intensity_mean = intensity_raw[
                [
                    index
                    for index, name in enumerate(names)
                    if fold_by_name[name] != fold
                ]
            ].mean(axis=0)
            intensity_std = intensity_raw[
                [
                    index
                    for index, name in enumerate(names)
                    if fold_by_name[name] != fold
                ]
            ].std(axis=0)
            intensity_std[intensity_std < 1e-6] = 1.0
            standardized = standardize(
                intensity_raw,
                intensity_mean,
                intensity_std,
            ).astype(np.float32)
            matrix[held_out] = standardized_distance(
                predicted[predictor][held_out],
                standardized[candidates[held_out]],
            )
        predicted_distances[predictor] = matrix

    shape_all = np.zeros((len(rows), top_k), dtype=np.float32)
    intensity_all = np.zeros((len(rows), top_k), dtype=np.float32)
    for fold in folds:
        held_out = np.asarray(
            [
                index
                for index, name in enumerate(names)
                if fold_by_name[name] == fold
            ],
            dtype=np.int32,
        )
        shape_all[held_out] = shape_distances[fold][held_out]
        intensity_all[held_out] = intensity_distances[fold][held_out]

    strategy_choices = {
        "v1": ("none", None, v1_choices),
        "frozen_dino": ("none", None, dino_ranks.argmin(axis=1)),
        "shape_oracle": ("offline_oracle", None, choices_from_distance(shape_all)),
        "intensity_oracle": (
            "offline_oracle",
            None,
            choices_from_distance(intensity_all),
        ),
        "factorized_full_oracle": (
            "offline_oracle",
            None,
            choices_from_distance(0.5 * shape_all + 0.5 * intensity_all),
        ),
        "dino_top8_intensity_oracle": (
            "offline_oracle",
            8,
            select_from_shortlist(intensity_all, dino_ranks, 8),
        ),
        "dino_top16_intensity_oracle": (
            "offline_oracle",
            16,
            select_from_shortlist(intensity_all, dino_ranks, 16),
        ),
        "dino_top8_v1": (
            "none",
            8,
            select_from_shortlist(v1_scores, dino_ranks, 8),
        ),
        "dino_top16_v1": (
            "none",
            16,
            select_from_shortlist(v1_scores, dino_ranks, 16),
        ),
    }
    for predictor in predictors:
        for shortlist in (8, 16):
            strategy_choices[f"dino_top{shortlist}_{predictor}"] = (
                predictor,
                shortlist,
                select_from_shortlist(
                    predicted_distances[predictor],
                    dino_ranks,
                    shortlist,
                ),
            )

    if not touch_cache:
        touch_cache = {}
    strategy_rows: dict[str, list[dict[str, str]]] = {}
    all_query_output = []
    metric_cache: dict[tuple[int, int], dict[str, float]] = {}
    for strategy, (predictor, shortlist, choices) in strategy_choices.items():
        output = build_query_output(
            strategy,
            predictor,
            shortlist,
            choices,
            rows,
            candidates,
            oracle_embedding_ranks,
            shape_distances,
            intensity_distances,
            predicted_distances,
            prediction_errors,
            prediction_errors["global_median"],
            fold_by_name,
            touch_cache,
            metric_cache,
            cfg,
        )
        strategy_rows[strategy] = output
        all_query_output.extend(output)
        print(
            {
                "strategy": strategy,
                "summary": summarize(output),
            },
            flush=True,
        )
    write_csv_rows(project_path(cfg["query_output_csv"]), all_query_output, QUERY_FIELDS)

    prediction_summary = {}
    for predictor in predictors:
        comparison = bootstrap_prediction_error(
            rows,
            prediction_errors[predictor],
            prediction_errors["global_median"],
            int(cfg["bootstrap_iterations"]),
            int(cfg["bootstrap_seed"]),
        )
        prediction_summary[predictor] = {
            "all": prediction_metrics(predicted[predictor], standardized_targets),
            "far_probe75_100": prediction_metrics(
                predicted[predictor][
                    [index for index, row in enumerate(rows) if int(row["probe"]) >= 75]
                ],
                standardized_targets[
                    [index for index, row in enumerate(rows) if int(row["probe"]) >= 75]
                ],
            ),
            "vs_global_median": comparison,
        }
    learned_best = min(
        learned_predictors,
        key=lambda name: (
            prediction_summary[name]["all"]["latent_mae"]
            + prediction_summary[name]["far_probe75_100"]["latent_mae"]
        ),
    )
    intensity_observable = bool(
        prediction_summary[learned_best]["vs_global_median"]["accepted"]
    )

    comparison_cfg = {
        "bootstrap_iterations": int(cfg["bootstrap_iterations"]),
        "bootstrap_seed": int(cfg["bootstrap_seed"]),
    }
    v1_reference = [v1_reference_by_name[name] for name in names]
    retrieval_summary = {}
    for strategy, output in strategy_rows.items():
        summary_rows = v1_reference if strategy == "v1" else output
        retrieval_summary[strategy] = {
            "summary": summarize(summary_rows),
            "vs_v1": (
                {"accepted": True, "identity": True}
                if strategy == "v1"
                else bootstrap_comparison(v1_reference, output, comparison_cfg)
            ),
        }
    deployable_strategies = [
        name
        for name in strategy_rows
        if name.startswith("dino_top")
        and not name.endswith("oracle")
        and name not in {"dino_top8_v1", "dino_top16_v1"}
    ]
    accepted_deployable = [
        name
        for name in deployable_strategies
        if retrieval_summary[name]["vs_v1"].get("accepted", False)
    ]
    motion_far_signal = bool(
        prediction_summary["motion_only"]["vs_global_median"]["far_probe75_100"][
            "bootstrap_95_ci"
        ][1]
        < 0
    )
    if accepted_deployable and intensity_observable:
        next_action = (
            "freeze the best factorized cascade for an independent development-validation run"
        )
    elif intensity_observable:
        next_action = (
            "intensity is predictable but the hard shortlist rule is not; tune only a "
            "low-capacity cascade score on strict OOF"
        )
    elif motion_far_signal:
        next_action = (
            "intensity is not observable overall but motion helps far; test the prescribed "
            "four-frame temporal branch without LoRA"
        )
    else:
        next_action = (
            "intensity is not reliably observable from current online inputs; retain V1 and "
            "treat intensity-ambiguous retrievals as cache misses or add control/proprioceptive signals"
        )
    report = {
        "mode": "phase4h_factorized_shape_intensity_predictability_oof_v1",
        "device": str(device),
        "primary_recipe": recipe_name,
        "factorization": {
            "shape_dim": SHAPE_DIM,
            "shape_definition": "8x8 binary-mask occupancy + area + centroid + second moments",
            "intensity_dim": INTENSITY_DIM,
            "intensity_fields": list(INTENSITY_FIELDS),
        },
        "intensity_prediction": prediction_summary,
        "best_learned_predictor": learned_best,
        "intensity_observable_overall_and_far": intensity_observable,
        "motion_far_signal": motion_far_signal,
        "retrieval": retrieval_summary,
        "accepted_deployable_strategies": accepted_deployable,
        "ready_for_development_validation": bool(
            intensity_observable and accepted_deployable
        ),
        "next_action": next_action,
        "training": training_reports,
        "integrity": {
            "source": "strict development 3-fold record-level OOF only",
            "c2_contact_box": "unchanged",
            "top32_candidates": "frozen V1 identity",
            "query_true_probe_used": False,
            "query_tactile_input": False,
            "query_tactile_usage": "offline factor labels, oracle bounds, and evaluation only",
            "sealed_final_holdout_rows_read": 0,
            "development_validation_rows_read": 0,
            "lora_used": False,
            "temporal_visual_frames_used": False,
        },
    }
    write_json(project_path(cfg["metrics_json"]), report)
    print(
        {
            key: value
            for key, value in report.items()
            if key not in {"training", "retrieval"}
        }
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Phase4H.2 factorized tactile-intensity predictability."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--section",
        default="phase4h_factorized_intensity_oof_v1",
    )
    args = parser.parse_args()
    evaluate(args.config, args.section)


if __name__ == "__main__":
    main()
