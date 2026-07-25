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

FACTOR_CANDIDATE_FIELDS = [
    "query_record_id",
    "query_image_name",
    "query_probe",
    "oof_fold",
    "candidate_rank",
    "v1_score",
    "dino_rank",
    "dino_score",
    "detail_patch_score",
    "context_patch_score",
    "wide_patch_score",
    "position_aware_match_score",
    "predicted_intensity_distance_global_median",
    "predicted_intensity_distance_motion_only",
    "predicted_intensity_distance_dino_only",
    "predicted_intensity_distance_dino_motion",
    "candidate_record_id",
    "candidate_image_name",
    "candidate_tactile_embedding_distance",
    "candidate_tactile_ssim",
    "candidate_tactile_mask_iou",
    "candidate_oracle_embedding_rank",
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
        "reused_checkpoint": False,
    }


def load_compatible_regressor_checkpoint(
    checkpoint_path: Path,
    input_dim: int,
    cfg: dict,
    metadata: dict,
    device: torch.device,
) -> LowCapacityIntensityRegressor | None:
    """Load a completed model only when its OOF identity and statistics match."""
    if not checkpoint_path.is_file():
        return None
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        if int(checkpoint.get("input_dim", -1)) != input_dim:
            return None
        saved_metadata = checkpoint.get("metadata", {})
        for key in ("scope", "predictor", "fold", "seed", "primary_recipe"):
            if saved_metadata.get(key) != metadata.get(key):
                return None
        for key in (
            "feature_mean",
            "feature_std",
            "intensity_mean",
            "intensity_std",
        ):
            saved = np.asarray(saved_metadata.get(key), dtype=np.float32)
            expected = np.asarray(metadata.get(key), dtype=np.float32)
            if saved.shape != expected.shape or not np.allclose(
                saved,
                expected,
                rtol=1e-6,
                atol=1e-7,
            ):
                return None
        model = LowCapacityIntensityRegressor(
            input_dim,
            int(cfg["hidden_dim"]),
            INTENSITY_DIM,
            float(cfg["dropout"]),
        ).to(device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        return model
    except (EOFError, KeyError, OSError, RuntimeError, TypeError, ValueError):
        return None


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
        if not len(selected):
            raise RuntimeError(f"No records are available for prediction regime {regime}")
        deltas = prediction_error[selected] - baseline_error[selected]
        selected_records = np.asarray(
            [rows[int(index)]["record_id"] for index in selected]
        )
        records, record_indices = np.unique(
            selected_records,
            return_inverse=True,
        )
        counts = np.bincount(record_indices, minlength=len(records)).astype(
            np.float64
        )
        sums = np.bincount(
            record_indices,
            weights=deltas.astype(np.float64),
            minlength=len(records),
        )
        draws = rng.integers(
            0,
            len(records),
            size=(iterations, len(records)),
        )
        samples = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
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


def _retrieval_metric_matrix(
    rows: list[dict[str, str]],
) -> np.ndarray:
    return np.asarray(
        [
            [
                float(row["tactile_diff_mae"]),
                float(row["tactile_ssim"]),
                float(row["tactile_mask_iou"]),
                float(int(row["ranker_oracle_embedding_rank"]) == 1),
            ]
            for row in rows
        ],
        dtype=np.float64,
    )


def _retrieval_metric_values(values: np.ndarray) -> dict[str, float]:
    means = values.mean(axis=0)
    return {
        "queries": float(len(values)),
        "tactile_diff_mae": float(means[0]),
        "tactile_ssim": float(means[1]),
        "tactile_mask_iou": float(means[2]),
        "oracle_top1": float(means[3]),
    }


def fast_bootstrap_comparison(
    v1_rows: list[dict[str, str]],
    current_rows: list[dict[str, str]],
    cfg: dict,
) -> dict:
    """Record bootstrap equivalent to Phase4G without Python query loops."""
    v1_by_name = {row["query_image_name"]: row for row in v1_rows}
    current_by_name = {row["query_image_name"]: row for row in current_rows}
    if (
        len(v1_by_name) != len(v1_rows)
        or len(current_by_name) != len(current_rows)
        or set(v1_by_name) != set(current_by_name)
    ):
        raise RuntimeError(
            "Bootstrap requires one-to-one V1 and current retrieval queries."
        )
    ordered_names = list(v1_by_name)
    base_values = _retrieval_metric_matrix(
        [v1_by_name[name] for name in ordered_names]
    )
    current_values = _retrieval_metric_matrix(
        [current_by_name[name] for name in ordered_names]
    )
    records = np.asarray(
        [v1_by_name[name]["query_record_id"] for name in ordered_names]
    )
    probes = np.asarray(
        [int(v1_by_name[name]["query_probe"]) for name in ordered_names]
    )
    all_records = np.unique(records)
    record_lookup = {
        record_id: index for index, record_id in enumerate(all_records)
    }
    metric_names = (
        "tactile_diff_mae",
        "tactile_ssim",
        "tactile_mask_iou",
        "oracle_top1",
    )
    rng = np.random.default_rng(int(cfg["bootstrap_seed"]))
    output = {}
    for regime, selected in (
        ("all", np.ones(len(ordered_names), dtype=bool)),
        ("far_probe75_100", probes >= 75),
    ):
        if not selected.any():
            raise RuntimeError(f"No queries are available for retrieval regime {regime}")
        regime_base = base_values[selected]
        regime_current = current_values[selected]
        regime_records = records[selected]
        record_indices = np.asarray(
            [record_lookup[record_id] for record_id in regime_records],
            dtype=np.int32,
        )
        counts = np.bincount(
            record_indices,
            minlength=len(all_records),
        ).astype(np.float64)
        base_sums = np.stack(
            [
                np.bincount(
                    record_indices,
                    weights=regime_base[:, column],
                    minlength=len(all_records),
                )
                for column in range(regime_base.shape[1])
            ],
            axis=1,
        )
        current_sums = np.stack(
            [
                np.bincount(
                    record_indices,
                    weights=regime_current[:, column],
                    minlength=len(all_records),
                )
                for column in range(regime_current.shape[1])
            ],
            axis=1,
        )
        draws = rng.integers(
            0,
            len(all_records),
            size=(int(cfg["bootstrap_iterations"]), len(all_records)),
        )
        denominators = counts[draws].sum(axis=1, keepdims=True)
        if (denominators == 0).any():
            raise RuntimeError(
                f"Record bootstrap drew no queries for retrieval regime {regime}"
            )
        samples = (
            current_sums[draws].sum(axis=1) - base_sums[draws].sum(axis=1)
        ) / denominators
        base = _retrieval_metric_values(regime_base)
        current = _retrieval_metric_values(regime_current)
        deltas = {
            metric: current[metric] - base[metric] for metric in metric_names
        }
        ci = {
            metric: [
                float(np.quantile(samples[:, column], 0.025)),
                float(np.quantile(samples[:, column], 0.975)),
            ]
            for column, metric in enumerate(metric_names)
        }
        point_pass = bool(
            current["tactile_diff_mae"] < base["tactile_diff_mae"]
            and current["tactile_ssim"] >= base["tactile_ssim"]
            and current["tactile_mask_iou"] >= base["tactile_mask_iou"]
            and current["oracle_top1"] >= base["oracle_top1"]
        )
        ci_pass = bool(
            ci["tactile_diff_mae"][1] < 0
            and ci["tactile_ssim"][0] >= 0
            and ci["tactile_mask_iou"][0] >= 0
        )
        output[regime] = {
            "v1": base,
            "fusion": current,
            "delta_fusion_minus_v1": deltas,
            "bootstrap_95_ci": ci,
            "point_pass": point_pass,
            "ci_pass": ci_pass,
        }
    output["accepted"] = bool(
        output["all"]["point_pass"]
        and output["all"]["ci_pass"]
        and output["far_probe75_100"]["point_pass"]
        and output["far_probe75_100"]["ci_pass"]
    )
    return output


def choices_from_distance(distances: np.ndarray) -> np.ndarray:
    return distances.argmin(axis=1).astype(np.int32)


def load_compatible_query_output(
    output_path: Path,
    strategy_choices: dict[str, tuple[str, int | None, np.ndarray]],
    rows: list[dict[str, str]],
    candidates: np.ndarray,
) -> dict[str, list[dict[str, str]]] | None:
    """Reuse completed tactile evaluations after verifying every selected pair."""
    if not output_path.is_file():
        return None
    try:
        cached = read_csv_rows(output_path)
    except (OSError, ValueError):
        return None
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in cached:
        grouped[row.get("strategy", "")].append(row)
    if set(grouped) != set(strategy_choices):
        return None
    names = [row["image_name"] for row in rows]
    output = {}
    for strategy, (predictor, shortlist, choices) in strategy_choices.items():
        by_name = {
            row["query_image_name"]: row for row in grouped[strategy]
        }
        if len(by_name) != len(grouped[strategy]) or set(by_name) != set(names):
            return None
        ordered = [by_name[name] for name in names]
        for index, cached_row in enumerate(ordered):
            selected_index = int(candidates[index, int(choices[index])])
            if (
                cached_row["selected_cache_image_name"]
                != rows[selected_index]["image_name"]
                or cached_row["predictor"] != predictor
                or cached_row["shortlist_k"]
                != ("" if shortlist is None else str(shortlist))
            ):
                return None
        output[strategy] = ordered
    return output


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


def build_factor_candidate_output(
    rows: list[dict[str, str]],
    candidates: np.ndarray,
    v1_groups: dict[str, list[dict[str, str]]],
    recipe_map: dict[tuple[str, str], dict[str, str]],
    dino_ranks: np.ndarray,
    oracle_embedding_ranks: np.ndarray,
    predicted_distances: dict[str, np.ndarray],
    fold_by_name: dict[str, str],
) -> list[dict[str, str]]:
    """Export online candidate signals plus offline labels for Phase4I."""
    output = []
    for query_index, query in enumerate(rows):
        query_name = query["image_name"]
        for candidate_index, v1_row in enumerate(v1_groups[query_name]):
            cache = rows[int(candidates[query_index, candidate_index])]
            recipe = recipe_map[(query_name, cache["image_name"])]
            output.append(
                {
                    "query_record_id": query["record_id"],
                    "query_image_name": query_name,
                    "query_probe": query["probe"],
                    "oof_fold": fold_by_name[query_name],
                    "candidate_rank": v1_row["candidate_rank"],
                    "v1_score": v1_row["candidate_score"],
                    "dino_rank": str(
                        int(dino_ranks[query_index, candidate_index])
                    ),
                    "dino_score": recipe["candidate_score"],
                    "detail_patch_score": recipe.get("detail_patch_score", ""),
                    "context_patch_score": recipe.get(
                        "context_patch_score",
                        "",
                    ),
                    "wide_patch_score": recipe.get("wide_patch_score", ""),
                    "position_aware_match_score": recipe.get(
                        "position_aware_match_score",
                        "",
                    ),
                    **{
                        f"predicted_intensity_distance_{predictor}": (
                            f"{predicted_distances[predictor][query_index, candidate_index]:.9f}"
                        )
                        for predictor in (
                            "global_median",
                            "motion_only",
                            "dino_only",
                            "dino_motion",
                        )
                    },
                    "candidate_record_id": cache["record_id"],
                    "candidate_image_name": cache["image_name"],
                    "candidate_tactile_embedding_distance": recipe[
                        "candidate_tactile_embedding_distance"
                    ],
                    "candidate_tactile_ssim": recipe.get(
                        "candidate_tactile_ssim",
                        "",
                    ),
                    "candidate_tactile_mask_iou": recipe.get(
                        "candidate_tactile_mask_iou",
                        "",
                    ),
                    "candidate_oracle_embedding_rank": str(
                        int(
                            oracle_embedding_ranks[
                                query_index,
                                candidate_index,
                            ]
                        )
                    ),
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
                checkpoint_path = (
                    checkpoint_dir / f"{predictor}_fold_{fold}_seed_{seed}.pt"
                )
                checkpoint_metadata = {
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
                }
                model = None
                if bool(cfg.get("reuse_compatible_checkpoints", False)):
                    model = load_compatible_regressor_checkpoint(
                        checkpoint_path,
                        features.shape[1],
                        cfg,
                        checkpoint_metadata,
                        device,
                    )
                if model is None:
                    model, report = train_regressor(
                        features,
                        intensity_standardized,
                        inner_fit,
                        inner_validation,
                        rows,
                        cfg,
                        device,
                        checkpoint_path,
                        checkpoint_metadata,
                    )
                else:
                    report = {
                        "best_epoch": None,
                        "best_validation_loss": None,
                        "epochs_ran": 0,
                        "history": [],
                        "reused_checkpoint": True,
                    }
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
                if report["reused_checkpoint"]:
                    print(
                        "phase4h.2 "
                        f"predictor={predictor} fold={fold} seed={seed} "
                        "checkpoint=reused",
                        flush=True,
                    )
                else:
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
    query_output_path = project_path(cfg["query_output_csv"])
    strategy_rows = None
    if bool(cfg.get("reuse_complete_query_output", False)):
        strategy_rows = load_compatible_query_output(
            query_output_path,
            strategy_choices,
            rows,
            candidates,
        )
    if strategy_rows is not None:
        print(f"phase4h.2: reusing {query_output_path}", flush=True)
        for strategy, output in strategy_rows.items():
            print(
                {"strategy": strategy, "summary": summarize(output)},
                flush=True,
            )
    else:
        strategy_rows = {}
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
        write_csv_rows(query_output_path, all_query_output, QUERY_FIELDS)

    factor_candidate_output = build_factor_candidate_output(
        rows,
        candidates,
        v1_groups,
        recipe_map,
        dino_ranks,
        oracle_embedding_ranks,
        predicted_distances,
        fold_by_name,
    )
    write_csv_rows(
        project_path(cfg["candidate_output_csv"]),
        factor_candidate_output,
        FACTOR_CANDIDATE_FIELDS,
    )
    print(
        f"phase4h.2: wrote {len(factor_candidate_output)} candidate rows",
        flush=True,
    )

    print("phase4h.2: starting vectorized record bootstrap", flush=True)
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
                else fast_bootstrap_comparison(v1_reference, output, comparison_cfg)
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
