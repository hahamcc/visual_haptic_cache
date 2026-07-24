"""Train strict-OOF DINO-to-tactile latent projectors on the frozen Top-32 cache."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .config import load_config, project_path
from .evaluate_oracle_tactile_retrieval import tactile_difference, tactile_metrics
from .phase4h_dino_adaptation import (
    TACTILE_LATENT_DIM,
    TactileLatentProjector,
    assert_development_only,
    candidate_groups,
    deployable_motion_feature,
    record_hash_split,
)
from .temporal_progress import DEFAULT_TTC_VALUES, masked_trajectory_features, read_trajectory_tracks
from .train_phase4b_predicted_box_cache_ranker import prediction_map, set_seed
from .train_soft_tactile_cache_ranker import ranks
from .utils import ensure_dir, read_csv_rows, write_csv_rows, write_json


QUERY_FIELDS = [
    "query_record_id", "query_image_name", "query_probe", "oof_fold", "recipe_name",
    "pred_x", "pred_y", "query_padding_ratio", "predicted_ttc", "ttc_entropy",
    "trajectory_stability", "ranker_best_score", "ranker_second_score", "ranker_margin",
    "ranker_margin_normalized", "ranker_entropy", "ranker_oracle_embedding_rank",
    "selected_cache_record_id", "selected_cache_image_name", "top3_cache_record_ids",
    "top3_cache_image_names", "predicted_tactile_latent_error",
    "tactile_diff_mae", "tactile_ssim", "tactile_mask_iou",
]
CANDIDATE_FIELDS = [
    "query_record_id", "query_image_name", "query_probe", "oof_fold", "recipe_name",
    "candidate_rank", "candidate_score", "predicted_tactile_latent_distance",
    "candidate_tactile_latent_distance", "detail_patch_score", "context_patch_score",
    "wide_patch_score", "position_aware_match_score", "hard_negative_flag",
    "candidate_record_id", "candidate_image_name", "candidate_oracle_embedding_rank",
]


def load_frontier(path: Path) -> tuple[str, Path]:
    import json

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    name = data["primary_recipe"]
    entry = next(item for item in data["selected"] if item["recipe"]["name"] == name)
    return name, Path(entry["feature_cache"])


def load_feature_cache(path: Path, expected_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_absolute():
        path = project_path(path)
    data = np.load(path, allow_pickle=False)
    names = [str(value) for value in data["image_names"]]
    if set(names) != set(expected_names):
        raise RuntimeError("Phase4H DINO feature cache does not match development-train rows")
    index = {name: idx for idx, name in enumerate(names)}
    order = np.asarray([index[name] for name in expected_names], dtype=np.int32)
    return data["query_features"][order].astype(np.float32), data["query_padding_ratio"][order].astype(np.float32)


def load_tactile_index(path: Path, expected_names: list[str]) -> dict:
    data = np.load(path, allow_pickle=False)
    names = [str(value) for value in data["image_names"]]
    if set(names) != set(expected_names):
        raise RuntimeError("Phase4H tactile index does not match development-train rows")
    source = {name: idx for idx, name in enumerate(names)}
    order = np.asarray([source[name] for name in expected_names], dtype=np.int32)
    return {
        "raw": data["tactile_latents"][order].astype(np.float32),
        "fold_names": [str(value) for value in data["fold_names"]],
        "fold_means": data["fold_means"].astype(np.float32),
        "fold_stds": data["fold_stds"].astype(np.float32),
        "full_mean": data["full_mean"].astype(np.float32),
        "full_std": data["full_std"].astype(np.float32),
    }


def standardize(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (values - mean) / np.maximum(std, 1e-6)


def score_latents(predicted: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
    return (predicted[:, None] - candidates).square().mean(dim=-1)


def target_distribution(distances: torch.Tensor, temperature: float) -> torch.Tensor:
    return torch.softmax(-distances / max(temperature, 1e-6), dim=1)


def hard_negative_loss(
    scores: torch.Tensor,
    targets: torch.Tensor,
    dino_ranks: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    oracle_rank = torch.argsort(torch.argsort(targets, dim=1), dim=1) + 1
    hard = (dino_ranks <= 8) & (oracle_rank >= 17)
    positive = scores.gather(1, targets.argmin(dim=1, keepdim=True))
    losses = F.relu(margin + positive - scores)
    selected = losses[hard]
    return selected.mean() if selected.numel() else scores.sum() * 0.0


def model_loss(
    model: TactileLatentProjector,
    features: torch.Tensor,
    query_latents: torch.Tensor,
    candidate_latents: torch.Tensor,
    dino_ranks: torch.Tensor,
    cfg: dict,
) -> tuple[torch.Tensor, dict[str, float]]:
    predicted = model(features)
    scores = score_latents(predicted, candidate_latents)
    true_distances = score_latents(query_latents, candidate_latents)
    distribution = target_distribution(true_distances, float(cfg["target_temperature"]))
    listwise = -(distribution * torch.log_softmax(-scores / float(cfg["score_temperature"]), dim=1)).sum(dim=1).mean()
    latent = F.smooth_l1_loss(predicted, query_latents)
    hard = hard_negative_loss(scores, true_distances, dino_ranks, float(cfg["hard_negative_margin"]))
    total = (
        float(cfg["listwise_loss_weight"]) * listwise
        + float(cfg["latent_loss_weight"]) * latent
        + float(cfg["hard_negative_loss_weight"]) * hard
    )
    return total, {
        "total": float(total.detach().cpu()),
        "listwise": float(listwise.detach().cpu()),
        "latent": float(latent.detach().cpu()),
        "hard_negative": float(hard.detach().cpu()),
    }


def validation_loss(
    model: TactileLatentProjector,
    features: np.ndarray,
    query_latents: np.ndarray,
    candidates: np.ndarray,
    all_latents: np.ndarray,
    dino_ranks: np.ndarray,
    indices: np.ndarray,
    cfg: dict,
    device: torch.device,
) -> float:
    model.eval()
    values = []
    batch_size = int(cfg["batch_size"])
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch = indices[start : start + batch_size]
            loss, _ = model_loss(
                model,
                torch.from_numpy(features[batch]).to(device),
                torch.from_numpy(query_latents[batch]).to(device),
                torch.from_numpy(all_latents[candidates[batch]]).to(device),
                torch.from_numpy(dino_ranks[batch]).to(device),
                cfg,
            )
            values.append(float(loss.cpu()) * len(batch))
    return float(sum(values) / max(len(indices), 1))


def train_one(
    features: np.ndarray,
    query_latents: np.ndarray,
    candidates: np.ndarray,
    all_latents: np.ndarray,
    dino_ranks: np.ndarray,
    fit_indices: np.ndarray,
    validation_indices: np.ndarray,
    cfg: dict,
    device: torch.device,
    checkpoint_path: Path,
    metadata: dict,
) -> tuple[TactileLatentProjector, dict]:
    model = TactileLatentProjector(features.shape[1], TACTILE_LATENT_DIM, float(cfg["dropout"])).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    best_loss, best_epoch, stale, history = float("inf"), 0, 0, []
    batch_size = int(cfg["batch_size"])
    for epoch in range(1, int(cfg["epochs"]) + 1):
        model.train()
        order = np.random.permutation(fit_indices)
        epoch_values = []
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            loss, parts = model_loss(
                model,
                torch.from_numpy(features[batch]).to(device),
                torch.from_numpy(query_latents[batch]).to(device),
                torch.from_numpy(all_latents[candidates[batch]]).to(device),
                torch.from_numpy(dino_ranks[batch]).to(device),
                cfg,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(cfg["gradient_clip"]))
            optimizer.step()
            epoch_values.append(parts)
        current = validation_loss(
            model, features, query_latents, candidates, all_latents,
            dino_ranks, validation_indices, cfg, device,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean([item["total"] for item in epoch_values])),
                "validation_loss": current,
            }
        )
        if current < best_loss - 1e-6:
            best_loss, best_epoch, stale = current, epoch, 0
            ensure_dir(checkpoint_path.parent)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "input_dim": features.shape[1],
                    "latent_dim": TACTILE_LATENT_DIM,
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


def predict_scores(
    model: TactileLatentProjector,
    features: np.ndarray,
    candidates: np.ndarray,
    all_latents: np.ndarray,
    indices: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    scores, predictions = [], []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch = indices[start : start + batch_size]
            predicted = model(torch.from_numpy(features[batch]).to(device))
            score = score_latents(
                predicted,
                torch.from_numpy(all_latents[candidates[batch]]).to(device),
            )
            predictions.append(predicted.cpu().numpy())
            scores.append(score.cpu().numpy())
    return np.concatenate(scores), np.concatenate(predictions)


def softmax_entropy(scores: np.ndarray) -> float:
    logits = -scores + float(scores.min())
    probabilities = np.exp(logits - logits.max())
    probabilities /= probabilities.sum()
    return float(-np.sum(probabilities * np.log(np.maximum(probabilities, 1e-12))) / math.log(len(scores)))


def summarize(rows: list[dict[str, str]]) -> dict:
    output = {}
    for name, subset in (
        ("all", rows),
        ("near_probe5_20", [row for row in rows if int(row["query_probe"]) <= 20]),
        ("mid_probe30_50", [row for row in rows if 30 <= int(row["query_probe"]) <= 50]),
        ("far_probe75_100", [row for row in rows if int(row["query_probe"]) >= 75]),
    ):
        output[name] = {
            "queries": len(subset),
            "tactile_diff_mae": float(np.mean([float(row["tactile_diff_mae"]) for row in subset])) if subset else None,
            "tactile_ssim": float(np.mean([float(row["tactile_ssim"]) for row in subset])) if subset else None,
            "tactile_mask_iou": float(np.mean([float(row["tactile_mask_iou"]) for row in subset])) if subset else None,
            "oracle_top1": float(np.mean([int(row["ranker_oracle_embedding_rank"]) == 1 for row in subset])) if subset else None,
            "oracle_top3": float(np.mean([int(row["ranker_oracle_embedding_rank"]) <= 3 for row in subset])) if subset else None,
        }
    return output


def build_outputs(
    rows: list[dict[str, str]],
    predictions: dict[str, dict[str, str]],
    recipe_name: str,
    candidates: np.ndarray,
    scores: np.ndarray,
    predicted_latents: np.ndarray,
    standardized_latents_by_fold: dict[str, np.ndarray],
    feature_padding: np.ndarray,
    online_motion: np.ndarray,
    recipe_candidate_map: dict[tuple[str, str], dict[str, str]],
    cfg: dict,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    touch_cache: dict[str, np.ndarray] = {}
    query_output, candidate_output = [], []
    for index, row in enumerate(rows):
        fold = predictions[row["image_name"]]["oof_fold"]
        latents = standardized_latents_by_fold[fold]
        target = ((latents[candidates[index]] - latents[index][None]) ** 2).mean(axis=1)
        order = np.argsort(scores[index], kind="stable")
        oracle_ranks = ranks(target)
        model_ranks = ranks(scores[index])
        choice = int(order[0])
        selected = rows[int(candidates[index, choice])]
        metric = tactile_metrics(
            tactile_difference(row["touch_path"], touch_cache, int(cfg["tactile_size"])),
            tactile_difference(selected["touch_path"], touch_cache, int(cfg["tactile_size"])),
            float(cfg["tactile_mask_threshold"]),
        )
        best, second = float(scores[index, order[0]]), float(scores[index, order[1]])
        probability_row = predictions[row["image_name"]]
        predicted_ttc = float(online_motion[index, 25] * 100.0)
        ttc_entropy = float(online_motion[index, 26])
        trajectory_stability = float(online_motion[index, 18])
        top3 = order[:3]
        query_output.append(
            {
                "query_record_id": row["record_id"],
                "query_image_name": row["image_name"],
                "query_probe": row["probe"],
                "oof_fold": fold,
                "recipe_name": recipe_name,
                "pred_x": probability_row["pred_x"],
                "pred_y": probability_row["pred_y"],
                "query_padding_ratio": f"{feature_padding[index]:.6f}",
                "predicted_ttc": f"{predicted_ttc:.6f}",
                "ttc_entropy": f"{ttc_entropy:.6f}",
                "trajectory_stability": f"{trajectory_stability:.6f}",
                "ranker_best_score": f"{best:.6f}",
                "ranker_second_score": f"{second:.6f}",
                "ranker_margin": f"{second - best:.6f}",
                "ranker_margin_normalized": f"{(second - best) / max(float(scores[index].std()), 1e-6):.6f}",
                "ranker_entropy": f"{softmax_entropy(scores[index]):.6f}",
                "ranker_oracle_embedding_rank": str(int(model_ranks[int(np.argmin(target))])),
                "selected_cache_record_id": selected["record_id"],
                "selected_cache_image_name": selected["image_name"],
                "top3_cache_record_ids": "|".join(rows[int(candidates[index, item])]["record_id"] for item in top3),
                "top3_cache_image_names": "|".join(rows[int(candidates[index, item])]["image_name"] for item in top3),
                "predicted_tactile_latent_error": f"{float(np.mean((predicted_latents[index] - latents[index]) ** 2)):.6f}",
                **{key: f"{metric[key]:.6f}" for key in ("tactile_diff_mae", "tactile_ssim", "tactile_mask_iou")},
            }
        )
        for rank, item in enumerate(order, start=1):
            item = int(item)
            cache = rows[int(candidates[index, item])]
            source = recipe_candidate_map[(row["image_name"], cache["image_name"])]
            candidate_output.append(
                {
                    "query_record_id": row["record_id"],
                    "query_image_name": row["image_name"],
                    "query_probe": row["probe"],
                    "oof_fold": fold,
                    "recipe_name": recipe_name,
                    "candidate_rank": str(rank),
                    "candidate_score": f"{scores[index, item]:.6f}",
                    "predicted_tactile_latent_distance": f"{scores[index, item]:.6f}",
                    "candidate_tactile_latent_distance": f"{target[item]:.6f}",
                    "detail_patch_score": source.get("detail_patch_score", ""),
                    "context_patch_score": source.get("context_patch_score", ""),
                    "wide_patch_score": source.get("wide_patch_score", ""),
                    "position_aware_match_score": source.get("position_aware_match_score", ""),
                    "hard_negative_flag": str(int(int(source["candidate_rank"]) <= 8 and oracle_ranks[item] >= 17)),
                    "candidate_record_id": cache["record_id"],
                    "candidate_image_name": cache["image_name"],
                    "candidate_oracle_embedding_rank": str(int(oracle_ranks[item])),
                }
            )
    return query_output, candidate_output


def train(config_path: str, section: str) -> dict:
    cfg = load_config(config_path)[section]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    partition_path = project_path(cfg["final_partition_csv"])
    samples = read_csv_rows(project_path(cfg["samples_csv"]))
    assert_development_only(samples, partition_path)
    rows = [row for row in samples if row["dataset_split"] == "train"]
    names = [row["image_name"] for row in rows]
    row_index = {name: index for index, name in enumerate(names)}
    predictions = prediction_map(
        read_csv_rows(project_path(cfg["oof_predictions_csv"])),
        rows,
        "train",
        "Phase4H tactile-alignment OOF",
    )
    fold_by_name = {name: prediction["oof_fold"] for name, prediction in predictions.items()}
    folds = sorted(set(fold_by_name.values()))
    recipe_name, feature_path = load_frontier(project_path(cfg["frontier_json"]))
    visual, padding = load_feature_cache(feature_path, names)
    tactile = load_tactile_index(project_path(cfg["tactile_index_npz"]), names)
    fold_stat_index = {name: idx for idx, name in enumerate(tactile["fold_names"])}
    standardized_by_fold = {
        fold: standardize(tactile["raw"], tactile["fold_means"][fold_stat_index[fold]], tactile["fold_stds"][fold_stat_index[fold]])
        for fold in folds
    }
    v1_groups = candidate_groups(read_csv_rows(project_path(cfg["v1_candidate_csv"])), int(cfg["geometry_filter_k"]))
    candidates = np.stack(
        [
            np.asarray([row_index[item["candidate_image_name"]] for item in v1_groups[name]], dtype=np.int32)
            for name in names
        ]
    )
    recipe_rows = [
        row for row in read_csv_rows(project_path(cfg["ablation_candidate_csv"]))
        if row["recipe_name"] == recipe_name
    ]
    recipe_candidate_map = {
        (row["query_image_name"], row["candidate_image_name"]): row for row in recipe_rows
    }
    if len(recipe_candidate_map) != len(rows) * int(cfg["geometry_filter_k"]):
        raise RuntimeError(f"Selected recipe {recipe_name} does not cover the fixed Top-32")
    dino_ranks = np.asarray(
        [
            [int(recipe_candidate_map[(name, item["candidate_image_name"])]["candidate_rank"]) for item in v1_groups[name]]
            for name in names
        ],
        dtype=np.int64,
    )

    tracks = read_trajectory_tracks(project_path(cfg["motion_tracks_csv"]))
    ttc_by_name = {}
    ttc_value = str(cfg.get("ttc_predictions_csv", "")).strip()
    ttc_path = project_path(ttc_value) if ttc_value else None
    if ttc_path is not None and ttc_path.is_file():
        ttc_by_name = {row["image_name"]: row for row in read_csv_rows(ttc_path) if row["image_name"] in row_index}
    online_motion = []
    for row in rows:
        trajectory, mask, quality = masked_trajectory_features(
            row, tracks, int(cfg["trajectory_history_frames"]),
            float(cfg["trajectory_spatial_scale_px"]), float(cfg["trajectory_speed_scale_px"]),
        )
        prediction = predictions[row["image_name"]]
        ttc_prediction = ttc_by_name.get(row["image_name"], prediction)
        online_motion.append(
            deployable_motion_feature(
                row, float(prediction["pred_x"]), float(prediction["pred_y"]),
                trajectory, mask, quality, ttc_prediction, cfg.get("ttc_values", DEFAULT_TTC_VALUES),
            )
        )
    online_motion = np.stack(online_motion).astype(np.float32)
    all_scores = np.zeros((len(rows), candidates.shape[1]), dtype=np.float32)
    all_predicted = np.zeros((len(rows), TACTILE_LATENT_DIM), dtype=np.float32)
    training_reports = []
    checkpoint_dir = project_path(cfg["checkpoint_dir"])
    ensure_dir(checkpoint_dir)
    seeds = [int(value) for value in cfg["seeds"]]
    for fold in folds:
        held_out = np.asarray([index for index, name in enumerate(names) if fold_by_name[name] == fold], dtype=np.int32)
        outer_fit = np.asarray([index for index, name in enumerate(names) if fold_by_name[name] != fold], dtype=np.int32)
        inner_validation = np.asarray(
            [
                index for index in outer_fit
                if record_hash_split(rows[int(index)]["record_id"], float(cfg["inner_validation_fraction"]), int(cfg["inner_split_seed"]) + int(fold))
            ],
            dtype=np.int32,
        )
        inner_fit = np.asarray([index for index in outer_fit if index not in set(inner_validation.tolist())], dtype=np.int32)
        if not len(inner_fit) or not len(inner_validation):
            raise RuntimeError(f"Fold {fold} inner record split is empty")
        mean, std = online_motion[inner_fit].mean(axis=0), online_motion[inner_fit].std(axis=0)
        std[std < 1e-6] = 1.0
        features = np.concatenate((visual, standardize(online_motion, mean, std)), axis=1).astype(np.float32)
        latents = standardized_by_fold[fold]
        seed_scores, seed_predictions = [], []
        for seed in seeds:
            set_seed(seed)
            model, report = train_one(
                features, latents, candidates, latents, dino_ranks,
                inner_fit, inner_validation, cfg, device,
                checkpoint_dir / f"fold_{fold}_seed_{seed}.pt",
                {
                    "scope": "strict_oof",
                    "fold": fold,
                    "seed": seed,
                    "recipe_name": recipe_name,
                    "motion_mean": mean,
                    "motion_std": std,
                    "tactile_mean": tactile["fold_means"][fold_stat_index[fold]],
                    "tactile_std": tactile["fold_stds"][fold_stat_index[fold]],
                    "query_true_probe_used": False,
                    "query_tactile_input": False,
                },
            )
            scores, predicted = predict_scores(
                model, features, candidates, latents, held_out, device, int(cfg["batch_size"]),
            )
            seed_scores.append(scores)
            seed_predictions.append(predicted)
            training_reports.append({"fold": fold, "seed": seed, **report})
        all_scores[held_out] = np.mean(seed_scores, axis=0)
        all_predicted[held_out] = np.mean(seed_predictions, axis=0)

    # Train deployable full-development checkpoints only after every strict OOF fold is complete.
    full_mean, full_std = online_motion.mean(axis=0), online_motion.std(axis=0)
    full_std[full_std < 1e-6] = 1.0
    full_features = np.concatenate((visual, standardize(online_motion, full_mean, full_std)), axis=1).astype(np.float32)
    full_latents = standardize(tactile["raw"], tactile["full_mean"], tactile["full_std"])
    full_validation = np.asarray(
        [
            index for index, row in enumerate(rows)
            if record_hash_split(row["record_id"], float(cfg["inner_validation_fraction"]), int(cfg["inner_split_seed"]) + 1000)
        ],
        dtype=np.int32,
    )
    full_fit = np.asarray([index for index in range(len(rows)) if index not in set(full_validation.tolist())], dtype=np.int32)
    for seed in seeds:
        set_seed(seed)
        _, report = train_one(
            full_features, full_latents, candidates, full_latents, dino_ranks,
            full_fit, full_validation, cfg, device,
            checkpoint_dir / f"full_seed_{seed}.pt",
            {
                "scope": "full_development_for_frozen_validation",
                "seed": seed,
                "recipe_name": recipe_name,
                "motion_mean": full_mean,
                "motion_std": full_std,
                "tactile_mean": tactile["full_mean"],
                "tactile_std": tactile["full_std"],
                "query_true_probe_used": False,
                "query_tactile_input": False,
            },
        )
        training_reports.append({"fold": "full", "seed": seed, **report})

    query_output, candidate_output = build_outputs(
        rows, predictions, recipe_name, candidates, all_scores, all_predicted,
        standardized_by_fold, padding, online_motion, recipe_candidate_map, cfg,
    )
    write_csv_rows(project_path(cfg["query_output_csv"]), query_output, QUERY_FIELDS)
    write_csv_rows(project_path(cfg["candidate_output_csv"]), candidate_output, CANDIDATE_FIELDS)
    summary = {
        "mode": "phase4h_strict_oof_dino_to_tactile_latent_alignment_v1",
        "device": str(device),
        "recipe_name": recipe_name,
        "seeds": seeds,
        "summary": summarize(query_output),
        "training": training_reports,
        "integrity": {
            "c2_contact_box": "unchanged",
            "top32_candidates": "frozen V1 identity",
            "same_record_candidates": 0,
            "sealed_final_holdout_rows_read": 0,
            "query_true_probe_used": False,
            "query_tactile_input": False,
            "query_tactile_usage": "offline listwise/latent supervision and evaluation only",
            "candidate_tactile_usage": "precomputed online cache index",
        },
    }
    write_json(project_path(cfg["metrics_json"]), summary)
    print({key: value for key, value in summary.items() if key != "training"})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train strict-OOF Phase4H DINO-to-tactile latent alignment.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--section", default="phase4h_tactile_alignment_oof_v1")
    args = parser.parse_args()
    train(args.config, args.section)


if __name__ == "__main__":
    main()
