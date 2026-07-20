from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .build_cache_retrieval import motion_geometry_feature, standardize, visual_patch_feature
from .config import load_config, project_path
from .evaluate_oracle_tactile_retrieval import tactile_difference, tactile_embedding, tactile_metrics
from .utils import ensure_dir, read_csv_rows, write_csv_rows, write_json


QUERY_FIELDS = [
    "query_record_id", "query_image_name", "query_probe", "current_cache_record_id", "matcher_cache_record_id",
    "tactile_oracle_cache_record_id", "current_key_rank_of_tactile_best", "matcher_rank_of_tactile_best",
    "matcher_tactile_best_top5", "current_tactile_diff_mae", "matcher_tactile_diff_mae", "oracle_tactile_diff_mae",
    "current_tactile_ssim", "matcher_tactile_ssim", "oracle_tactile_ssim", "current_tactile_mask_iou",
    "matcher_tactile_mask_iou", "oracle_tactile_mask_iou", "current_tactile_embedding_distance",
    "matcher_tactile_embedding_distance", "oracle_tactile_embedding_distance",
    "current_tactile_area_delta", "matcher_tactile_area_delta", "oracle_tactile_area_delta",
    "current_tactile_centroid_distance", "matcher_tactile_centroid_distance", "oracle_tactile_centroid_distance",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def record_number(record_id: str) -> int:
    try:
        return int(record_id.rsplit("_", 1)[-1])
    except ValueError as exc:
        raise ValueError(f"Unexpected record ID: {record_id}") from exc


def ranks(values: np.ndarray) -> np.ndarray:
    result = np.empty(len(values), dtype=np.int32)
    result[np.argsort(values, kind="stable")] = np.arange(1, len(values) + 1)
    return result


@dataclass
class CandidateGroups:
    features: np.ndarray
    targets: np.ndarray
    current_scores: np.ndarray
    cache_indices: np.ndarray
    rows: list[dict[str, str]]


class LocalCacheMatcher(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


def build_groups(
    query_rows: list[dict[str, str]],
    cache_rows: list[dict[str, str]],
    cache_geometry_z: np.ndarray,
    cache_visual_z: np.ndarray,
    cache_tactile_embeddings: np.ndarray,
    geometry_mean: np.ndarray,
    geometry_std: np.ndarray,
    visual_mean: np.ndarray,
    visual_std: np.ndarray,
    crop_size: int,
    tactile_size: int,
    filter_k: int,
    geometry_weight: float,
    visual_weight: float,
    exclude_same_record: bool,
) -> CandidateGroups:
    features, targets, current_scores, cache_indices = [], [], [], []
    for query in query_rows:
        x, y = float(query["target_tip_x"]), float(query["target_tip_y"])
        query_geometry = (motion_geometry_feature(query, x, y) - geometry_mean) / geometry_std
        geometry_distances = np.linalg.norm(cache_geometry_z - query_geometry[None], axis=1)
        allowed = np.ones(len(cache_rows), dtype=bool)
        if exclude_same_record:
            allowed = np.asarray([cache["record_id"] != query["record_id"] for cache in cache_rows])
        allowed_indices = np.flatnonzero(allowed)
        local_k = min(filter_k, len(allowed_indices))
        allowed_distances = geometry_distances[allowed_indices]
        shortlist = allowed_indices[np.argpartition(allowed_distances, local_k - 1)[:local_k]]
        shortlist = shortlist[np.argsort(geometry_distances[shortlist], kind="stable")]
        query_visual = (visual_patch_feature(query["vision_path"], x, y, crop_size) - visual_mean) / visual_std
        visual_distances = np.linalg.norm(cache_visual_z[shortlist] - query_visual[None], axis=1)
        current = (
            geometry_weight * geometry_distances[shortlist] / math.sqrt(cache_geometry_z.shape[1])
            + visual_weight * visual_distances / math.sqrt(cache_visual_z.shape[1])
        )
        query_diff = tactile_difference(query["touch_path"], {}, tactile_size)
        tactile_target = np.linalg.norm(cache_tactile_embeddings[shortlist] - tactile_embedding(query_diff)[None], axis=1)
        pair_features = np.concatenate(
            [
                np.abs(cache_visual_z[shortlist] - query_visual[None]),
                np.abs(cache_geometry_z[shortlist] - query_geometry[None]),
                np.repeat(query_geometry[None], len(shortlist), axis=0),
                cache_geometry_z[shortlist],
                current[:, None],
            ],
            axis=1,
        ).astype(np.float32)
        features.append(pair_features)
        targets.append(tactile_target.astype(np.float32))
        current_scores.append(current.astype(np.float32))
        cache_indices.append(shortlist.astype(np.int32))
    return CandidateGroups(
        features=np.stack(features), targets=np.stack(targets), current_scores=np.stack(current_scores),
        cache_indices=np.stack(cache_indices), rows=query_rows,
    )


def prediction_scores(model: nn.Module, features: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    scores: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            batch = torch.from_numpy(features[start:start + batch_size]).to(device)
            scores.append(model(batch.reshape(-1, batch.shape[-1])).reshape(batch.shape[:2]).cpu().numpy())
    return np.concatenate(scores, axis=0)


def selection_summary(groups: CandidateGroups, scores: np.ndarray) -> dict[str, float]:
    best_index = np.argmin(groups.targets, axis=1)
    score_ranks = np.stack([ranks(row) for row in scores])
    best_ranks = score_ranks[np.arange(len(best_index)), best_index]
    return {
        "tactile_best_top1_rate": float(np.mean(best_ranks == 1)),
        "tactile_best_top5_rate": float(np.mean(best_ranks <= 5)),
        "median_tactile_best_rank": float(np.median(best_ranks)),
    }


def tactile_summary(rows: list[dict[str, str]], prefix: str) -> dict[str, float | int]:
    result: dict[str, float | int] = {"queries": len(rows)}
    for metric in ("tactile_diff_mae", "tactile_ssim", "tactile_mask_iou", "tactile_embedding_distance"):
        values = np.asarray([float(row[f"{prefix}_{metric}"]) for row in rows], dtype=np.float32)
        result[f"mean_{metric}"] = float(values.mean())
        result[f"median_{metric}"] = float(np.median(values))
    return result


def evaluate_queries(
    groups: CandidateGroups,
    matcher_scores: np.ndarray,
    cache_rows: list[dict[str, str]],
    tactile_size: int,
    tactile_threshold: float,
) -> tuple[list[dict[str, str]], dict]:
    diff_cache: dict[str, np.ndarray] = {}
    output_rows: list[dict[str, str]] = []
    current_scores = groups.current_scores
    for index, query in enumerate(groups.rows):
        tactile_best = int(np.argmin(groups.targets[index]))
        current = int(np.argmin(current_scores[index]))
        matcher = int(np.argmin(matcher_scores[index]))
        matcher_rank = int(ranks(matcher_scores[index])[tactile_best])
        selected_indices = {"current": current, "matcher": matcher, "oracle": tactile_best}
        query_diff = tactile_difference(query["touch_path"], diff_cache, tactile_size)
        metrics = {}
        for name, local_index in selected_indices.items():
            cache = cache_rows[int(groups.cache_indices[index, local_index])]
            metrics[name] = tactile_metrics(query_diff, tactile_difference(cache["touch_path"], diff_cache, tactile_size), tactile_threshold)
        output_rows.append({
            "query_record_id": query["record_id"], "query_image_name": query["image_name"], "query_probe": query["probe"],
            "current_cache_record_id": cache_rows[int(groups.cache_indices[index, current])]["record_id"],
            "matcher_cache_record_id": cache_rows[int(groups.cache_indices[index, matcher])]["record_id"],
            "tactile_oracle_cache_record_id": cache_rows[int(groups.cache_indices[index, tactile_best])]["record_id"],
            "current_key_rank_of_tactile_best": str(int(ranks(current_scores[index])[tactile_best])),
            "matcher_rank_of_tactile_best": str(matcher_rank), "matcher_tactile_best_top5": str(int(matcher_rank <= 5)),
            **{f"{name}_{metric}": f"{value:.6f}" for name, values in metrics.items() for metric, value in values.items()},
        })
    summary = {
        "current_key": tactile_summary(output_rows, "current"),
        "tactile_matcher": tactile_summary(output_rows, "matcher"),
        "tactile_oracle_within_geometry_topk": tactile_summary(output_rows, "oracle"),
    }
    return output_rows, summary


def train(config_path: str, section: str) -> dict:
    cfg = load_config(config_path)[section]
    seed = int(cfg.get("seed", 20260725))
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = read_csv_rows(project_path(cfg["samples_csv"]))
    final_min_record = int(cfg.get("final_holdout_min_record", 950))
    forbidden = sorted({row["record_id"] for row in rows if record_number(row["record_id"]) >= final_min_record})
    if forbidden:
        raise RuntimeError(f"Refusing to access final-holdout records: {forbidden[:5]}")
    train_rows = [row for row in rows if row["dataset_split"] == "train"]
    val_rows = [row for row in rows if row["dataset_split"] == "val"]
    if not train_rows or not val_rows:
        raise RuntimeError("Need fixed train and validation rows.")
    crop_size, tactile_size = int(cfg.get("cache_crop_size", 48)), int(cfg.get("tactile_size", 96))
    filter_k = min(int(cfg.get("geometry_filter_k", 32)), len(train_rows))
    geometry_weight, visual_weight = float(cfg.get("geometry_weight", 1.0)), float(cfg.get("visual_weight", 1.0))
    cache_geometry = np.stack([motion_geometry_feature(row, float(row["target_tip_x"]), float(row["target_tip_y"])) for row in train_rows])
    cache_geometry_z, geometry_mean, geometry_std = standardize(cache_geometry, cache_geometry)
    cache_visual = np.stack([visual_patch_feature(row["vision_path"], float(row["target_tip_x"]), float(row["target_tip_y"]), crop_size) for row in train_rows])
    cache_visual_z, visual_mean, visual_std = standardize(cache_visual, cache_visual)
    diff_cache: dict[str, np.ndarray] = {}
    cache_tactile_embeddings = np.stack([tactile_embedding(tactile_difference(row["touch_path"], diff_cache, tactile_size)) for row in train_rows])
    diff_cache.clear()
    train_groups = build_groups(
        train_rows, train_rows, cache_geometry_z, cache_visual_z, cache_tactile_embeddings, geometry_mean, geometry_std,
        visual_mean, visual_std, crop_size, tactile_size, filter_k, geometry_weight, visual_weight, exclude_same_record=True,
    )
    val_groups = build_groups(
        val_rows, train_rows, cache_geometry_z, cache_visual_z, cache_tactile_embeddings, geometry_mean, geometry_std,
        visual_mean, visual_std, crop_size, tactile_size, filter_k, geometry_weight, visual_weight, exclude_same_record=False,
    )
    feature_mean = train_groups.features.reshape(-1, train_groups.features.shape[-1]).mean(axis=0)
    feature_std = train_groups.features.reshape(-1, train_groups.features.shape[-1]).std(axis=0)
    feature_std[feature_std < 1e-6] = 1.0
    train_groups.features = (train_groups.features - feature_mean) / feature_std
    val_groups.features = (val_groups.features - feature_mean) / feature_std
    model = LocalCacheMatcher(train_groups.features.shape[-1], int(cfg.get("hidden_size", 128)), float(cfg.get("dropout", 0.1))).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("learning_rate", 0.001)), weight_decay=float(cfg.get("weight_decay", 1e-4)))
    batch_size, epochs = int(cfg.get("batch_size", 64)), int(cfg.get("epochs", 80))
    target_temperature = float(cfg.get("target_temperature", 0.02))
    listwise_weight = float(cfg.get("listwise_weight", 0.5))
    target_std = max(float(train_groups.targets.std()), 1e-6)
    best_top1, best_epoch, stale = -1.0, 0, 0
    history: list[dict[str, float | int]] = []
    checkpoint_dir = project_path(cfg["checkpoint_dir"])
    ensure_dir(checkpoint_dir)
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(train_groups.rows))
        losses = []
        for start in range(0, len(order), batch_size):
            index = order[start:start + batch_size].numpy()
            features = torch.from_numpy(train_groups.features[index]).to(device)
            targets = torch.from_numpy(train_groups.targets[index]).to(device)
            predictions = model(features.reshape(-1, features.shape[-1])).reshape(features.shape[:2])
            regression = torch.nn.functional.smooth_l1_loss((predictions - targets) / target_std, torch.zeros_like(predictions))
            target_probabilities = torch.softmax(-targets / target_temperature, dim=1)
            listwise = -(target_probabilities * torch.log_softmax(-predictions / target_temperature, dim=1)).sum(dim=1).mean()
            loss = regression + listwise_weight * listwise
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_scores = prediction_scores(model, val_groups.features, device, batch_size)
        val_selection = selection_summary(val_groups, val_scores)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), **val_selection})
        if val_selection["tactile_best_top1_rate"] > best_top1 + 1e-6:
            best_top1, best_epoch, stale = val_selection["tactile_best_top1_rate"], epoch, 0
            torch.save({
                "model_state": model.state_dict(), "input_dim": train_groups.features.shape[-1], "feature_mean": feature_mean,
                "feature_std": feature_std, "geometry_mean": geometry_mean, "geometry_std": geometry_std,
                "visual_mean": visual_mean, "visual_std": visual_std, "config_section": section,
            }, checkpoint_dir / "best.pt")
        else:
            stale += 1
        if stale >= int(cfg.get("early_stopping_patience", 12)):
            break
    checkpoint = torch.load(checkpoint_dir / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    val_scores = prediction_scores(model, val_groups.features, device, batch_size)
    train_scores = prediction_scores(model, train_groups.features, device, batch_size)
    query_rows, tactile_results = evaluate_queries(val_groups, val_scores, train_rows, tactile_size, float(cfg.get("tactile_mask_threshold", 0.04)))
    summary = {
        "mode": "validation_only_tactile_aware_local_cache_matcher", "device": str(device), "cache_size": len(train_rows),
        "train_queries": len(train_groups.rows), "validation_queries": len(val_groups.rows), "filter_k": filter_k,
        "feature_dim": int(train_groups.features.shape[-1]), "best_epoch": best_epoch, "epochs_ran": len(history),
        "train_selection": selection_summary(train_groups, train_scores), "validation_selection": selection_summary(val_groups, val_scores),
        "validation_tactile_metrics": tactile_results, "final_holdout_min_record": final_min_record,
        "supervision_note": "Tactile difference embeddings supervise training only; the deployed matcher input contains local RGB and geometry/motion features only.",
        "checkpoint": str(checkpoint_dir / "best.pt"),
    }
    write_csv_rows(project_path(cfg["query_output_csv"]), query_rows, QUERY_FIELDS)
    write_json(project_path(cfg["metrics_json"]), {**summary, "history": history})
    print(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a tactile-aware local visual/geometry cache matcher.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--section", default="tactile_cache_matcher_phase35_v3")
    args = parser.parse_args()
    train(args.config, args.section)


if __name__ == "__main__":
    main()
