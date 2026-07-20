from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .build_cache_retrieval import crop_contact_patch, motion_geometry_feature, standardize, visual_patch_feature_from_patch
from .config import load_config, project_path
from .evaluate_oracle_tactile_retrieval import tactile_difference, tactile_embedding, tactile_metrics
from .utils import ensure_dir, read_csv_rows, write_csv_rows, write_json


QUERY_FIELDS = [
    "query_record_id", "query_image_name", "query_probe", "current_cache_record_id", "soft_ranker_cache_record_id",
    "tactile_oracle_cache_record_id", "current_key_rank_of_tactile_best", "soft_ranker_rank_of_tactile_best",
    "soft_ranker_tactile_best_top5", "current_tactile_diff_mae", "soft_ranker_tactile_diff_mae", "oracle_tactile_diff_mae",
    "current_tactile_ssim", "soft_ranker_tactile_ssim", "oracle_tactile_ssim", "current_tactile_mask_iou",
    "soft_ranker_tactile_mask_iou", "oracle_tactile_mask_iou", "current_tactile_area_delta", "soft_ranker_tactile_area_delta",
    "oracle_tactile_area_delta", "current_tactile_centroid_distance", "soft_ranker_tactile_centroid_distance",
    "oracle_tactile_centroid_distance", "current_tactile_embedding_distance", "soft_ranker_tactile_embedding_distance",
    "oracle_tactile_embedding_distance",
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


def image_tensor(images: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.transpose(images, (0, 3, 1, 2)).astype(np.float32, copy=False))


@dataclass
class CandidateGroups:
    candidates: np.ndarray
    targets: np.ndarray
    current_scores: np.ndarray


class PatchEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 24, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(24, 48, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(48, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.layers(images).flatten(1)


class SoftTactileRanker(nn.Module):
    def __init__(self, geometry_dim: int, dropout: float) -> None:
        super().__init__()
        self.encoder = PatchEncoder()
        feature_dim = 64 * 3 + geometry_dim * 3 + 1
        self.head = nn.Sequential(
            nn.Linear(feature_dim, 128), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(inplace=True), nn.Linear(64, 1),
        )

    def forward(self, query_images: torch.Tensor, candidate_images: torch.Tensor, query_geometry: torch.Tensor, candidate_geometry: torch.Tensor, current_scores: torch.Tensor) -> torch.Tensor:
        batch, candidates = candidate_images.shape[:2]
        query_feature = self.encoder(query_images)
        candidate_feature = self.encoder(candidate_images.reshape(batch * candidates, *candidate_images.shape[2:])).reshape(batch, candidates, -1)
        query_feature = query_feature[:, None].expand(-1, candidates, -1)
        query_geometry = query_geometry[:, None].expand(-1, candidates, -1)
        features = torch.cat(
            [query_feature, candidate_feature, torch.abs(query_feature - candidate_feature), query_geometry,
             candidate_geometry, torch.abs(query_geometry - candidate_geometry), current_scores[:, :, None]],
            dim=2,
        )
        return self.head(features).squeeze(-1)


def load_patches(rows: list[dict[str, str]], crop_size: int, geometry_mean: np.ndarray, geometry_std: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    patches, geometry, hand_visual = [], [], []
    for row in rows:
        x, y = float(row["target_tip_x"]), float(row["target_tip_y"])
        patch = crop_contact_patch(row["vision_path"], x, y, crop_size)
        patches.append(patch)
        geometry.append((motion_geometry_feature(row, x, y) - geometry_mean) / geometry_std)
        hand_visual.append(visual_patch_feature_from_patch(patch))
    return np.stack(patches).astype(np.float32), np.stack(geometry).astype(np.float32), np.stack(hand_visual).astype(np.float32)


def build_groups(
    query_rows: list[dict[str, str]],
    query_geometry: np.ndarray,
    query_hand_visual: np.ndarray,
    query_tactile_embeddings: np.ndarray,
    cache_rows: list[dict[str, str]],
    cache_geometry: np.ndarray,
    cache_hand_visual: np.ndarray,
    cache_tactile_embeddings: np.ndarray,
    filter_k: int,
    exclude_same_record: bool,
) -> CandidateGroups:
    candidates, targets, current_scores = [], [], []
    for index, query in enumerate(query_rows):
        geometry_distances = np.linalg.norm(cache_geometry - query_geometry[index][None], axis=1)
        allowed = np.ones(len(cache_rows), dtype=bool)
        if exclude_same_record:
            allowed = np.asarray([row["record_id"] != query["record_id"] for row in cache_rows])
        allowed_indices = np.flatnonzero(allowed)
        local_k = min(filter_k, len(allowed_indices))
        shortlist = allowed_indices[np.argpartition(geometry_distances[allowed_indices], local_k - 1)[:local_k]]
        shortlist = shortlist[np.argsort(geometry_distances[shortlist], kind="stable")]
        visual_distances = np.linalg.norm(cache_hand_visual[shortlist] - query_hand_visual[index][None], axis=1)
        current = geometry_distances[shortlist] / math.sqrt(cache_geometry.shape[1]) + visual_distances / math.sqrt(cache_hand_visual.shape[1])
        target = np.linalg.norm(cache_tactile_embeddings[shortlist] - query_tactile_embeddings[index][None], axis=1)
        candidates.append(shortlist.astype(np.int32))
        targets.append(target.astype(np.float32))
        current_scores.append(current.astype(np.float32))
    return CandidateGroups(np.stack(candidates), np.stack(targets), np.stack(current_scores))


def predict(model: SoftTactileRanker, groups: CandidateGroups, query_patches: np.ndarray, query_geometry: np.ndarray, cache_patches: np.ndarray, cache_geometry: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(query_patches), batch_size):
            end = start + batch_size
            indices = groups.candidates[start:end]
            query_image = image_tensor(query_patches[start:end]).to(device)
            candidate_image = image_tensor(cache_patches[indices].reshape(-1, *cache_patches.shape[1:])).reshape(len(indices), indices.shape[1], 3, *cache_patches.shape[1:3]).to(device)
            query_geo = torch.from_numpy(query_geometry[start:end]).to(device)
            candidate_geo = torch.from_numpy(cache_geometry[indices]).to(device)
            current = torch.from_numpy(groups.current_scores[start:end]).to(device)
            outputs.append(model(query_image, candidate_image, query_geo, candidate_geo, current).cpu().numpy())
    return np.concatenate(outputs)


def selection_summary(targets: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    best = np.argmin(targets, axis=1)
    score_ranks = np.stack([ranks(row) for row in scores])
    best_ranks = score_ranks[np.arange(len(best)), best]
    return {
        "tactile_best_top1_rate": float(np.mean(best_ranks == 1)),
        "tactile_best_top5_rate": float(np.mean(best_ranks <= 5)),
        "median_tactile_best_rank": float(np.median(best_ranks)),
    }


def metric_summary(rows: list[dict[str, str]], prefix: str) -> dict[str, float | int]:
    result: dict[str, float | int] = {"queries": len(rows)}
    for metric in ("tactile_diff_mae", "tactile_ssim", "tactile_mask_iou", "tactile_embedding_distance"):
        values = np.asarray([float(row[f"{prefix}_{metric}"]) for row in rows], dtype=np.float32)
        result[f"mean_{metric}"] = float(values.mean())
        result[f"median_{metric}"] = float(np.median(values))
    return result


def evaluate(
    groups: CandidateGroups,
    scores: np.ndarray,
    query_rows: list[dict[str, str]],
    query_tactile_images: np.ndarray,
    cache_rows: list[dict[str, str]],
    tactile_size: int,
    tactile_threshold: float,
) -> tuple[list[dict[str, str]], dict, dict[str, float]]:
    output_rows: list[dict[str, str]] = []
    diff_cache: dict[str, np.ndarray] = {}
    for index, query in enumerate(query_rows):
        tactile_best = int(np.argmin(groups.targets[index]))
        current = int(np.argmin(groups.current_scores[index]))
        ranker = int(np.argmin(scores[index]))
        ranker_rank = int(ranks(scores[index])[tactile_best])
        selections = {"current": current, "soft_ranker": ranker, "oracle": tactile_best}
        metrics = {}
        for name, local_index in selections.items():
            cache = cache_rows[int(groups.candidates[index, local_index])]
            metrics[name] = tactile_metrics(query_tactile_images[index], tactile_difference(cache["touch_path"], diff_cache, tactile_size), tactile_threshold)
        output_rows.append({
            "query_record_id": query["record_id"], "query_image_name": query["image_name"], "query_probe": query["probe"],
            "current_cache_record_id": cache_rows[int(groups.candidates[index, current])]["record_id"],
            "soft_ranker_cache_record_id": cache_rows[int(groups.candidates[index, ranker])]["record_id"],
            "tactile_oracle_cache_record_id": cache_rows[int(groups.candidates[index, tactile_best])]["record_id"],
            "current_key_rank_of_tactile_best": str(int(ranks(groups.current_scores[index])[tactile_best])),
            "soft_ranker_rank_of_tactile_best": str(ranker_rank), "soft_ranker_tactile_best_top5": str(int(ranker_rank <= 5)),
            **{f"{name}_{metric}": f"{value:.6f}" for name, values in metrics.items() for metric, value in values.items()},
        })
    return output_rows, {
        "current_key": metric_summary(output_rows, "current"), "soft_tactile_ranker": metric_summary(output_rows, "soft_ranker"),
        "tactile_oracle_within_geometry_topk": metric_summary(output_rows, "oracle"),
    }, {
        "current": selection_summary(groups.targets, groups.current_scores), "soft_ranker": selection_summary(groups.targets, scores),
    }


def train(config_path: str, section: str) -> dict:
    cfg = load_config(config_path)[section]
    set_seed(int(cfg.get("seed", 20260727)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = read_csv_rows(project_path(cfg["samples_csv"]))
    final_min_record = int(cfg.get("final_holdout_min_record", 950))
    forbidden = sorted({row["record_id"] for row in rows if record_number(row["record_id"]) >= final_min_record})
    if forbidden:
        raise RuntimeError(f"Refusing to access final-holdout records: {forbidden[:5]}")
    cache_rows = [row for row in rows if row["dataset_split"] == "train"]
    val_rows = [row for row in rows if row["dataset_split"] == "val"]
    overlap = {row["record_id"] for row in cache_rows} & {row["record_id"] for row in val_rows}
    if overlap:
        raise RuntimeError(f"Train and validation records overlap: {sorted(overlap)[:5]}")
    crop_size, tactile_size = int(cfg.get("cache_crop_size", 48)), int(cfg.get("tactile_size", 96))
    filter_k = min(int(cfg.get("geometry_filter_k", 32)), len(cache_rows))
    raw_cache_geometry = np.stack([motion_geometry_feature(row, float(row["target_tip_x"]), float(row["target_tip_y"])) for row in cache_rows])
    cache_geometry, geometry_mean, geometry_std = standardize(raw_cache_geometry, raw_cache_geometry)
    cache_patches, _, cache_hand_visual_raw = load_patches(cache_rows, crop_size, geometry_mean, geometry_std)
    val_patches, val_geometry, val_hand_visual_raw = load_patches(val_rows, crop_size, geometry_mean, geometry_std)
    cache_hand_visual, hand_mean, hand_std = standardize(cache_hand_visual_raw, cache_hand_visual_raw)
    val_hand_visual = (val_hand_visual_raw - hand_mean) / hand_std
    cache_geometry = cache_geometry.astype(np.float32)
    record_to_tactile: dict[str, np.ndarray] = {}
    diff_cache: dict[str, np.ndarray] = {}
    for row in cache_rows:
        if row["record_id"] not in record_to_tactile:
            record_to_tactile[row["record_id"]] = tactile_difference(row["touch_path"], diff_cache, tactile_size)
    cache_tactile_embeddings = np.stack([tactile_embedding(record_to_tactile[row["record_id"]]) for row in cache_rows]).astype(np.float32)
    train_tactile_embeddings = cache_tactile_embeddings.copy()
    val_tactile_images = np.stack([tactile_difference(row["touch_path"], diff_cache, tactile_size) for row in val_rows]).astype(np.float32)
    val_tactile_embeddings = np.stack([tactile_embedding(image) for image in val_tactile_images]).astype(np.float32)
    diff_cache.clear()
    train_groups = build_groups(cache_rows, cache_geometry, cache_hand_visual, train_tactile_embeddings, cache_rows, cache_geometry, cache_hand_visual, cache_tactile_embeddings, filter_k, True)
    val_groups = build_groups(val_rows, val_geometry, val_hand_visual, val_tactile_embeddings, cache_rows, cache_geometry, cache_hand_visual, cache_tactile_embeddings, filter_k, False)
    model = SoftTactileRanker(cache_geometry.shape[1], float(cfg.get("dropout", 0.1))).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("learning_rate", 0.001)), weight_decay=float(cfg.get("weight_decay", 1e-4)))
    batch_size, epochs = int(cfg.get("batch_size", 16)), int(cfg.get("epochs", 60))
    target_std = max(float(train_groups.targets.std()), 1e-6)
    target_temperature = float(cfg.get("target_temperature", 0.02))
    listwise_weight = float(cfg.get("listwise_weight", 1.0))
    checkpoint_dir = project_path(cfg["checkpoint_dir"])
    ensure_dir(checkpoint_dir)
    best_top1, best_epoch, stale, history = -1.0, 0, 0, []
    for epoch in range(1, epochs + 1):
        model.train()
        order = np.random.permutation(len(cache_rows))
        losses = []
        for start in range(0, len(order), batch_size):
            query_indices = order[start:start + batch_size]
            candidates = train_groups.candidates[query_indices]
            query_image = image_tensor(cache_patches[query_indices]).to(device)
            candidate_image = image_tensor(cache_patches[candidates].reshape(-1, *cache_patches.shape[1:])).reshape(len(query_indices), filter_k, 3, *cache_patches.shape[1:3]).to(device)
            query_geo = torch.from_numpy(cache_geometry[query_indices]).to(device)
            candidate_geo = torch.from_numpy(cache_geometry[candidates]).to(device)
            current = torch.from_numpy(train_groups.current_scores[query_indices]).to(device)
            targets = torch.from_numpy(train_groups.targets[query_indices]).to(device)
            predictions = model(query_image, candidate_image, query_geo, candidate_geo, current)
            regression = torch.nn.functional.smooth_l1_loss((predictions - targets) / target_std, torch.zeros_like(predictions))
            target_distribution = torch.softmax(-targets / target_temperature, dim=1)
            listwise = -(target_distribution * torch.log_softmax(-predictions / target_temperature, dim=1)).sum(dim=1).mean()
            loss = regression + listwise_weight * listwise
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_scores = predict(model, val_groups, val_patches, val_geometry, cache_patches, cache_geometry, device, batch_size)
        selection = selection_summary(val_groups.targets, val_scores)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), **selection})
        top1 = selection["tactile_best_top1_rate"]
        if top1 > best_top1 + 1e-6:
            best_top1, best_epoch, stale = top1, epoch, 0
            torch.save({"model_state": model.state_dict(), "geometry_mean": geometry_mean, "geometry_std": geometry_std, "config_section": section}, checkpoint_dir / "best.pt")
        else:
            stale += 1
        if stale >= int(cfg.get("early_stopping_patience", 10)):
            break
    checkpoint = torch.load(checkpoint_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    val_scores = predict(model, val_groups, val_patches, val_geometry, cache_patches, cache_geometry, device, batch_size)
    query_rows, tactile_summary, selection = evaluate(val_groups, val_scores, val_rows, val_tactile_images, cache_rows, tactile_size, float(cfg.get("tactile_mask_threshold", 0.04)))
    summary = {
        "mode": "validation_only_soft_supervised_local_tactile_ranker", "device": str(device), "cache_size": len(cache_rows),
        "train_queries": len(cache_rows), "validation_queries": len(val_rows), "geometry_filter_k": filter_k,
        "best_epoch": best_epoch, "epochs_ran": len(history), "validation_selection": selection,
        "validation_tactile_metrics": tactile_summary, "final_holdout_min_record": final_min_record,
        "supervision_note": "The ranker sees only query/cache local RGB patches and geometry. Tactile embedding distance creates a soft multi-positive listwise training target.",
        "checkpoint": str(checkpoint_dir / "best.pt"), "history": history,
    }
    write_csv_rows(project_path(cfg["query_output_csv"]), query_rows, QUERY_FIELDS)
    write_json(project_path(cfg["metrics_json"]), summary)
    print({key: value for key, value in summary.items() if key != "history"})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a soft-supervised local visual tactile cache ranker.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--section", default="soft_tactile_cache_ranker_phase35_v3")
    args = parser.parse_args()
    train(args.config, args.section)


if __name__ == "__main__":
    main()
