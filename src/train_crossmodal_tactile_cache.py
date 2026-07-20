from __future__ import annotations

import argparse
import math
import random
from collections import defaultdict

import numpy as np
import torch
from torch import nn

from .build_cache_retrieval import crop_contact_patch, motion_geometry_feature, standardize, visual_patch_feature_from_patch
from .config import load_config, project_path
from .evaluate_oracle_tactile_retrieval import tactile_difference, tactile_embedding, tactile_metrics
from .utils import ensure_dir, read_csv_rows, write_csv_rows, write_json


QUERY_FIELDS = [
    "query_record_id", "query_image_name", "query_probe", "current_cache_record_id", "crossmodal_cache_record_id",
    "tactile_oracle_cache_record_id", "current_key_rank_of_tactile_best", "crossmodal_rank_of_tactile_best",
    "crossmodal_tactile_best_top5", "current_tactile_diff_mae", "crossmodal_tactile_diff_mae", "oracle_tactile_diff_mae",
    "current_tactile_ssim", "crossmodal_tactile_ssim", "oracle_tactile_ssim", "current_tactile_mask_iou",
    "crossmodal_tactile_mask_iou", "oracle_tactile_mask_iou", "current_tactile_embedding_distance",
    "crossmodal_tactile_embedding_distance", "oracle_tactile_embedding_distance",
    "current_tactile_area_delta", "crossmodal_tactile_area_delta", "oracle_tactile_area_delta",
    "current_tactile_centroid_distance", "crossmodal_tactile_centroid_distance", "oracle_tactile_centroid_distance",
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


class ImageEncoder(nn.Module):
    def __init__(self, channels: int, output_dim: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 96, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.projection = nn.Linear(96, output_dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.projection(self.features(image).flatten(1))


class VisualGeometryEncoder(nn.Module):
    def __init__(self, geometry_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.visual = ImageEncoder(3, latent_dim)
        self.geometry = nn.Sequential(nn.Linear(geometry_dim, 64), nn.ReLU(inplace=True), nn.Linear(64, latent_dim))
        self.fuse = nn.Linear(latent_dim * 2, latent_dim)

    def forward(self, image: torch.Tensor, geometry: torch.Tensor) -> torch.Tensor:
        latent = self.fuse(torch.cat([self.visual(image), self.geometry(geometry)], dim=1))
        return torch.nn.functional.normalize(latent, dim=1)


class TactileEncoder(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.encoder = ImageEncoder(3, latent_dim)

    def forward(self, tactile: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(self.encoder(tactile), dim=1)


def tensor_images(images: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.transpose(images, (0, 3, 1, 2)).astype(np.float32, copy=False))


def load_visual_patches(rows: list[dict[str, str]], crop_size: int, geometry_mean: np.ndarray, geometry_std: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    patches, geometry = [], []
    for row in rows:
        x, y = float(row["target_tip_x"]), float(row["target_tip_y"])
        patches.append(crop_contact_patch(row["vision_path"], x, y, crop_size))
        geometry.append((motion_geometry_feature(row, x, y) - geometry_mean) / geometry_std)
    return np.stack(patches).astype(np.float32), np.stack(geometry).astype(np.float32)


def encode_visual(model: VisualGeometryEncoder, patches: np.ndarray, geometry: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(patches), batch_size):
            image = tensor_images(patches[start:start + batch_size]).to(device)
            geo = torch.from_numpy(geometry[start:start + batch_size]).to(device)
            outputs.append(model(image, geo).cpu().numpy())
    return np.concatenate(outputs)


def encode_tactile(model: TactileEncoder, images: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            outputs.append(model(tensor_images(images[start:start + batch_size]).to(device)).cpu().numpy())
    return np.concatenate(outputs)


def selection_summary(targets: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    best = np.argmin(targets, axis=1)
    score_ranks = np.stack([ranks(-row) for row in scores])
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
    query_rows: list[dict[str, str]],
    query_visual: np.ndarray,
    query_geometry: np.ndarray,
    query_tactile_embeddings: np.ndarray,
    query_tactile_images: np.ndarray,
    cache_rows: list[dict[str, str]],
    cache_geometry: np.ndarray,
    cache_hand_visual: np.ndarray,
    query_hand_visual: np.ndarray,
    cache_tactile_embeddings: np.ndarray,
    cache_crossmodal_embeddings: np.ndarray,
    geometry_filter_k: int,
    tactile_size: int,
    tactile_threshold: float,
) -> tuple[list[dict[str, str]], dict, dict[str, float]]:
    output_rows: list[dict[str, str]] = []
    targets, current_scores, crossmodal_scores = [], [], []
    diff_cache: dict[str, np.ndarray] = {}
    for index, query in enumerate(query_rows):
        geometry_distances = np.linalg.norm(cache_geometry - query_geometry[index][None], axis=1)
        shortlist = np.argpartition(geometry_distances, geometry_filter_k - 1)[:geometry_filter_k]
        shortlist = shortlist[np.argsort(geometry_distances[shortlist], kind="stable")]
        target = np.linalg.norm(cache_tactile_embeddings[shortlist] - query_tactile_embeddings[index][None], axis=1)
        visual_distances = np.linalg.norm(cache_hand_visual[shortlist] - query_hand_visual[index][None], axis=1)
        current = geometry_distances[shortlist] / math.sqrt(cache_geometry.shape[1]) + visual_distances / math.sqrt(cache_hand_visual.shape[1])
        crossmodal = query_visual[index] @ cache_crossmodal_embeddings[shortlist].T
        targets.append(target)
        current_scores.append(-current)
        crossmodal_scores.append(crossmodal)
        tactile_best = int(np.argmin(target))
        current_choice = int(np.argmax(-current))
        crossmodal_choice = int(np.argmax(crossmodal))
        crossmodal_rank = int(ranks(-crossmodal)[tactile_best])
        query_diff = query_tactile_images[index]
        selected = {"current": current_choice, "crossmodal": crossmodal_choice, "oracle": tactile_best}
        metrics = {}
        for name, local_index in selected.items():
            cache = cache_rows[int(shortlist[local_index])]
            metrics[name] = tactile_metrics(query_diff, tactile_difference(cache["touch_path"], diff_cache, tactile_size), tactile_threshold)
        output_rows.append({
            "query_record_id": query["record_id"], "query_image_name": query["image_name"], "query_probe": query["probe"],
            "current_cache_record_id": cache_rows[int(shortlist[current_choice])]["record_id"],
            "crossmodal_cache_record_id": cache_rows[int(shortlist[crossmodal_choice])]["record_id"],
            "tactile_oracle_cache_record_id": cache_rows[int(shortlist[tactile_best])]["record_id"],
            "current_key_rank_of_tactile_best": str(int(ranks(current)[tactile_best])),
            "crossmodal_rank_of_tactile_best": str(crossmodal_rank), "crossmodal_tactile_best_top5": str(int(crossmodal_rank <= 5)),
            **{f"{name}_{metric}": f"{value:.6f}" for name, values in metrics.items() for metric, value in values.items()},
        })
    target_array, current_array, crossmodal_array = np.stack(targets), np.stack(current_scores), np.stack(crossmodal_scores)
    summary = {
        "current_geometry_key": metric_summary(output_rows, "current"),
        "crossmodal_matcher": metric_summary(output_rows, "crossmodal"),
        "tactile_oracle_within_geometry_topk": metric_summary(output_rows, "oracle"),
    }
    return output_rows, summary, {"current": selection_summary(target_array, current_array), "crossmodal": selection_summary(target_array, crossmodal_array)}


def train(config_path: str, section: str) -> dict:
    cfg = load_config(config_path)[section]
    seed = int(cfg.get("seed", 20260726))
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = read_csv_rows(project_path(cfg["samples_csv"]))
    final_min_record = int(cfg.get("final_holdout_min_record", 950))
    forbidden = sorted({row["record_id"] for row in rows if record_number(row["record_id"]) >= final_min_record})
    if forbidden:
        raise RuntimeError(f"Refusing to access final-holdout records: {forbidden[:5]}")
    cache_rows = [row for row in rows if row["dataset_split"] == "train"]
    val_rows = [row for row in rows if row["dataset_split"] == "val"]
    crop_size, tactile_size = int(cfg.get("cache_crop_size", 48)), int(cfg.get("tactile_size", 96))
    overlap = {row["record_id"] for row in cache_rows} & {row["record_id"] for row in val_rows}
    if overlap:
        raise RuntimeError(f"Train cache and validation query records must be disjoint: {sorted(overlap)[:5]}")
    filter_k = min(int(cfg.get("geometry_filter_k", 32)), len(cache_rows))
    raw_cache_geometry = np.stack([motion_geometry_feature(row, float(row["target_tip_x"]), float(row["target_tip_y"])) for row in cache_rows])
    cache_geometry, geometry_mean, geometry_std = standardize(raw_cache_geometry, raw_cache_geometry)
    cache_patches, _ = load_visual_patches(cache_rows, crop_size, geometry_mean, geometry_std)
    val_patches, val_geometry = load_visual_patches(val_rows, crop_size, geometry_mean, geometry_std)
    cache_geometry = cache_geometry.astype(np.float32)
    record_to_cache_indices: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(cache_rows):
        record_to_cache_indices[row["record_id"]].append(index)
    train_record_ids = sorted(record_to_cache_indices)
    record_to_tactile_index = {record_id: index for index, record_id in enumerate(train_record_ids)}
    diff_cache: dict[str, np.ndarray] = {}
    train_tactile_images = np.stack([tactile_difference(cache_rows[record_to_cache_indices[record_id][0]]["touch_path"], diff_cache, tactile_size) for record_id in train_record_ids]).astype(np.float32)
    cache_tactile_indices = np.asarray([record_to_tactile_index[row["record_id"]] for row in cache_rows], dtype=np.int32)
    train_tactile_embeddings = np.stack([tactile_embedding(image) for image in train_tactile_images]).astype(np.float32)
    cache_tactile_embeddings = train_tactile_embeddings[cache_tactile_indices]
    val_tactile_images = np.stack([tactile_difference(row["touch_path"], diff_cache, tactile_size) for row in val_rows]).astype(np.float32)
    val_tactile_embeddings = np.stack([tactile_embedding(image) for image in val_tactile_images]).astype(np.float32)
    diff_cache.clear()
    cache_hand_visual_raw = np.stack([visual_patch_feature_from_patch(patch) for patch in cache_patches])
    cache_hand_visual, hand_visual_mean, hand_visual_std = standardize(cache_hand_visual_raw, cache_hand_visual_raw)
    val_hand_visual_raw = np.stack([visual_patch_feature_from_patch(patch) for patch in val_patches])
    val_hand_visual = (val_hand_visual_raw - hand_visual_mean) / hand_visual_std
    latent_dim = int(cfg.get("latent_dim", 64))
    visual_model = VisualGeometryEncoder(cache_geometry.shape[1], latent_dim).to(device)
    tactile_model = TactileEncoder(latent_dim).to(device)
    optimizer = torch.optim.AdamW(
        list(visual_model.parameters()) + list(tactile_model.parameters()),
        lr=float(cfg.get("learning_rate", 0.001)), weight_decay=float(cfg.get("weight_decay", 1e-4)),
    )
    train_indices_by_record = {record_id: np.asarray(indices, dtype=np.int32) for record_id, indices in record_to_cache_indices.items()}
    batch_size, epochs = int(cfg.get("batch_size", 64)), int(cfg.get("epochs", 100))
    temperature = float(cfg.get("temperature", 0.07))
    validation_interval = int(cfg.get("validation_interval", 2))
    checkpoint_dir = project_path(cfg["checkpoint_dir"])
    ensure_dir(checkpoint_dir)
    best_top1, best_epoch, stale, history = -1.0, 0, 0, []
    for epoch in range(1, epochs + 1):
        visual_model.train()
        tactile_model.train()
        random.shuffle(train_record_ids)
        losses = []
        for start in range(0, len(train_record_ids), batch_size):
            record_batch = train_record_ids[start:start + batch_size]
            sample_indices = [int(random.choice(train_indices_by_record[record_id])) for record_id in record_batch]
            tactile_indices = [record_to_tactile_index[record_id] for record_id in record_batch]
            visual = visual_model(tensor_images(cache_patches[sample_indices]).to(device), torch.from_numpy(cache_geometry[sample_indices]).to(device))
            tactile = tactile_model(tensor_images(train_tactile_images[tactile_indices]).to(device))
            logits = visual @ tactile.T / temperature
            labels = torch.arange(len(record_batch), device=device)
            loss = 0.5 * (torch.nn.functional.cross_entropy(logits, labels) + torch.nn.functional.cross_entropy(logits.T, labels))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if epoch % validation_interval:
            continue
        val_visual = encode_visual(visual_model, val_patches, val_geometry, device, batch_size)
        cache_crossmodal = encode_tactile(tactile_model, train_tactile_images, device, batch_size)[cache_tactile_indices]
        _, _, selection = evaluate(
            val_rows, val_visual, val_geometry, val_tactile_embeddings, val_tactile_images, cache_rows, cache_geometry, cache_hand_visual, val_hand_visual,
            cache_tactile_embeddings, cache_crossmodal, filter_k, tactile_size, float(cfg.get("tactile_mask_threshold", 0.04)),
        )
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), **{f"val_{key}_{metric}": value for key, values in selection.items() for metric, value in values.items()}})
        top1 = selection["crossmodal"]["tactile_best_top1_rate"]
        if top1 > best_top1 + 1e-6:
            best_top1, best_epoch, stale = top1, epoch, 0
            torch.save({
                "visual_model": visual_model.state_dict(), "tactile_model": tactile_model.state_dict(), "latent_dim": latent_dim,
                "geometry_mean": geometry_mean, "geometry_std": geometry_std, "config_section": section,
            }, checkpoint_dir / "best.pt")
        else:
            stale += 1
        if stale >= int(cfg.get("early_stopping_patience", 12)):
            break
    checkpoint = torch.load(checkpoint_dir / "best.pt", map_location=device, weights_only=False)
    visual_model.load_state_dict(checkpoint["visual_model"])
    tactile_model.load_state_dict(checkpoint["tactile_model"])
    val_visual = encode_visual(visual_model, val_patches, val_geometry, device, batch_size)
    cache_crossmodal = encode_tactile(tactile_model, train_tactile_images, device, batch_size)[cache_tactile_indices]
    query_rows, tactile_summary, selection = evaluate(
        val_rows, val_visual, val_geometry, val_tactile_embeddings, val_tactile_images, cache_rows, cache_geometry, cache_hand_visual, val_hand_visual,
        cache_tactile_embeddings, cache_crossmodal, filter_k, tactile_size, float(cfg.get("tactile_mask_threshold", 0.04)),
    )
    summary = {
        "mode": "validation_only_crossmodal_visual_to_tactile_cache", "device": str(device), "cache_size": len(cache_rows),
        "train_records": len(train_record_ids), "validation_queries": len(val_rows), "geometry_filter_k": filter_k,
        "best_epoch": best_epoch, "epochs_ran": (history[-1]["epoch"] if history else 0), "validation_selection": selection,
        "validation_tactile_metrics": tactile_summary, "final_holdout_min_record": final_min_record,
        "training_note": "Each contrastive batch contains one random pre-contact view per train record, so identical tactile contact frames are never treated as negatives.",
        "checkpoint": str(checkpoint_dir / "best.pt"), "history": history,
    }
    write_csv_rows(project_path(cfg["query_output_csv"]), query_rows, QUERY_FIELDS)
    write_json(project_path(cfg["metrics_json"]), summary)
    print({key: value for key, value in summary.items() if key != "history"})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a cross-modal visual-to-tactile local cache encoder.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--section", default="crossmodal_tactile_cache_phase35_v3")
    args = parser.parse_args()
    train(args.config, args.section)


if __name__ == "__main__":
    main()
