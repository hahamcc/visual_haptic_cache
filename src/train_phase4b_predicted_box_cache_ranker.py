from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .build_cache_retrieval import crop_contact_patch, motion_geometry_feature, standardize, visual_patch_feature_from_patch
from .config import load_config, project_path
from .evaluate_oracle_tactile_retrieval import tactile_difference, tactile_embedding, tactile_metrics
from .train_soft_tactile_cache_ranker import CandidateGroups, SoftTactileRanker, image_tensor, predict, ranks
from .utils import ensure_dir, read_csv_rows, write_csv_rows, write_json


METRICS = [
    "tactile_diff_mae", "tactile_ssim", "tactile_mask_iou", "tactile_area_delta",
    "tactile_centroid_distance", "tactile_embedding_distance",
]
QUERY_FIELDS = [
    "query_record_id", "query_image_name", "query_probe", "pred_x", "pred_y",
    "current_cache_record_id", "ranker_cache_record_id", "tactile_oracle_cache_record_id",
    "current_rank_of_tactile_best", "ranker_rank_of_tactile_best",
    *[f"{prefix}_{metric}" for prefix in ("current", "ranker", "oracle") for metric in METRICS],
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def record_number(record_id: str) -> int:
    return int(record_id.rsplit("_", 1)[-1])


def is_final_holdout(row: dict[str, str]) -> bool:
    # Only this fixed split-0 partition is sealed; split 1 validly has rec_01000+.
    return row.get("split") == "0" and 950 <= record_number(row["record_id"]) <= 999


def metric_summary(rows: list[dict[str, str]], prefix: str) -> dict[str, float | int | None]:
    result: dict[str, float | int | None] = {"queries": len(rows)}
    for metric in METRICS:
        values = np.asarray([float(row[f"{prefix}_{metric}"]) for row in rows], dtype=np.float32)
        result[f"mean_{metric}"] = float(values.mean()) if len(values) else None
        result[f"median_{metric}"] = float(np.median(values)) if len(values) else None
    if prefix == "oracle":
        ranks_array = np.ones(len(rows), dtype=np.int32)
    else:
        ranks_array = np.asarray([int(row[f"{prefix}_rank_of_tactile_best"]) for row in rows], dtype=np.int32)
    result["tactile_best_top1_rate"] = float(np.mean(ranks_array == 1)) if len(rows) else None
    result["tactile_best_top5_rate"] = float(np.mean(ranks_array <= 5)) if len(rows) else None
    result["median_tactile_best_rank"] = float(np.median(ranks_array)) if len(rows) else None
    return result


def prediction_map(
    predictions: list[dict[str, str]],
    rows: list[dict[str, str]],
    split: str,
    label: str,
) -> dict[str, dict[str, str]]:
    expected = {row["image_name"] for row in rows if row["dataset_split"] == split}
    selected = [row for row in predictions if row.get("dataset_split") == split and row["image_name"] in expected]
    found = {row["image_name"] for row in selected}
    if len(selected) != len(found) or found != expected:
        raise RuntimeError(f"{label} predictions do not cover {split} exactly: expected={len(expected)} got={len(found)}")
    if any(is_final_holdout(row) for row in selected):
        raise RuntimeError(f"{label} predictions include sealed final-holdout data.")
    return {row["image_name"]: row for row in selected}


def build_groups(
    query_rows: list[dict[str, str]],
    query_geometry: np.ndarray,
    query_hand: np.ndarray,
    query_tactile: list[np.ndarray],
    cache_rows: list[dict[str, str]],
    cache_geometry: np.ndarray,
    cache_hand: np.ndarray,
    cache_tactile_embeddings: np.ndarray,
    query_tactile_embeddings: np.ndarray,
    touch_for_row,
    filter_k: int,
    exclude_same_record: bool,
    target_mode: str,
) -> CandidateGroups:
    candidates, targets, current_scores = [], [], []
    cache_record_ids = np.asarray([row["record_id"] for row in cache_rows])
    for index, query in enumerate(query_rows):
        geometry_distances = np.linalg.norm(cache_geometry - query_geometry[index][None], axis=1)
        allowed = np.ones(len(cache_rows), dtype=bool)
        if exclude_same_record:
            allowed = cache_record_ids != query["record_id"]
        allowed_indices = np.flatnonzero(allowed)
        shortlist = allowed_indices[np.argpartition(geometry_distances[allowed_indices], filter_k - 1)[:filter_k]]
        shortlist = shortlist[np.argsort(geometry_distances[shortlist], kind="stable")]
        visual_distances = np.linalg.norm(cache_hand[shortlist] - query_hand[index][None], axis=1)
        current = geometry_distances[shortlist] / math.sqrt(cache_geometry.shape[1]) + visual_distances / math.sqrt(cache_hand.shape[1])
        if target_mode == "tactile_mae":
            cache_tactile = np.stack([touch_for_row(cache_rows[int(item)]) for item in shortlist])
            target = np.abs(cache_tactile - query_tactile[index][None]).mean(axis=(1, 2, 3))
        elif target_mode == "tactile_embedding":
            target = np.linalg.norm(cache_tactile_embeddings[shortlist] - query_tactile_embeddings[index][None], axis=1)
        else:
            raise ValueError(f"Unsupported target_mode: {target_mode}")
        candidates.append(shortlist.astype(np.int32))
        targets.append(target.astype(np.float32))
        current_scores.append(current.astype(np.float32))
    return CandidateGroups(np.stack(candidates), np.stack(targets), np.stack(current_scores))


def train(config_path: str, section: str) -> dict:
    cfg = load_config(config_path)[section]
    set_seed(int(cfg.get("seed", 20260731)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = read_csv_rows(project_path(cfg["samples_csv"]))
    if any(is_final_holdout(row) for row in rows):
        raise RuntimeError("Refusing to access sealed final-holdout samples.")
    cache_rows = [row for row in rows if row["dataset_split"] == "train"]
    val_rows = [row for row in rows if row["dataset_split"] == "val"]
    if {row["record_id"] for row in cache_rows} & {row["record_id"] for row in val_rows}:
        raise RuntimeError("Train cache and validation query records overlap.")

    train_coordinate_source = str(cfg.get("train_coordinate_source", "predicted"))
    validation_coordinate_source = str(cfg.get("validation_coordinate_source", "predicted"))
    target_mode = str(cfg.get("target_mode", "tactile_mae"))
    train_prediction_by_name: dict[str, dict[str, str]] | None = None
    val_prediction_by_name: dict[str, dict[str, str]] | None = None
    if train_coordinate_source == "predicted":
        oof_predictions = read_csv_rows(project_path(cfg["train_oof_predictions_csv"]))
        train_prediction_by_name = prediction_map(oof_predictions, cache_rows, "train", "OOF train")
        if not all(row.get("oof_fold", "") != "" for row in train_prediction_by_name.values()):
            raise RuntimeError("Training predictions must include a strict OOF fold ID.")
    elif train_coordinate_source != "target":
        raise ValueError(f"Unsupported train_coordinate_source: {train_coordinate_source}")
    if validation_coordinate_source == "predicted":
        val_predictions = read_csv_rows(project_path(cfg["validation_predictions_csv"]))
        val_prediction_by_name = prediction_map(val_predictions, val_rows, "val", "validation")
    elif validation_coordinate_source != "target":
        raise ValueError(f"Unsupported validation_coordinate_source: {validation_coordinate_source}")

    crop_size, tactile_size = int(cfg.get("cache_crop_size", 48)), int(cfg.get("tactile_size", 96))
    filter_k = min(int(cfg.get("geometry_filter_k", 32)), len(cache_rows))
    threshold = float(cfg.get("tactile_mask_threshold", 0.04))
    raw_cache_geometry = np.stack([
        motion_geometry_feature(row, float(row["target_tip_x"]), float(row["target_tip_y"])) for row in cache_rows
    ])
    cache_geometry, geometry_mean, geometry_std = standardize(raw_cache_geometry, raw_cache_geometry)
    cache_geometry = cache_geometry.astype(np.float32)
    cache_patches = np.stack([
        crop_contact_patch(row["vision_path"], float(row["target_tip_x"]), float(row["target_tip_y"]), crop_size) for row in cache_rows
    ]).astype(np.float32)
    cache_hand_raw = np.stack([visual_patch_feature_from_patch(patch) for patch in cache_patches])
    cache_hand, hand_mean, hand_std = standardize(cache_hand_raw, cache_hand_raw)

    diff_cache: dict[str, np.ndarray] = {}

    def touch_for_row(row: dict[str, str]) -> np.ndarray:
        return tactile_difference(row["touch_path"], diff_cache, tactile_size)

    cache_tactile_embeddings = np.stack([tactile_embedding(touch_for_row(row)) for row in cache_rows]).astype(np.float32)

    def query_inputs(
        query_rows: list[dict[str, str]], coordinate_source: str, predictions_by_name: dict[str, dict[str, str]] | None,
    ):
        patches, geometry, hand, tactile, coordinates = [], [], [], [], []
        for row in query_rows:
            if coordinate_source == "predicted":
                if predictions_by_name is None:
                    raise RuntimeError("Missing predictions for predicted coordinate source.")
                prediction = predictions_by_name[row["image_name"]]
                x, y = float(prediction["pred_x"]), float(prediction["pred_y"])
            else:
                x, y = float(row["target_tip_x"]), float(row["target_tip_y"])
            patch = crop_contact_patch(row["vision_path"], x, y, crop_size)
            patches.append(patch)
            geometry.append((motion_geometry_feature(row, x, y) - geometry_mean) / geometry_std)
            hand.append((visual_patch_feature_from_patch(patch) - hand_mean) / hand_std)
            tactile.append(touch_for_row(row))
            coordinates.append((x, y))
        return (
            np.stack(patches).astype(np.float32), np.stack(geometry).astype(np.float32),
            np.stack(hand).astype(np.float32), tactile, coordinates,
        )

    train_patches, train_geometry, train_hand, train_tactile, _ = query_inputs(cache_rows, train_coordinate_source, train_prediction_by_name)
    val_patches, val_geometry, val_hand, val_tactile, val_coordinates = query_inputs(val_rows, validation_coordinate_source, val_prediction_by_name)
    train_tactile_embeddings = np.stack([tactile_embedding(image) for image in train_tactile]).astype(np.float32)
    val_tactile_embeddings = np.stack([tactile_embedding(image) for image in val_tactile]).astype(np.float32)
    train_groups = build_groups(
        cache_rows, train_geometry, train_hand, train_tactile, cache_rows, cache_geometry, cache_hand,
        cache_tactile_embeddings, train_tactile_embeddings, touch_for_row, filter_k, True, target_mode,
    )
    val_groups = build_groups(
        val_rows, val_geometry, val_hand, val_tactile, cache_rows, cache_geometry, cache_hand,
        cache_tactile_embeddings, val_tactile_embeddings, touch_for_row, filter_k, False, target_mode,
    )

    model = SoftTactileRanker(cache_geometry.shape[1], float(cfg.get("dropout", 0.1))).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(cfg.get("learning_rate", 0.001)), weight_decay=float(cfg.get("weight_decay", 1e-4)),
    )
    batch_size, epochs = int(cfg.get("batch_size", 16)), int(cfg.get("epochs", 80))
    target_std = max(float(train_groups.targets.std()), 1e-6)
    temperature = float(cfg.get("target_temperature", 0.002))
    listwise_weight = float(cfg.get("listwise_weight", 1.0))
    checkpoint_dir = project_path(cfg["checkpoint_dir"])
    ensure_dir(checkpoint_dir)
    best_top1, best_epoch, stale, history = -1.0, 0, 0, []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        order = np.random.permutation(len(cache_rows))
        for start in range(0, len(cache_rows), batch_size):
            indices = order[start:start + batch_size]
            candidates = train_groups.candidates[indices]
            query_image = image_tensor(train_patches[indices]).to(device)
            candidate_image = image_tensor(cache_patches[candidates].reshape(-1, *cache_patches.shape[1:])).reshape(
                len(indices), filter_k, 3, *cache_patches.shape[1:3]
            ).to(device)
            query_geo = torch.from_numpy(train_geometry[indices]).to(device)
            candidate_geo = torch.from_numpy(cache_geometry[candidates]).to(device)
            current = torch.from_numpy(train_groups.current_scores[indices]).to(device)
            targets = torch.from_numpy(train_groups.targets[indices]).to(device)
            scores = model(query_image, candidate_image, query_geo, candidate_geo, current)
            regression = nn.functional.smooth_l1_loss((scores - targets) / target_std, torch.zeros_like(scores))
            target_distribution = torch.softmax(-targets / temperature, dim=1)
            listwise = -(target_distribution * torch.log_softmax(-scores / temperature, dim=1)).sum(dim=1).mean()
            loss = regression + listwise_weight * listwise
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_scores = predict(model, val_groups, val_patches, val_geometry, cache_patches, cache_geometry, device, batch_size)
        best_rank = np.argmin(val_groups.targets, axis=1)
        score_rank = np.stack([ranks(row) for row in val_scores])
        top1 = float(np.mean(score_rank[np.arange(len(best_rank)), best_rank] == 1))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_tactile_best_top1_rate": top1})
        if top1 > best_top1 + 1e-6:
            best_top1, best_epoch, stale = top1, epoch, 0
            torch.save({
                "model_state": model.state_dict(), "geometry_mean": geometry_mean, "geometry_std": geometry_std,
                "config_section": section, "query_coordinate_source": "strict_oof_c2_top1",
            }, checkpoint_dir / "best.pt")
        else:
            stale += 1
        if stale >= int(cfg.get("early_stopping_patience", 12)):
            break

    checkpoint = torch.load(checkpoint_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    val_scores = predict(model, val_groups, val_patches, val_geometry, cache_patches, cache_geometry, device, batch_size)
    metric_cache: dict[str, np.ndarray] = {}
    output_rows: list[dict[str, str]] = []
    for index, query in enumerate(val_rows):
        current = int(np.argmin(val_groups.current_scores[index]))
        ranker = int(np.argmin(val_scores[index]))
        oracle = int(np.argmin(val_groups.targets[index]))
        local_choices = {"current": current, "ranker": ranker, "oracle": oracle}
        values = {}
        caches = {}
        for name, local_index in local_choices.items():
            cache = cache_rows[int(val_groups.candidates[index, local_index])]
            caches[name] = cache
            values[name] = tactile_metrics(
                val_tactile[index], tactile_difference(cache["touch_path"], metric_cache, tactile_size), threshold,
            )
        x, y = val_coordinates[index]
        output_rows.append({
            "query_record_id": query["record_id"], "query_image_name": query["image_name"], "query_probe": query["probe"],
            "pred_x": f"{x:.3f}", "pred_y": f"{y:.3f}",
            "current_cache_record_id": caches["current"]["record_id"], "ranker_cache_record_id": caches["ranker"]["record_id"],
            "tactile_oracle_cache_record_id": caches["oracle"]["record_id"],
            "current_rank_of_tactile_best": str(int(ranks(val_groups.current_scores[index])[oracle])),
            "ranker_rank_of_tactile_best": str(int(ranks(val_scores[index])[oracle])),
            **{f"{name}_{metric}": f"{metric_values[metric]:.6f}" for name, metric_values in values.items() for metric in METRICS},
        })
    summary = {
        "mode": "phase4b_predicted_box_local_tactile_cache_ranker", "device": str(device),
        "train_queries": len(cache_rows), "validation_queries": len(val_rows), "cache_size": len(cache_rows),
        "geometry_filter_k": filter_k, "best_epoch": best_epoch, "epochs_ran": len(history),
        "validation": {name: metric_summary(output_rows, name) for name in ("current", "ranker", "oracle")},
        "far_probe75_100": {name: metric_summary([row for row in output_rows if int(row["query_probe"]) >= 75], name) for name in ("current", "ranker", "oracle")},
        "checkpoint": str(checkpoint_dir / "best.pt"), "history": history,
        "integrity": {
            "train_coordinate_source": train_coordinate_source,
            "validation_coordinate_source": validation_coordinate_source,
            "train_predictions": "strict record-level OOF C2 Top-1" if train_coordinate_source == "predicted" else "GT contact coordinate",
            "validation_predictions": "C2 refit Top-1" if validation_coordinate_source == "predicted" else "GT contact coordinate",
            "same_record_cache_excluded_for_train": True, "sealed_final_holdout_rows_read": 0,
        },
        "supervision_note": f"Training target is {target_mode}. Tactile labels appear only in train/validation supervision; inference uses local RGB crops and geometry.",
    }
    write_csv_rows(project_path(cfg["query_output_csv"]), output_rows, QUERY_FIELDS)
    write_json(project_path(cfg["metrics_json"]), summary)
    print({key: value for key, value in summary.items() if key != "history"})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a V4 local tactile cache ranker from strict OOF predicted contact boxes.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--section", default="phase4b_predicted_box_cache_ranker_v4")
    args = parser.parse_args()
    train(args.config, args.section)


if __name__ == "__main__":
    main()
