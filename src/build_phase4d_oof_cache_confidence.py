from __future__ import annotations

import argparse
import math

import numpy as np
import torch
from torch import nn

from .build_cache_retrieval import crop_contact_patch, motion_geometry_feature, standardize, visual_patch_feature_from_patch
from .config import load_config, project_path
from .evaluate_oracle_tactile_retrieval import tactile_difference, tactile_embedding, tactile_metrics
from .train_phase4b_predicted_box_cache_ranker import build_groups, is_final_holdout, prediction_map, set_seed
from .train_soft_tactile_cache_ranker import CandidateGroups, SoftTactileRanker, image_tensor, predict, ranks
from .utils import ensure_dir, read_csv_rows, write_csv_rows, write_json


METRICS = [
    "tactile_diff_mae", "tactile_ssim", "tactile_mask_iou", "tactile_area_delta",
    "tactile_centroid_distance", "tactile_embedding_distance",
]
FIELDS = [
    "query_record_id", "query_image_name", "query_probe", "oof_fold", "pred_x", "pred_y",
    "ranker_best_score", "ranker_second_score", "ranker_margin", "ranker_margin_normalized",
    "current_best_score", "current_second_score", "current_margin", "ranker_oracle_embedding_rank",
    "retrieved_cache_record_id", "retrieved_cache_image_name", *METRICS,
]


def query_inputs(
    query_rows: list[dict[str, str]],
    prediction_by_name: dict[str, dict[str, str]],
    geometry_mean: np.ndarray,
    geometry_std: np.ndarray,
    hand_mean: np.ndarray,
    hand_std: np.ndarray,
    crop_size: int,
    touch_for_row,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray], list[tuple[float, float]]]:
    patches, geometry, hand, tactile, coordinates = [], [], [], [], []
    for row in query_rows:
        prediction = prediction_by_name[row["image_name"]]
        x, y = float(prediction["pred_x"]), float(prediction["pred_y"])
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


def train_fold(
    fold: str,
    all_cache_rows: list[dict[str, str]],
    fold_by_name: dict[str, str],
    prediction_by_name: dict[str, dict[str, str]],
    cfg: dict,
    device: torch.device,
) -> tuple[list[dict[str, str]], dict]:
    fit_rows = [row for row in all_cache_rows if fold_by_name[row["image_name"]] != fold]
    query_rows = [row for row in all_cache_rows if fold_by_name[row["image_name"]] == fold]
    if {row["record_id"] for row in fit_rows} & {row["record_id"] for row in query_rows}:
        raise RuntimeError(f"Fold {fold} has record leakage between fit and OOF query rows.")
    crop_size, tactile_size = int(cfg.get("cache_crop_size", 48)), int(cfg.get("tactile_size", 96))
    filter_k = min(int(cfg.get("geometry_filter_k", 32)), len(fit_rows))
    threshold = float(cfg.get("tactile_mask_threshold", 0.04))

    raw_fit_geometry = np.stack([
        motion_geometry_feature(row, float(row["target_tip_x"]), float(row["target_tip_y"])) for row in fit_rows
    ])
    _, geometry_mean, geometry_std = standardize(raw_fit_geometry, raw_fit_geometry)
    raw_all_geometry = np.stack([
        motion_geometry_feature(row, float(row["target_tip_x"]), float(row["target_tip_y"])) for row in all_cache_rows
    ])
    fit_geometry = ((raw_fit_geometry - geometry_mean) / geometry_std).astype(np.float32)
    all_geometry = ((raw_all_geometry - geometry_mean) / geometry_std).astype(np.float32)
    fit_patches = np.stack([
        crop_contact_patch(row["vision_path"], float(row["target_tip_x"]), float(row["target_tip_y"]), crop_size) for row in fit_rows
    ]).astype(np.float32)
    all_patches = np.stack([
        crop_contact_patch(row["vision_path"], float(row["target_tip_x"]), float(row["target_tip_y"]), crop_size) for row in all_cache_rows
    ]).astype(np.float32)
    fit_hand_raw = np.stack([visual_patch_feature_from_patch(patch) for patch in fit_patches])
    all_hand_raw = np.stack([visual_patch_feature_from_patch(patch) for patch in all_patches])
    fit_hand, hand_mean, hand_std = standardize(fit_hand_raw, fit_hand_raw)
    all_hand = ((all_hand_raw - hand_mean) / hand_std).astype(np.float32)

    diff_cache: dict[str, np.ndarray] = {}

    def touch_for_row(row: dict[str, str]) -> np.ndarray:
        return tactile_difference(row["touch_path"], diff_cache, tactile_size)

    fit_tactile_embeddings = np.stack([tactile_embedding(touch_for_row(row)) for row in fit_rows]).astype(np.float32)
    all_tactile_embeddings = np.stack([tactile_embedding(touch_for_row(row)) for row in all_cache_rows]).astype(np.float32)
    fit_patches_query, fit_geometry_query, fit_hand_query, fit_tactile, _ = query_inputs(
        fit_rows, prediction_by_name, geometry_mean, geometry_std, hand_mean, hand_std, crop_size, touch_for_row
    )
    query_patches, query_geometry, query_hand, query_tactile, query_coordinates = query_inputs(
        query_rows, prediction_by_name, geometry_mean, geometry_std, hand_mean, hand_std, crop_size, touch_for_row
    )
    fit_tactile_query_embeddings = np.stack([tactile_embedding(item) for item in fit_tactile]).astype(np.float32)
    query_tactile_embeddings = np.stack([tactile_embedding(item) for item in query_tactile]).astype(np.float32)
    train_groups = build_groups(
        fit_rows, fit_geometry_query, fit_hand_query, fit_tactile, fit_rows, fit_geometry, fit_hand,
        fit_tactile_embeddings, fit_tactile_query_embeddings, touch_for_row, filter_k, True, "tactile_embedding",
    )
    query_groups = build_groups(
        query_rows, query_geometry, query_hand, query_tactile, all_cache_rows, all_geometry, all_hand,
        all_tactile_embeddings, query_tactile_embeddings, touch_for_row, min(filter_k, len(all_cache_rows)), True, "tactile_embedding",
    )
    model = SoftTactileRanker(fit_geometry.shape[1], float(cfg.get("dropout", 0.1))).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(cfg.get("learning_rate", 0.001)), weight_decay=float(cfg.get("weight_decay", 1e-4)),
    )
    target_std = max(float(train_groups.targets.std()), 1e-6)
    temperature = float(cfg.get("target_temperature", 0.02))
    listwise_weight = float(cfg.get("listwise_weight", 1.0))
    batch_size = int(cfg.get("batch_size", 16))
    for _ in range(int(cfg.get("epochs", 4))):
        model.train()
        order = np.random.permutation(len(fit_rows))
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            candidates = train_groups.candidates[indices]
            query_image = image_tensor(fit_patches_query[indices]).to(device)
            candidate_image = image_tensor(fit_patches[candidates].reshape(-1, *fit_patches.shape[1:])).reshape(
                len(indices), filter_k, 3, *fit_patches.shape[1:3]
            ).to(device)
            query_geo = torch.from_numpy(fit_geometry_query[indices]).to(device)
            candidate_geo = torch.from_numpy(fit_geometry[candidates]).to(device)
            current = torch.from_numpy(train_groups.current_scores[indices]).to(device)
            targets = torch.from_numpy(train_groups.targets[indices]).to(device)
            scores = model(query_image, candidate_image, query_geo, candidate_geo, current)
            regression = nn.functional.smooth_l1_loss((scores - targets) / target_std, torch.zeros_like(scores))
            distribution = torch.softmax(-targets / temperature, dim=1)
            listwise = -(distribution * torch.log_softmax(-scores / temperature, dim=1)).sum(dim=1).mean()
            optimizer.zero_grad(set_to_none=True)
            (regression + listwise_weight * listwise).backward()
            optimizer.step()

    scores = predict(model, query_groups, query_patches, query_geometry, all_patches, all_geometry, device, batch_size)
    metric_cache: dict[str, np.ndarray] = {}
    output_rows = []
    for index, query in enumerate(query_rows):
        score_order = np.argsort(scores[index], kind="stable")
        current_order = np.argsort(query_groups.current_scores[index], kind="stable")
        selected = int(score_order[0])
        cache = all_cache_rows[int(query_groups.candidates[index, selected])]
        metrics = tactile_metrics(query_tactile[index], tactile_difference(cache["touch_path"], metric_cache, tactile_size), threshold)
        best, second = float(scores[index, score_order[0]]), float(scores[index, score_order[1]])
        current_best = float(query_groups.current_scores[index, current_order[0]])
        current_second = float(query_groups.current_scores[index, current_order[1]])
        x, y = query_coordinates[index]
        output_rows.append({
            "query_record_id": query["record_id"], "query_image_name": query["image_name"], "query_probe": query["probe"], "oof_fold": fold,
            "pred_x": f"{x:.3f}", "pred_y": f"{y:.3f}", "ranker_best_score": f"{best:.6f}", "ranker_second_score": f"{second:.6f}",
            "ranker_margin": f"{second - best:.6f}", "ranker_margin_normalized": f"{(second - best) / max(float(scores[index].std()), 1e-6):.6f}",
            "current_best_score": f"{current_best:.6f}", "current_second_score": f"{current_second:.6f}", "current_margin": f"{current_second - current_best:.6f}",
            "ranker_oracle_embedding_rank": str(int(ranks(scores[index])[int(np.argmin(query_groups.targets[index]))])),
            "retrieved_cache_record_id": cache["record_id"], "retrieved_cache_image_name": cache["image_name"],
            **{key: f"{value:.6f}" for key, value in metrics.items()},
        })
    checkpoint_dir = project_path(cfg["checkpoint_dir"])
    ensure_dir(checkpoint_dir)
    torch.save({"model_state": model.state_dict(), "fold": fold, "fit_records": len({row['record_id'] for row in fit_rows})}, checkpoint_dir / f"fold_{fold}.pt")
    return output_rows, {"fold": fold, "fit_queries": len(fit_rows), "oof_queries": len(query_rows), "fit_records": len({row['record_id'] for row in fit_rows}), "oof_records": len({row['record_id'] for row in query_rows})}


def build(config_path: str, section: str) -> dict:
    cfg = load_config(config_path)[section]
    set_seed(int(cfg.get("seed", 20260802)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = read_csv_rows(project_path(cfg["samples_csv"]))
    if any(is_final_holdout(row) for row in rows):
        raise RuntimeError("Refusing to access sealed final-holdout samples.")
    cache_rows = [row for row in rows if row["dataset_split"] == "train"]
    predictions = read_csv_rows(project_path(cfg["oof_predictions_csv"]))
    prediction_by_name = prediction_map(predictions, cache_rows, "train", "OOF cache confidence")
    fold_by_name = {name: prediction["oof_fold"] for name, prediction in prediction_by_name.items()}
    folds = sorted(set(fold_by_name.values()))
    if len(folds) < 2 or any(not fold for fold in folds):
        raise RuntimeError(f"Expected at least two OOF folds, got {folds}")
    output_rows, fold_summaries = [], []
    for fold in folds:
        rows_for_fold, fold_summary = train_fold(fold, cache_rows, fold_by_name, prediction_by_name, cfg, device)
        output_rows.extend(rows_for_fold)
        fold_summaries.append(fold_summary)
    if len(output_rows) != len(cache_rows) or len({row["query_image_name"] for row in output_rows}) != len(cache_rows):
        raise RuntimeError("OOF cache-confidence outputs do not cover train queries exactly once.")
    if any(is_final_holdout(row) for row in output_rows):
        raise RuntimeError("OOF cache-confidence output includes final holdout.")
    summary = {
        "mode": "phase4d_strict_oof_cache_confidence", "device": str(device), "queries": len(output_rows),
        "folds": fold_summaries, "epochs_per_fold": int(cfg.get("epochs", 4)), "checkpoint_dir": str(project_path(cfg["checkpoint_dir"])),
        "integrity": {
            "contact_predictions": "strict record-level C2 OOF", "cache_ranker_training": "each cache-ranker fold excludes its OOF query records",
            "cache_at_oof_inference": "full train cache excluding same record", "sealed_final_holdout_rows_read": 0,
        },
        "note": "True tactile metrics are stored only as offline labels for the future miss predictor; ranker inference uses local RGB crops, geometry, and cache candidates.",
    }
    write_csv_rows(project_path(cfg["output_csv"]), output_rows, FIELDS)
    write_json(project_path(cfg["metrics_json"]), summary)
    print(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build strict OOF cache-ranker confidence labels for Phase 4D.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--section", default="phase4d_oof_cache_confidence_v4")
    args = parser.parse_args()
    build(args.config, args.section)


if __name__ == "__main__":
    main()
