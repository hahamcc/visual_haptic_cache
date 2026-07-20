from __future__ import annotations

import argparse
import math

import numpy as np
import torch

from .build_cache_retrieval import crop_contact_patch, motion_geometry_feature, standardize, visual_patch_feature_from_patch
from .config import load_config, project_path
from .evaluate_oracle_tactile_retrieval import tactile_difference, tactile_embedding, tactile_metrics
from .train_soft_tactile_cache_ranker import CandidateGroups, SoftTactileRanker, predict, ranks
from .utils import read_csv_rows, write_csv_rows, write_json


METRICS = [
    "tactile_diff_mae", "tactile_ssim", "tactile_mask_iou", "tactile_area_delta",
    "tactile_centroid_distance", "tactile_embedding_distance",
]
QUERY_FIELDS = [
    "query_record_id", "query_image_name", "query_probe", "pred_x", "pred_y",
    "ranker_best_score", "ranker_second_score", "ranker_margin", "ranker_margin_normalized",
    "current_best_score", "current_second_score", "current_margin", "ranker_oracle_embedding_rank",
    *METRICS,
]


def record_number(record_id: str) -> int:
    return int(record_id.rsplit("_", 1)[-1])


def is_final_holdout(row: dict[str, str]) -> bool:
    return row.get("split") == "0" and 950 <= record_number(row["record_id"]) <= 999


def rankdata(values: np.ndarray) -> np.ndarray:
    result = np.empty(len(values), dtype=np.float32)
    result[np.argsort(values, kind="stable")] = np.arange(len(values), dtype=np.float32)
    return result


def spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 2 or float(left.std()) < 1e-8 or float(right.std()) < 1e-8:
        return None
    left_rank, right_rank = rankdata(left), rankdata(right)
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def selected_summary(rows: list[dict[str, str]], confidence: str, coverages: list[float]) -> list[dict[str, float | int]]:
    ordered = sorted(rows, key=lambda row: float(row[confidence]), reverse=True)
    output = []
    for coverage in coverages:
        count = max(1, int(math.ceil(len(ordered) * coverage)))
        selected = ordered[:count]
        output.append({
            "coverage": coverage,
            "queries": count,
            "mean_tactile_diff_mae": float(np.mean([float(row["tactile_diff_mae"]) for row in selected])),
            "mean_tactile_ssim": float(np.mean([float(row["tactile_ssim"]) for row in selected])),
            "mean_tactile_mask_iou": float(np.mean([float(row["tactile_mask_iou"]) for row in selected])),
            "mean_oracle_embedding_rank": float(np.mean([float(row["ranker_oracle_embedding_rank"]) for row in selected])),
        })
    return output


def audit(config_path: str, section: str) -> dict:
    cfg = load_config(config_path)[section]
    rows = read_csv_rows(project_path(cfg["samples_csv"]))
    if any(is_final_holdout(row) for row in rows):
        raise RuntimeError("Refusing to access sealed final-holdout samples.")
    cache_rows = [row for row in rows if row["dataset_split"] == "train"]
    val_by_name = {row["image_name"]: row for row in rows if row["dataset_split"] == "val"}
    predictions = [
        row for row in read_csv_rows(project_path(cfg["predictions_csv"]))
        if row.get("dataset_split") == "val" and row["image_name"] in val_by_name
    ]
    if len(predictions) != len(val_by_name):
        raise RuntimeError(f"Need one validation prediction per query, got {len(predictions)} for {len(val_by_name)} rows.")
    if any(is_final_holdout(row) for row in predictions):
        raise RuntimeError("Prediction file includes sealed final-holdout data.")
    if {row["record_id"] for row in cache_rows} & {row["record_id"] for row in val_by_name.values()}:
        raise RuntimeError("Train cache and validation records overlap.")

    crop_size, tactile_size = int(cfg.get("cache_crop_size", 48)), int(cfg.get("tactile_size", 96))
    filter_k = min(int(cfg.get("geometry_filter_k", 32)), len(cache_rows))
    threshold = float(cfg.get("tactile_mask_threshold", 0.04))
    raw_geometry = np.stack([
        motion_geometry_feature(row, float(row["target_tip_x"]), float(row["target_tip_y"])) for row in cache_rows
    ])
    cache_geometry, geometry_mean, geometry_std = standardize(raw_geometry, raw_geometry)
    cache_geometry = cache_geometry.astype(np.float32)
    cache_patches = np.stack([
        crop_contact_patch(row["vision_path"], float(row["target_tip_x"]), float(row["target_tip_y"]), crop_size) for row in cache_rows
    ]).astype(np.float32)
    cache_hand_raw = np.stack([visual_patch_feature_from_patch(patch) for patch in cache_patches])
    cache_hand, hand_mean, hand_std = standardize(cache_hand_raw, cache_hand_raw)

    diff_cache: dict[str, np.ndarray] = {}

    def touch(row: dict[str, str]) -> np.ndarray:
        return tactile_difference(row["touch_path"], diff_cache, tactile_size)

    cache_tactile = np.stack([tactile_embedding(touch(row)) for row in cache_rows]).astype(np.float32)
    ordered_predictions = sorted(predictions, key=lambda row: row["image_name"])
    query_rows, query_patches, query_geometry, query_hand, query_tactile = [], [], [], [], []
    candidates, targets, current_scores = [], [], []
    for prediction in ordered_predictions:
        row = val_by_name[prediction["image_name"]]
        x, y = float(prediction["pred_x"]), float(prediction["pred_y"])
        patch = crop_contact_patch(row["vision_path"], x, y, crop_size)
        geometry = ((motion_geometry_feature(row, x, y) - geometry_mean) / geometry_std).astype(np.float32)
        hand = ((visual_patch_feature_from_patch(patch) - hand_mean) / hand_std).astype(np.float32)
        tactile = touch(row)
        geometry_distance = np.linalg.norm(cache_geometry - geometry[None], axis=1)
        shortlist = np.argpartition(geometry_distance, filter_k - 1)[:filter_k]
        shortlist = shortlist[np.argsort(geometry_distance[shortlist], kind="stable")]
        visual_distance = np.linalg.norm(cache_hand[shortlist] - hand[None], axis=1)
        current = geometry_distance[shortlist] / math.sqrt(cache_geometry.shape[1]) + visual_distance / math.sqrt(cache_hand.shape[1])
        query_rows.append(row)
        query_patches.append(patch)
        query_geometry.append(geometry)
        query_hand.append(hand)
        query_tactile.append(tactile)
        candidates.append(shortlist.astype(np.int32))
        targets.append(np.linalg.norm(cache_tactile[shortlist] - tactile_embedding(tactile)[None], axis=1).astype(np.float32))
        current_scores.append(current.astype(np.float32))
    query_patches = np.stack(query_patches).astype(np.float32)
    query_geometry = np.stack(query_geometry).astype(np.float32)
    groups = CandidateGroups(np.stack(candidates), np.stack(targets), np.stack(current_scores))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = project_path(cfg["checkpoint"])
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = SoftTactileRanker(cache_geometry.shape[1], float(cfg.get("dropout", 0.1))).to(device)
    model.load_state_dict(checkpoint["model_state"])
    scores = predict(model, groups, query_patches, query_geometry, cache_patches, cache_geometry, device, int(cfg.get("batch_size", 16)))

    metric_cache: dict[str, np.ndarray] = {}
    output_rows = []
    for index, query in enumerate(query_rows):
        score_order = np.argsort(scores[index], kind="stable")
        current_order = np.argsort(groups.current_scores[index], kind="stable")
        selected = int(score_order[0])
        cache = cache_rows[int(groups.candidates[index, selected])]
        values = tactile_metrics(query_tactile[index], tactile_difference(cache["touch_path"], metric_cache, tactile_size), threshold)
        best, second = float(scores[index, score_order[0]]), float(scores[index, score_order[1]])
        scale = max(float(scores[index].std()), 1e-6)
        current_best, current_second = float(groups.current_scores[index, current_order[0]]), float(groups.current_scores[index, current_order[1]])
        prediction = ordered_predictions[index]
        output_rows.append({
            "query_record_id": query["record_id"], "query_image_name": query["image_name"], "query_probe": query["probe"],
            "pred_x": prediction["pred_x"], "pred_y": prediction["pred_y"],
            "ranker_best_score": f"{best:.6f}", "ranker_second_score": f"{second:.6f}",
            "ranker_margin": f"{second - best:.6f}", "ranker_margin_normalized": f"{(second - best) / scale:.6f}",
            "current_best_score": f"{current_best:.6f}", "current_second_score": f"{current_second:.6f}",
            "current_margin": f"{current_second - current_best:.6f}",
            "ranker_oracle_embedding_rank": str(int(ranks(scores[index])[int(np.argmin(groups.targets[index]))])),
            **{key: f"{value:.6f}" for key, value in values.items()},
        })

    coverages = [float(value) for value in cfg.get("coverages", [0.1, 0.2, 0.3, 0.5, 0.7, 1.0])]
    all_mae = np.asarray([float(row["tactile_diff_mae"]) for row in output_rows])
    all_ssim = np.asarray([float(row["tactile_ssim"]) for row in output_rows])
    all_iou = np.asarray([float(row["tactile_mask_iou"]) for row in output_rows])
    confidence_fields = ["ranker_margin", "ranker_margin_normalized", "ranker_best_score", "current_margin"]
    correlation = {}
    for field in confidence_fields:
        values = np.asarray([float(row[field]) for row in output_rows])
        # Lower score is better; the other confidence features are higher-is-better.
        signed = -values if field == "ranker_best_score" else values
        correlation[field] = {
            "spearman_with_negative_mae": spearman(signed, -all_mae),
            "spearman_with_ssim": spearman(signed, all_ssim),
            "spearman_with_iou": spearman(signed, all_iou),
            "selective_curve": selected_summary(output_rows, field, coverages) if field != "ranker_best_score" else selected_summary(
                [dict(row, ranker_best_score=str(-float(row["ranker_best_score"]))) for row in output_rows], "ranker_best_score", coverages
            ),
        }
    summary = {
        "mode": "phase4c_validation_only_cache_confidence_audit", "device": str(device),
        "validation_queries": len(output_rows), "far_queries": sum(int(row["query_probe"]) >= 75 for row in output_rows),
        "geometry_filter_k": filter_k, "checkpoint": str(checkpoint_path),
        "correlation": correlation,
        "overall": {
            "mean_tactile_diff_mae": float(all_mae.mean()), "mean_tactile_ssim": float(all_ssim.mean()),
            "mean_tactile_mask_iou": float(all_iou.mean()),
        },
        "integrity": {"validation_only": True, "sealed_final_holdout_rows_read": 0, "online_features_only": confidence_fields},
        "note": "This is a diagnostic only. It does not choose a cache-miss threshold; true tactile metrics are used solely to test whether online confidence scores predict retrieval quality.",
    }
    write_csv_rows(project_path(cfg["output_csv"]), output_rows, QUERY_FIELDS)
    write_json(project_path(cfg["metrics_json"]), summary)
    print(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit whether Phase 4B cache-ranker confidence can support cache-miss gating.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--section", default="phase4c_cache_confidence_audit_v4")
    args = parser.parse_args()
    audit(args.config, args.section)


if __name__ == "__main__":
    main()
