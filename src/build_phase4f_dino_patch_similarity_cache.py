"""Training-free DINOv2 patch-alignment diagnostic for Phase 4F cache ranking.

This experiment deliberately retains the V1 geometry Top-K shortlist and uses
no tactile signal at inference.  It tests whether frozen DINO patch tokens
carry useful local correspondence before adding another learned ranking head.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from torch.nn import functional as F

from .build_cache_retrieval import crop_contact_patch, motion_geometry_feature, standardize, visual_patch_feature_from_patch
from .build_phase4e_oof_multiscale_cache import build_shortlists, normalized_features
from .build_phase4f_dino_cross_attention_cache import embedding_matrix, encode, tactile_targets
from .config import load_config, project_path
from .evaluate_oracle_tactile_retrieval import tactile_difference, tactile_metrics
from .phase4f_dino_cross_attention import FrozenDinoV2
from .train_phase4b_predicted_box_cache_ranker import is_final_holdout, prediction_map, set_seed
from .train_soft_tactile_cache_ranker import ranks
from .utils import read_csv_rows, write_csv_rows, write_json


QUERY_FIELDS = [
    "query_record_id", "query_image_name", "query_probe", "oof_fold", "pred_x", "pred_y",
    "selected_cache_record_id", "selected_cache_image_name", "ranker_best_score", "ranker_margin",
    "ranker_oracle_embedding_rank", "tactile_diff_mae", "tactile_ssim", "tactile_mask_iou",
]
CANDIDATE_FIELDS = [
    "query_record_id", "query_image_name", "query_probe", "oof_fold", "candidate_rank", "candidate_score",
    "detail_patch_similarity", "context_patch_similarity", "patch_similarity_score", "candidate_record_id",
    "candidate_image_name", "candidate_tactile_embedding_distance", "candidate_tactile_ssim",
    "candidate_tactile_mask_iou", "candidate_oracle_embedding_rank",
]


def local_patch_similarity(query: torch.Tensor, cache: torch.Tensor, radius: int) -> torch.Tensor:
    """Symmetric local best-match cosine similarity for aligned DINO patch grids."""
    batch, tokens, _ = query.shape
    _, candidates, cache_tokens, _ = cache.shape
    side = int(round(tokens ** 0.5))
    if tokens != cache_tokens or side * side != tokens:
        raise ValueError(f"Expected equal square token grids, got query={tuple(query.shape)}, cache={tuple(cache.shape)}")
    axis = torch.arange(side, device=query.device)
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    positions = torch.stack((x.reshape(-1), y.reshape(-1)), dim=1)
    local = (positions[:, None] - positions[None, :]).abs().amax(dim=-1) <= radius
    cosine = torch.einsum("btd,bksd->bkts", F.normalize(query, dim=-1), F.normalize(cache, dim=-1))
    cosine = cosine.masked_fill(~local[None, None], torch.finfo(cosine.dtype).min)
    query_to_cache = cosine.amax(dim=-1).mean(dim=-1)
    cache_to_query = cosine.amax(dim=-2).mean(dim=-1)
    return 0.5 * (query_to_cache + cache_to_query)


def patch_scores(
    query_detail: np.ndarray, cache_detail: np.ndarray, query_context: np.ndarray, cache_context: np.ndarray,
    candidates: np.ndarray, detail_weight: float, radius: int, device: torch.device, batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    detail_scores, context_scores = [], []
    with torch.no_grad():
        for start in range(0, len(query_detail), batch_size):
            end = start + batch_size
            indices = candidates[start:end]
            detail_scores.append(local_patch_similarity(
                torch.from_numpy(query_detail[start:end]).to(device), torch.from_numpy(cache_detail[indices]).to(device), radius,
            ).cpu().numpy())
            context_scores.append(local_patch_similarity(
                torch.from_numpy(query_context[start:end]).to(device), torch.from_numpy(cache_context[indices]).to(device), radius,
            ).cpu().numpy())
    detail_scores, context_scores = np.concatenate(detail_scores), np.concatenate(context_scores)
    similarity = detail_weight * detail_scores + (1.0 - detail_weight) * context_scores
    return detail_scores, context_scores, -similarity


def fold_run(
    fold: str, rows: list[dict[str, str]], folds: dict[str, str], predictions: dict[str, dict[str, str]], cfg: dict, device: torch.device,
) -> tuple[list[dict], list[dict]]:
    fit = [row for row in rows if folds[row["image_name"]] != fold]
    query = [row for row in rows if folds[row["image_name"]] == fold]
    if {row["record_id"] for row in fit} & {row["record_id"] for row in query}:
        raise RuntimeError("Record leakage in Phase4F patch-similarity fold")
    detail_size, context_size = int(cfg["detail_crop_size"]), int(cfg["context_crop_size"])
    raw_fit_geometry = np.stack([motion_geometry_feature(row, float(row["target_tip_x"]), float(row["target_tip_y"])) for row in fit])
    _, geometry_mean, geometry_std = standardize(raw_fit_geometry, raw_fit_geometry)
    fit_gt = [(float(row["target_tip_x"]), float(row["target_tip_y"])) for row in fit]
    cache_gt = [(float(row["target_tip_x"]), float(row["target_tip_y"])) for row in rows]
    raw_fit_detail = np.stack([visual_patch_feature_from_patch(crop_contact_patch(row["vision_path"], x, y, detail_size)) for row, (x, y) in zip(fit, fit_gt, strict=True)])
    raw_fit_context = np.stack([visual_patch_feature_from_patch(crop_contact_patch(row["vision_path"], x, y, context_size)) for row, (x, y) in zip(fit, fit_gt, strict=True)])
    _, detail_mean, detail_std = standardize(raw_fit_detail, raw_fit_detail)
    _, context_mean, context_std = standardize(raw_fit_context, raw_fit_context)
    cache_detail, cache_context, cache_geometry, cache_detail_hand, cache_context_hand = normalized_features(
        rows, cache_gt, geometry_mean, geometry_std, detail_mean, detail_std, context_mean, context_std, detail_size, context_size,
    )
    query_xy = [(float(predictions[row["image_name"]]["pred_x"]), float(predictions[row["image_name"]]["pred_y"])) for row in query]
    query_detail, query_context, query_geometry, query_detail_hand, query_context_hand = normalized_features(
        query, query_xy, geometry_mean, geometry_std, detail_mean, detail_std, context_mean, context_std, detail_size, context_size,
    )
    tactile_cache: dict[str, np.ndarray] = {}
    def touch(row: dict[str, str]) -> np.ndarray:
        return tactile_difference(row["touch_path"], tactile_cache, int(cfg["tactile_size"]))
    print(f"phase4f patch-sim fold {fold}: caching tactile embeddings for development_cache={len(rows)}", flush=True)
    cache_embeddings = embedding_matrix(rows, touch, f"patch-sim fold {fold} development cache")
    embedding_by_name = {row["image_name"]: value for row, value in zip(rows, cache_embeddings, strict=True)}
    query_embeddings = np.stack([embedding_by_name[row["image_name"]] for row in query]).astype(np.float32)
    candidates, targets, _, _, _, _ = build_shortlists(
        query, query_geometry, query_detail_hand, query_context_hand, query_embeddings,
        rows, cache_geometry, cache_detail_hand, cache_context_hand, cache_embeddings,
        min(int(cfg["geometry_filter_k"]), len(rows)), True,
    )
    if any(row["record_id"] == rows[int(cache_index)]["record_id"] for row, group in zip(query, candidates, strict=True) for cache_index in group):
        raise RuntimeError("Phase4F patch-similarity shortlist contains a same-record cache entry.")
    print(f"phase4f patch-sim fold {fold}: loading frozen {cfg['dino_model']} and encoding visual patches", flush=True)
    backbone = FrozenDinoV2(str(cfg["dino_model"]), int(cfg["dino_image_size"])).to(device)
    cache_detail_tokens = encode(backbone, cache_detail, device, int(cfg["batch_size"]))
    cache_context_tokens = encode(backbone, cache_context, device, int(cfg["batch_size"]))
    query_detail_tokens = encode(backbone, query_detail, device, int(cfg["batch_size"]))
    query_context_tokens = encode(backbone, query_context, device, int(cfg["batch_size"]))
    detail_similarity, context_similarity, scores = patch_scores(
        query_detail_tokens, cache_detail_tokens, query_context_tokens, cache_context_tokens, candidates,
        float(cfg["detail_similarity_weight"]), int(cfg["local_match_radius"]), device, int(cfg["batch_size"]),
    )
    order = np.argsort(scores, axis=1, kind="stable")
    eval_ssim, eval_iou = tactile_targets(query, rows, candidates, touch, float(cfg["tactile_mask_threshold"]), f"patch-sim fold {fold} OOF")
    queries, candidate_rows = [], []
    for index, row in enumerate(query):
        choice = int(order[index, 0])
        selected = rows[int(candidates[index, choice])]
        metric = tactile_metrics(touch(row), touch(selected), float(cfg["tactile_mask_threshold"]))
        oracle = int(np.argmin(targets[index]))
        queries.append({
            "query_record_id": row["record_id"], "query_image_name": row["image_name"], "query_probe": row["probe"], "oof_fold": fold,
            "pred_x": f"{query_xy[index][0]:.3f}", "pred_y": f"{query_xy[index][1]:.3f}",
            "selected_cache_record_id": selected["record_id"], "selected_cache_image_name": selected["image_name"],
            "ranker_best_score": f"{scores[index, choice]:.6f}", "ranker_margin": f"{scores[index, order[index, 1]] - scores[index, choice]:.6f}",
            "ranker_oracle_embedding_rank": str(int(ranks(scores[index])[oracle])),
            **{key: f"{metric[key]:.6f}" for key in ("tactile_diff_mae", "tactile_ssim", "tactile_mask_iou")},
        })
        for rank, item in enumerate(order[index], start=1):
            cache = rows[int(candidates[index, item])]
            candidate_rows.append({
                "query_record_id": row["record_id"], "query_image_name": row["image_name"], "query_probe": row["probe"], "oof_fold": fold,
                "candidate_rank": str(rank), "candidate_score": f"{scores[index, item]:.6f}",
                "detail_patch_similarity": f"{detail_similarity[index, item]:.6f}",
                "context_patch_similarity": f"{context_similarity[index, item]:.6f}",
                "patch_similarity_score": f"{-scores[index, item]:.6f}",
                "candidate_record_id": cache["record_id"], "candidate_image_name": cache["image_name"],
                "candidate_tactile_embedding_distance": f"{targets[index, item]:.6f}",
                "candidate_tactile_ssim": f"{eval_ssim[index, item]:.6f}", "candidate_tactile_mask_iou": f"{eval_iou[index, item]:.6f}",
                "candidate_oracle_embedding_rank": str(int(ranks(targets[index])[item])),
            })
    return queries, candidate_rows


def build(config_path: str, section: str) -> dict:
    cfg = load_config(config_path)[section]
    set_seed(int(cfg["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = read_csv_rows(project_path(cfg["samples_csv"]))
    if any(is_final_holdout(row) for row in rows):
        raise RuntimeError("Phase4F patch-similarity diagnostic refuses sealed final-holdout samples.")
    rows = [row for row in rows if row["dataset_split"] == "train"]
    predictions = prediction_map(read_csv_rows(project_path(cfg["oof_predictions_csv"])), rows, "train", "Phase4F patch-similarity strict OOF")
    folds = {name: prediction["oof_fold"] for name, prediction in predictions.items()}
    queries, candidates = [], []
    for fold in sorted(set(folds.values())):
        query_rows, candidate_rows = fold_run(fold, rows, folds, predictions, cfg, device)
        queries.extend(query_rows)
        candidates.extend(candidate_rows)
    if len({row["query_image_name"] for row in queries}) != len(rows):
        raise RuntimeError("OOF patch-similarity output must cover every development query once.")
    write_csv_rows(project_path(cfg["query_output_csv"]), queries, QUERY_FIELDS)
    write_csv_rows(project_path(cfg["candidate_output_csv"]), candidates, CANDIDATE_FIELDS)
    summary = {
        "mode": "phase4f_strict_oof_dinov2_training_free_local_patch_similarity",
        "queries": len(queries), "candidates": len(candidates), "dino_model": cfg["dino_model"],
        "local_match_radius": cfg["local_match_radius"], "detail_similarity_weight": cfg["detail_similarity_weight"],
        "integrity": {"evaluation_cache": "complete_development_pool", "same_record_cache_excluded": True, "sealed_final_holdout_rows_read": 0, "query_tactile_usage": "offline evaluation only"},
    }
    write_json(project_path(cfg["metrics_json"]), summary)
    print(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build training-free strict OOF DINOv2 local patch-similarity cache outputs.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--section", default="phase4f_dino_patch_similarity_oof_v1")
    args = parser.parse_args()
    build(args.config, args.section)


if __name__ == "__main__":
    main()
