from __future__ import annotations

import argparse
import math

import numpy as np
import torch
from torch import nn

from .build_cache_retrieval import crop_contact_patch, motion_geometry_feature, standardize, visual_patch_feature_from_patch
from .config import load_config, project_path
from .evaluate_oracle_tactile_retrieval import tactile_difference, tactile_embedding, tactile_metrics
from .train_phase4b_predicted_box_cache_ranker import is_final_holdout, prediction_map, set_seed
from .train_soft_tactile_cache_ranker import PatchEncoder, image_tensor, ranks
from .utils import ensure_dir, read_csv_rows, write_csv_rows, write_json


TACTILE_METRICS = [
    "tactile_diff_mae", "tactile_ssim", "tactile_mask_iou", "tactile_area_delta",
    "tactile_centroid_distance", "tactile_embedding_distance",
]
QUERY_FIELDS = [
    "query_record_id", "query_image_name", "query_probe", "oof_fold", "pred_x", "pred_y", "c2_pred_score",
    "ranker_best_score", "ranker_second_score", "ranker_margin", "ranker_margin_normalized",
    "hand_best_score", "hand_second_score", "hand_margin", "ranker_oracle_embedding_rank",
    "selected_cache_record_id", "selected_cache_image_name", "top3_cache_record_ids", "top3_cache_image_names",
    "top3_tactile_embedding_disagreement", "top3_score_std", "geometry_distance", "detail_visual_distance",
    "context_visual_distance", "trajectory_real_point_count", "trajectory_history_span_frames",
    "trajectory_padding_ratio", "trajectory_cumulative_displacement", *TACTILE_METRICS,
]
CANDIDATE_FIELDS = [
    "query_record_id", "query_image_name", "query_probe", "oof_fold", "candidate_rank", "candidate_score",
    "hand_score", "geometry_distance", "detail_visual_distance", "context_visual_distance",
    "candidate_record_id", "candidate_image_name", "candidate_tactile_embedding_distance",
]


class MultiScaleTactileRanker(nn.Module):
    """Ranks a fixed geometry shortlist without using query tactile observations."""

    def __init__(self, geometry_dim: int, dropout: float) -> None:
        super().__init__()
        self.detail_encoder = PatchEncoder()
        self.context_encoder = PatchEncoder()
        image_feature_dim = 64 * 3 * 2
        geometry_feature_dim = geometry_dim * 3
        self.head = nn.Sequential(
            nn.Linear(image_feature_dim + geometry_feature_dim + 1, 192), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(192, 64), nn.ReLU(inplace=True), nn.Linear(64, 1),
        )

    @staticmethod
    def _pair_features(encoder: nn.Module, query_images: torch.Tensor, candidate_images: torch.Tensor) -> torch.Tensor:
        batch, candidates = candidate_images.shape[:2]
        query_feature = encoder(query_images)
        candidate_feature = encoder(candidate_images.reshape(batch * candidates, *candidate_images.shape[2:]))
        candidate_feature = candidate_feature.reshape(batch, candidates, -1)
        query_feature = query_feature[:, None].expand(-1, candidates, -1)
        return torch.cat([query_feature, candidate_feature, torch.abs(query_feature - candidate_feature)], dim=2)

    def forward(
        self,
        query_detail: torch.Tensor,
        candidate_detail: torch.Tensor,
        query_context: torch.Tensor,
        candidate_context: torch.Tensor,
        query_geometry: torch.Tensor,
        candidate_geometry: torch.Tensor,
        hand_scores: torch.Tensor,
    ) -> torch.Tensor:
        batch, candidates = candidate_detail.shape[:2]
        detail = self._pair_features(self.detail_encoder, query_detail, candidate_detail)
        context = self._pair_features(self.context_encoder, query_context, candidate_context)
        query_geometry = query_geometry[:, None].expand(-1, candidates, -1)
        geometry = torch.cat([query_geometry, candidate_geometry, torch.abs(query_geometry - candidate_geometry)], dim=2)
        return self.head(torch.cat([detail, context, geometry, hand_scores[:, :, None]], dim=2)).squeeze(-1)


def normalized_features(
    rows: list[dict[str, str]], coordinates: list[tuple[float, float]], geometry_mean: np.ndarray, geometry_std: np.ndarray,
    detail_mean: np.ndarray, detail_std: np.ndarray, context_mean: np.ndarray, context_std: np.ndarray,
    detail_size: int, context_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    detail_patches, context_patches, geometry, detail_hand, context_hand = [], [], [], [], []
    for row, (x, y) in zip(rows, coordinates, strict=True):
        detail = crop_contact_patch(row["vision_path"], x, y, detail_size)
        context = crop_contact_patch(row["vision_path"], x, y, context_size)
        detail_patches.append(detail)
        context_patches.append(context)
        geometry.append((motion_geometry_feature(row, x, y) - geometry_mean) / geometry_std)
        detail_hand.append((visual_patch_feature_from_patch(detail) - detail_mean) / detail_std)
        context_hand.append((visual_patch_feature_from_patch(context) - context_mean) / context_std)
    return (
        np.stack(detail_patches).astype(np.float32), np.stack(context_patches).astype(np.float32),
        np.stack(geometry).astype(np.float32), np.stack(detail_hand).astype(np.float32), np.stack(context_hand).astype(np.float32),
    )


def build_shortlists(
    query_rows: list[dict[str, str]], query_geometry: np.ndarray, query_detail_hand: np.ndarray, query_context_hand: np.ndarray,
    query_embeddings: np.ndarray, cache_rows: list[dict[str, str]], cache_geometry: np.ndarray,
    cache_detail_hand: np.ndarray, cache_context_hand: np.ndarray, cache_embeddings: np.ndarray, filter_k: int,
    exclude_same_record: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    candidates, targets, hand_scores, geometry_values, detail_values, context_values = [], [], [], [], [], []
    cache_record_ids = np.asarray([row["record_id"] for row in cache_rows])
    for index, row in enumerate(query_rows):
        geometry_distance = np.linalg.norm(cache_geometry - query_geometry[index][None], axis=1)
        allowed = np.ones(len(cache_rows), dtype=bool)
        if exclude_same_record:
            allowed = cache_record_ids != row["record_id"]
        indices = np.flatnonzero(allowed)
        local_k = min(filter_k, len(indices))
        shortlist = indices[np.argpartition(geometry_distance[indices], local_k - 1)[:local_k]]
        shortlist = shortlist[np.argsort(geometry_distance[shortlist], kind="stable")]
        detail_distance = np.linalg.norm(cache_detail_hand[shortlist] - query_detail_hand[index][None], axis=1)
        context_distance = np.linalg.norm(cache_context_hand[shortlist] - query_context_hand[index][None], axis=1)
        hand = (
            geometry_distance[shortlist] / math.sqrt(cache_geometry.shape[1])
            + detail_distance / math.sqrt(cache_detail_hand.shape[1])
            + context_distance / math.sqrt(cache_context_hand.shape[1])
        )
        candidates.append(shortlist.astype(np.int32))
        targets.append(np.linalg.norm(cache_embeddings[shortlist] - query_embeddings[index][None], axis=1).astype(np.float32))
        hand_scores.append(hand.astype(np.float32))
        geometry_values.append(geometry_distance[shortlist].astype(np.float32))
        detail_values.append(detail_distance.astype(np.float32))
        context_values.append(context_distance.astype(np.float32))
    return tuple(np.stack(values) for values in (candidates, targets, hand_scores, geometry_values, detail_values, context_values))


def score_model(
    model: MultiScaleTactileRanker, candidates: np.ndarray, hand_scores: np.ndarray, query_detail: np.ndarray,
    query_context: np.ndarray, query_geometry: np.ndarray, cache_detail: np.ndarray, cache_context: np.ndarray,
    cache_geometry: np.ndarray, device: torch.device, batch_size: int,
) -> np.ndarray:
    model.eval()
    results = []
    with torch.no_grad():
        for start in range(0, len(query_detail), batch_size):
            end = start + batch_size
            indices = candidates[start:end]
            batch, local_k = indices.shape
            scores = model(
                image_tensor(query_detail[start:end]).to(device),
                image_tensor(cache_detail[indices].reshape(-1, *cache_detail.shape[1:])).reshape(batch, local_k, 3, *cache_detail.shape[1:3]).to(device),
                image_tensor(query_context[start:end]).to(device),
                image_tensor(cache_context[indices].reshape(-1, *cache_context.shape[1:])).reshape(batch, local_k, 3, *cache_context.shape[1:3]).to(device),
                torch.from_numpy(query_geometry[start:end]).to(device), torch.from_numpy(cache_geometry[indices]).to(device),
                torch.from_numpy(hand_scores[start:end]).to(device),
            )
            results.append(scores.cpu().numpy())
    return np.concatenate(results)


def train_fold(
    fold: str, rows: list[dict[str, str]], fold_by_name: dict[str, str], prediction_by_name: dict[str, dict[str, str]],
    cfg: dict, device: torch.device,
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict]:
    fit_rows = [row for row in rows if fold_by_name[row["image_name"]] != fold]
    query_rows = [row for row in rows if fold_by_name[row["image_name"]] == fold]
    if {row["record_id"] for row in fit_rows} & {row["record_id"] for row in query_rows}:
        raise RuntimeError(f"Fold {fold} has record leakage between ranker fit and OOF queries.")
    detail_size, context_size = int(cfg["detail_crop_size"]), int(cfg["context_crop_size"])
    filter_k = min(int(cfg["geometry_filter_k"]), len(fit_rows))
    threshold, batch_size = float(cfg["tactile_mask_threshold"]), int(cfg["batch_size"])

    raw_fit_geometry = np.stack([motion_geometry_feature(row, float(row["target_tip_x"]), float(row["target_tip_y"])) for row in fit_rows])
    _, geometry_mean, geometry_std = standardize(raw_fit_geometry, raw_fit_geometry)
    fit_gt_coordinates = [(float(row["target_tip_x"]), float(row["target_tip_y"])) for row in fit_rows]
    all_gt_coordinates = [(float(row["target_tip_x"]), float(row["target_tip_y"])) for row in rows]
    raw_fit_detail = np.stack([visual_patch_feature_from_patch(crop_contact_patch(row["vision_path"], x, y, detail_size)) for row, (x, y) in zip(fit_rows, fit_gt_coordinates, strict=True)])
    raw_fit_context = np.stack([visual_patch_feature_from_patch(crop_contact_patch(row["vision_path"], x, y, context_size)) for row, (x, y) in zip(fit_rows, fit_gt_coordinates, strict=True)])
    _, detail_mean, detail_std = standardize(raw_fit_detail, raw_fit_detail)
    _, context_mean, context_std = standardize(raw_fit_context, raw_fit_context)
    fit_detail, fit_context, fit_geometry, fit_detail_hand, fit_context_hand = normalized_features(
        fit_rows, fit_gt_coordinates, geometry_mean, geometry_std, detail_mean, detail_std, context_mean, context_std, detail_size, context_size,
    )
    cache_detail, cache_context, cache_geometry, cache_detail_hand, cache_context_hand = normalized_features(
        rows, all_gt_coordinates, geometry_mean, geometry_std, detail_mean, detail_std, context_mean, context_std, detail_size, context_size,
    )
    fit_predicted_coordinates = [(float(prediction_by_name[row["image_name"]]["pred_x"]), float(prediction_by_name[row["image_name"]]["pred_y"])) for row in fit_rows]
    query_coordinates = [(float(prediction_by_name[row["image_name"]]["pred_x"]), float(prediction_by_name[row["image_name"]]["pred_y"])) for row in query_rows]
    fit_query_detail, fit_query_context, fit_query_geometry, fit_query_detail_hand, fit_query_context_hand = normalized_features(
        fit_rows, fit_predicted_coordinates, geometry_mean, geometry_std, detail_mean, detail_std, context_mean, context_std, detail_size, context_size,
    )
    query_detail, query_context, query_geometry, query_detail_hand, query_context_hand = normalized_features(
        query_rows, query_coordinates, geometry_mean, geometry_std, detail_mean, detail_std, context_mean, context_std, detail_size, context_size,
    )

    tactile_cache: dict[str, np.ndarray] = {}
    def touch(row: dict[str, str]) -> np.ndarray:
        return tactile_difference(row["touch_path"], tactile_cache, int(cfg["tactile_size"]))

    cache_embeddings = np.stack([tactile_embedding(touch(row)) for row in rows]).astype(np.float32)
    fit_embeddings = np.stack([tactile_embedding(touch(row)) for row in fit_rows]).astype(np.float32)
    query_embeddings = np.stack([tactile_embedding(touch(row)) for row in query_rows]).astype(np.float32)
    fit_groups = build_shortlists(
        fit_rows, fit_query_geometry, fit_query_detail_hand, fit_query_context_hand, fit_embeddings,
        fit_rows, fit_geometry, fit_detail_hand, fit_context_hand, fit_embeddings, filter_k, True,
    )
    query_groups = build_shortlists(
        query_rows, query_geometry, query_detail_hand, query_context_hand, query_embeddings,
        rows, cache_geometry, cache_detail_hand, cache_context_hand, cache_embeddings, min(filter_k, len(rows)), True,
    )
    fit_candidates, fit_targets, fit_hand, *_ = fit_groups
    candidates, targets, hand, geometry_values, detail_values, context_values = query_groups
    model = MultiScaleTactileRanker(fit_geometry.shape[1], float(cfg.get("dropout", 0.1))).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]))
    target_std = max(float(fit_targets.std()), 1e-6)
    temperature, listwise_weight = float(cfg["target_temperature"]), float(cfg["listwise_weight"])
    for _ in range(int(cfg["epochs"])):
        model.train()
        order = np.random.permutation(len(fit_rows))
        for start in range(0, len(order), batch_size):
            sample = order[start:start + batch_size]
            local_candidates = fit_candidates[sample]
            batch, local_k = local_candidates.shape
            scores = model(
                image_tensor(fit_query_detail[sample]).to(device), image_tensor(fit_detail[local_candidates].reshape(-1, *fit_detail.shape[1:])).reshape(batch, local_k, 3, *fit_detail.shape[1:3]).to(device),
                image_tensor(fit_query_context[sample]).to(device), image_tensor(fit_context[local_candidates].reshape(-1, *fit_context.shape[1:])).reshape(batch, local_k, 3, *fit_context.shape[1:3]).to(device),
                torch.from_numpy(fit_query_geometry[sample]).to(device), torch.from_numpy(fit_geometry[local_candidates]).to(device), torch.from_numpy(fit_hand[sample]).to(device),
            )
            target = torch.from_numpy(fit_targets[sample]).to(device)
            regression = nn.functional.smooth_l1_loss((scores - target) / target_std, torch.zeros_like(scores))
            distribution = torch.softmax(-target / temperature, dim=1)
            listwise = -(distribution * torch.log_softmax(-scores / temperature, dim=1)).sum(dim=1).mean()
            optimizer.zero_grad(set_to_none=True)
            (regression + listwise_weight * listwise).backward()
            optimizer.step()

    scores = score_model(model, candidates, hand, query_detail, query_context, query_geometry, cache_detail, cache_context, cache_geometry, device, batch_size)
    checkpoint_dir = project_path(cfg["checkpoint_dir"])
    ensure_dir(checkpoint_dir)
    torch.save({"model_state": model.state_dict(), "fold": fold, "fit_records": len({row['record_id'] for row in fit_rows})}, checkpoint_dir / f"fold_{fold}.pt")
    query_output, candidate_output, metric_cache = [], [], {}
    for index, query in enumerate(query_rows):
        order = np.argsort(scores[index], kind="stable")
        hand_order = np.argsort(hand[index], kind="stable")
        selected = int(order[0])
        top3 = order[:3]
        selected_cache = rows[int(candidates[index, selected])]
        selected_metrics = tactile_metrics(touch(query), tactile_difference(selected_cache["touch_path"], metric_cache, int(cfg["tactile_size"])), threshold)
        top3_embeddings = cache_embeddings[candidates[index, top3]]
        disagreement = float(np.mean(np.linalg.norm(top3_embeddings[:, None] - top3_embeddings[None, :], axis=2)))
        best, second = float(scores[index, order[0]]), float(scores[index, order[1]])
        hand_best, hand_second = float(hand[index, hand_order[0]]), float(hand[index, hand_order[1]])
        prediction = prediction_by_name[query["image_name"]]
        query_output.append({
            "query_record_id": query["record_id"], "query_image_name": query["image_name"], "query_probe": query["probe"], "oof_fold": fold,
            "pred_x": f"{query_coordinates[index][0]:.3f}", "pred_y": f"{query_coordinates[index][1]:.3f}", "c2_pred_score": prediction.get("pred_score", ""),
            "ranker_best_score": f"{best:.6f}", "ranker_second_score": f"{second:.6f}", "ranker_margin": f"{second - best:.6f}", "ranker_margin_normalized": f"{(second - best) / max(float(scores[index].std()), 1e-6):.6f}",
            "hand_best_score": f"{hand_best:.6f}", "hand_second_score": f"{hand_second:.6f}", "hand_margin": f"{hand_second - hand_best:.6f}",
            "ranker_oracle_embedding_rank": str(int(ranks(scores[index])[int(np.argmin(targets[index]))])),
            "selected_cache_record_id": selected_cache["record_id"], "selected_cache_image_name": selected_cache["image_name"],
            "top3_cache_record_ids": "|".join(rows[int(candidates[index, item])]["record_id"] for item in top3),
            "top3_cache_image_names": "|".join(rows[int(candidates[index, item])]["image_name"] for item in top3),
            "top3_tactile_embedding_disagreement": f"{disagreement:.6f}", "top3_score_std": f"{float(scores[index, top3].std()):.6f}",
            "geometry_distance": f"{float(geometry_values[index, selected]):.6f}", "detail_visual_distance": f"{float(detail_values[index, selected]):.6f}", "context_visual_distance": f"{float(context_values[index, selected]):.6f}",
            "trajectory_real_point_count": query["trajectory_real_point_count"], "trajectory_history_span_frames": query["trajectory_history_span_frames"],
            "trajectory_padding_ratio": query["trajectory_padding_ratio"], "trajectory_cumulative_displacement": query["trajectory_cumulative_displacement"],
            **{key: f"{value:.6f}" for key, value in selected_metrics.items()},
        })
        for rank, item in enumerate(order, start=1):
            cache = rows[int(candidates[index, item])]
            candidate_output.append({
                "query_record_id": query["record_id"], "query_image_name": query["image_name"], "query_probe": query["probe"], "oof_fold": fold,
                "candidate_rank": str(rank), "candidate_score": f"{float(scores[index, item]):.6f}", "hand_score": f"{float(hand[index, item]):.6f}",
                "geometry_distance": f"{float(geometry_values[index, item]):.6f}", "detail_visual_distance": f"{float(detail_values[index, item]):.6f}", "context_visual_distance": f"{float(context_values[index, item]):.6f}",
                "candidate_record_id": cache["record_id"], "candidate_image_name": cache["image_name"], "candidate_tactile_embedding_distance": f"{float(targets[index, item]):.6f}",
            })
    return query_output, candidate_output, {"fold": fold, "fit_records": len({row['record_id'] for row in fit_rows}), "oof_records": len({row['record_id'] for row in query_rows}), "oof_queries": len(query_rows)}


def summarize(rows: list[dict[str, str]]) -> dict:
    result = {"queries": len(rows)}
    for label, selected in (("all", rows), ("far_probe75_100", [row for row in rows if int(row["query_probe"]) >= 75])):
        result[label] = {
            "queries": len(selected),
            **{f"mean_{metric}": float(np.mean([float(row[metric]) for row in selected])) if selected else None for metric in TACTILE_METRICS},
            "tactile_best_top1_rate": float(np.mean([int(row["ranker_oracle_embedding_rank"]) == 1 for row in selected])) if selected else None,
            "tactile_best_top3_rate": float(np.mean([int(row["ranker_oracle_embedding_rank"]) <= 3 for row in selected])) if selected else None,
        }
    return result


def build(config_path: str, section: str) -> dict:
    cfg = load_config(config_path)[section]
    set_seed(int(cfg["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = read_csv_rows(project_path(cfg["samples_csv"]))
    if any(is_final_holdout(row) for row in rows):
        raise RuntimeError("Refusing to access sealed final-holdout samples.")
    rows = [row for row in rows if row["dataset_split"] == "train"]
    predictions = read_csv_rows(project_path(cfg["oof_predictions_csv"]))
    prediction_by_name = prediction_map(predictions, rows, "train", "Phase4E strict OOF")
    fold_by_name = {name: prediction["oof_fold"] for name, prediction in prediction_by_name.items()}
    folds = sorted(set(fold_by_name.values()))
    if len(folds) < 2 or any(not fold for fold in folds):
        raise RuntimeError(f"Expected multiple OOF folds, got {folds}")
    query_rows, candidate_rows, folds_summary = [], [], []
    for fold in folds:
        fold_queries, fold_candidates, fold_summary = train_fold(fold, rows, fold_by_name, prediction_by_name, cfg, device)
        query_rows.extend(fold_queries)
        candidate_rows.extend(fold_candidates)
        folds_summary.append(fold_summary)
    if len(query_rows) != len(rows) or len({row["query_image_name"] for row in query_rows}) != len(rows):
        raise RuntimeError("OOF query output must cover every development-pool train query exactly once.")
    if any(is_final_holdout(row) for row in query_rows):
        raise RuntimeError("OOF output includes sealed final-holdout data.")
    write_csv_rows(project_path(cfg["query_output_csv"]), query_rows, QUERY_FIELDS)
    write_csv_rows(project_path(cfg["candidate_output_csv"]), candidate_rows, CANDIDATE_FIELDS)
    summary = {
        "mode": "phase4e_strict_oof_multiscale_cache_candidates", "device": str(device), "folds": folds_summary,
        "detail_crop_size": int(cfg["detail_crop_size"]), "context_crop_size": int(cfg["context_crop_size"]), "geometry_filter_k": int(cfg["geometry_filter_k"]),
        "query_summary": summarize(query_rows),
        "integrity": {
            "contact_predictions": "strict record-level C2 OOF", "ranker_fit": "each ranker fold excludes its OOF query records",
            "query_cache": "same-record cache entries excluded", "sealed_final_holdout_rows_read": 0,
            "query_tactile_usage": "offline ranking supervision and evaluation only; never an inference input",
        },
    }
    write_json(project_path(cfg["metrics_json"]), summary)
    print(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build strict OOF multi-scale cache candidates for Phase 4E.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--section", default="phase4e_oof_multiscale_cache_v1")
    args = parser.parse_args()
    build(args.config, args.section)


if __name__ == "__main__":
    main()
