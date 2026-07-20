from __future__ import annotations

import argparse
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn

from .build_cache_retrieval import crop_contact_patch_from_image, visual_patch_feature_from_patch
from .config import load_config, project_path
from .temporal_progress import DEFAULT_TTC_VALUES, masked_trajectory_features, motion_basis, read_trajectory_tracks
from .train_proposal_ranker import ProposalRanker, parse_points, recent_motion, soft_ttc_features
from .utils import read_csv_rows, write_csv_rows, write_json


OUTPUT_FIELDS = [
    "dataset_split", "record_id", "image_name", "probe", "case_type", "top1_x", "top1_y",
    "ranked_x", "ranked_y", "top1_error_px", "ranked_error_px", "top1_box48", "ranked_box48",
    "selected_original_rank", "gate_selected", "ranker_advantage", "candidate_ranker_scores",
]

DIAGNOSTIC_CASE_FIELDS = [
    "dataset_split", "record_id", "image_name", "probe", "case_type",
    "c2_top1_rank", "raw_ranker_rank", "positive_rank", "raw_ranker_box48",
    "c2_top1_box48", "positive_score", "c2_top1_score", "positive_minus_c2_score",
    "raw_best_score", "raw_best_minus_c2_score", "correct_candidate_rank",
    "top1_heatmap_score", "top2_heatmap_score", "top2_to_top1_ratio",
    "predicted_ttc", "motion_speed", "motion_stability", "motion_cumulative",
    "trajectory_padding_ratio", "trajectory_real_point_count", "trajectory_history_span_frames",
]


FEATURE_MODE_SLICES = {
    "heatmap": (0, 4),
    "heatmap_geometry": (0, 28),
    "all": None,
}
BASE_FEATURE_SIZE = 28
FULL_FEATURE_SIZE_WITH_VISUAL = 119


class LinearProposalRanker(nn.Module):
    """Low-capacity scorer for testing whether MLP capacity causes OOF memorization."""

    def __init__(self, feature_size: int) -> None:
        super().__init__()
        self.scorer = nn.Linear(feature_size, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.scorer(features).squeeze(-1)


def build_ranker(feature_size: int, cfg: dict) -> nn.Module:
    architecture = str(cfg.get("ranker_architecture", "mlp"))
    if architecture == "linear":
        return LinearProposalRanker(feature_size)
    if architecture == "mlp":
        return ProposalRanker(feature_size, int(cfg.get("hidden_size", 32)), float(cfg.get("dropout", 0.1)))
    raise ValueError(f"Unknown ranker_architecture={architecture}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def classify_group(group: dict) -> str:
    if group["candidates"][0]["box48"]:
        return "easy"
    if any(candidate["box48"] for candidate in group["candidates"]):
        return "rank_hard"
    return "proposal_miss"


def build_group(
    row: dict[str, str],
    prediction: dict[str, str],
    trajectory: np.ndarray,
    trajectory_mask: np.ndarray,
    crop_size: int,
    contextual_candidate_features: bool = False,
) -> dict:
    points = parse_points(prediction["topk_points"])
    target = np.asarray([float(row["target_tip_x"]), float(row["target_tip_y"])], dtype=np.float32)
    width, height = float(row["image_width"]), float(row["image_height"])
    tip = np.asarray([float(row["tip_x"]), float(row["tip_y"])], dtype=np.float32)
    predicted_delta = np.asarray([
        float(prediction.get("predicted_delta_x") or 0.0),
        float(prediction.get("predicted_delta_y") or 0.0),
    ], dtype=np.float32)
    endpoint = tip + predicted_delta
    pose_direction = np.asarray([float(row["direction_x"]), float(row["direction_y"])], dtype=np.float32)
    pose_norm = float(np.linalg.norm(pose_direction))
    if pose_norm > 1e-6:
        pose_direction /= pose_norm
    direction, speed, stability, cumulative = recent_motion(trajectory, trajectory_mask, pose_direction)
    perpendicular = np.asarray([-direction[1], direction[0]], dtype=np.float32)
    pose_perpendicular = np.asarray([-pose_direction[1], pose_direction[0]], dtype=np.float32)
    predicted_ttc = float(prediction.get("predicted_ttc") or 0.0)
    ttc_features = soft_ttc_features(predicted_ttc, np.asarray(DEFAULT_TTC_VALUES, dtype=np.float32))
    top_score = max(float(points[0][2]), 1e-6)
    image = Image.open(row["vision_path"]).convert("RGB")
    candidates = []
    for rank, (x, y, heatmap_score) in enumerate(points):
        point = np.asarray([x, y], dtype=np.float32)
        tip_vector = point - tip
        endpoint_vector = point - endpoint
        base = np.asarray([
            heatmap_score, math.log(max(heatmap_score, 1e-8)), heatmap_score / top_score, rank / max(len(points) - 1, 1),
            x / width, y / height,
            tip_vector[0] / 48.0, tip_vector[1] / 48.0, float(np.linalg.norm(tip_vector)) / 48.0,
            float(np.dot(tip_vector, direction)) / 48.0, float(np.dot(tip_vector, perpendicular)) / 48.0,
            endpoint_vector[0] / 48.0, endpoint_vector[1] / 48.0, float(np.linalg.norm(endpoint_vector)) / 48.0,
            float(np.dot(tip_vector, pose_direction)) / 48.0, float(np.dot(tip_vector, pose_perpendicular)) / 48.0,
            speed, stability, cumulative, predicted_ttc / 100.0, float(row.get("trajectory_padding_ratio", 0.0)),
        ], dtype=np.float32)
        patch = crop_contact_patch_from_image(image, x, y, crop_size)
        feature = np.concatenate([base, ttc_features, visual_patch_feature_from_patch(patch)], axis=0)
        error = float(np.linalg.norm(point - target))
        candidates.append({
            "x": x, "y": y, "error": error, "box48": abs(x - target[0]) <= 24.0 and abs(y - target[1]) <= 24.0,
            "feature": feature, "heatmap_score": float(heatmap_score),
        })
    if contextual_candidate_features:
        score_ratios = np.asarray([item["heatmap_score"] / top_score for item in candidates], dtype=np.float32)
        top_point = np.asarray([candidates[0]["x"], candidates[0]["y"]], dtype=np.float32)
        for candidate in candidates:
            relative = np.asarray([candidate["x"], candidate["y"]], dtype=np.float32) - top_point
            context = np.concatenate([
                score_ratios,
                np.asarray([
                    relative[0] / 48.0,
                    relative[1] / 48.0,
                    float(np.linalg.norm(relative)) / 48.0,
                    float(np.dot(relative, direction)) / 48.0,
                    float(np.dot(relative, perpendicular)) / 48.0,
                ], dtype=np.float32),
            ])
            candidate["feature"] = np.concatenate([candidate["feature"], context], axis=0)
    return {
        "row": row,
        "prediction": prediction,
        "candidates": candidates,
        "features": np.stack([item["feature"] for item in candidates]),
        "predicted_ttc": predicted_ttc,
        "motion_speed": speed,
        "motion_stability": stability,
        "motion_cumulative": cumulative,
    }


def load_groups(config: dict, cfg: dict, prediction_csv: str, expected_split: str) -> list[dict]:
    model_cfg = config[str(cfg["contact_model_section"])]
    rows_by_name = {row["image_name"]: row for row in read_csv_rows(project_path(model_cfg["samples_csv"]))}
    tracks = read_trajectory_tracks(project_path(model_cfg["motion_tracks_csv"]))
    history = int(model_cfg["trajectory_history_frames"])
    spatial_scale = float(model_cfg["trajectory_spatial_scale_px"])
    speed_scale = float(model_cfg["trajectory_speed_scale_px"])
    groups = []
    for prediction in read_csv_rows(project_path(prediction_csv)):
        if prediction["dataset_split"] != expected_split:
            continue
        row = rows_by_name[prediction["image_name"]]
        if row["dataset_split"] != expected_split:
            raise ValueError(f"Split mismatch for {row['image_name']}")
        trajectory, mask, _ = masked_trajectory_features(row, tracks, history, spatial_scale, speed_scale)
        groups.append(build_group(
            row,
            prediction,
            trajectory,
            mask,
            int(cfg.get("cache_crop_size", 48)),
            bool(cfg.get("contextual_candidate_features", False)),
        ))
    if not groups:
        raise ValueError(f"No {expected_split} groups in {prediction_csv}")
    return groups


def normalize(groups: list[dict], mean: np.ndarray, std: np.ndarray) -> torch.Tensor:
    features = np.stack([group["features"] for group in groups]).astype(np.float32)
    return torch.from_numpy((features - mean[None, None]) / std[None, None]).float()


def select_feature_mode(groups: list[dict], feature_mode: str) -> None:
    """Keep a named prefix of the fixed candidate descriptor for ablation."""
    if feature_mode == "heatmap_geometry_context":
        for group in groups:
            features = group["features"]
            if features.shape[-1] <= FULL_FEATURE_SIZE_WITH_VISUAL:
                raise ValueError("heatmap_geometry_context requires contextual_candidate_features=true")
            group["features"] = np.concatenate(
                [features[:, :BASE_FEATURE_SIZE], features[:, FULL_FEATURE_SIZE_WITH_VISUAL:]], axis=-1
            )
        return
    if feature_mode not in FEATURE_MODE_SLICES:
        raise ValueError(f"Unknown feature_mode={feature_mode}; expected one of {sorted(FEATURE_MODE_SLICES)}")
    feature_slice = FEATURE_MODE_SLICES[feature_mode]
    if feature_slice is None:
        return
    start, end = feature_slice
    for group in groups:
        group["features"] = group["features"][:, start:end]


def group_indices(groups: list[dict]) -> tuple[list[int], list[int]]:
    hard = [index for index, group in enumerate(groups) if classify_group(group) == "rank_hard"]
    easy = [index for index, group in enumerate(groups) if classify_group(group) == "easy"]
    return hard, easy


def hard_loss(scores: torch.Tensor, groups: list[dict], indices: list[int], margin: float, aux_weight: float) -> torch.Tensor:
    losses = []
    weights = []
    for index in indices:
        candidates = groups[index]["candidates"]
        positives = [rank for rank, item in enumerate(candidates) if item["box48"]]
        positive = min(positives, key=lambda rank: candidates[rank]["error"])
        pair_loss = nn.functional.softplus(margin - (scores[index, positive] - scores[index, 0]))
        negatives = [rank for rank, item in enumerate(candidates) if rank != positive and not item["box48"] and rank != 0]
        if negatives:
            auxiliary = torch.stack([nn.functional.softplus(margin - (scores[index, positive] - scores[index, rank])) for rank in negatives]).mean()
            pair_loss = pair_loss + aux_weight * auxiliary
        probe = int(groups[index]["row"]["probe"])
        losses.append(pair_loss)
        weights.append(6.0 if probe >= 75 else 4.0)
    return torch.sum(torch.stack(losses) * torch.tensor(weights, device=scores.device)) / sum(weights)


def stability_loss(scores: torch.Tensor, indices: list[int], margin: float) -> torch.Tensor:
    if not indices:
        return scores.sum() * 0.0
    values = [nn.functional.softplus(margin - (scores[index, 0] - torch.max(scores[index, 1:]))) for index in indices]
    return torch.stack(values).mean()


def percentile(values: list[float], q: float) -> float | None:
    return float(np.percentile(values, q)) if values else None


def model_scores(model: ProposalRanker, features: torch.Tensor, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(features.to(device)).cpu().numpy()


def score_validation(
    model: ProposalRanker,
    features: torch.Tensor,
    groups: list[dict],
    device: torch.device,
    margin: float,
) -> tuple[list[dict[str, str]], dict]:
    scores = model_scores(model, features, device)
    rows = []
    for group, group_scores in zip(groups, scores):
        candidates = group["candidates"]
        row = group["row"]
        best = int(np.argmax(group_scores))
        advantage = float(group_scores[best] - group_scores[0])
        use_ranker = int(row["probe"]) >= 75 and math.isfinite(margin) and best != 0 and advantage >= margin
        selected = best if use_ranker else 0
        original = candidates[0]
        ranked = candidates[selected]
        rows.append({
            "dataset_split": "val", "record_id": row["record_id"], "image_name": row["image_name"], "probe": row["probe"],
            "case_type": classify_group(group), "top1_x": f"{original['x']:.3f}", "top1_y": f"{original['y']:.3f}",
            "ranked_x": f"{ranked['x']:.3f}", "ranked_y": f"{ranked['y']:.3f}",
            "top1_error_px": f"{original['error']:.3f}", "ranked_error_px": f"{ranked['error']:.3f}",
            "top1_box48": "1" if original["box48"] else "0", "ranked_box48": "1" if ranked["box48"] else "0",
            "selected_original_rank": str(selected + 1), "gate_selected": "1" if use_ranker else "0",
            "ranker_advantage": f"{advantage:.6f}", "candidate_ranker_scores": ";".join(f"{value:.6f}" for value in group_scores),
        })
    far = [row for row in rows if int(row["probe"]) >= 75]
    far_hard = [row for row in far if row["case_type"] == "rank_hard"]
    far_easy = [row for row in far if row["case_type"] == "easy"]
    before = sum(row["top1_box48"] == "1" for row in far) / len(far)
    after = sum(row["ranked_box48"] == "1" for row in far) / len(far)
    metrics = {
        "far_samples": len(far), "far_box48_before": before, "far_box48_after": after, "far_box48_delta": after - before,
        "far_rank_hard_total": len(far_hard),
        "far_rank_hard_corrected": sum(row["top1_box48"] == "0" and row["ranked_box48"] == "1" for row in far_hard),
        "far_easy_total": len(far_easy),
        "far_easy_corruption": sum(row["top1_box48"] == "1" and row["ranked_box48"] == "0" for row in far_easy),
        "gate_selected_queries": sum(row["gate_selected"] == "1" for row in far),
        "far_p75_error_px": percentile([float(row["ranked_error_px"]) for row in far], 75),
        "far_p90_error_px": percentile([float(row["ranked_error_px"]) for row in far], 90),
    }
    metrics["passed"] = (
        metrics["far_rank_hard_corrected"] >= 2
        and metrics["far_box48_delta"] >= 0.05
        and metrics["far_easy_corruption"] <= 1
    )
    return rows, metrics


def scalar_summary(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p10": None, "p90": None}
    return {
        "count": len(values), "mean": float(np.mean(values)), "median": float(np.median(values)),
        "p10": percentile(values, 10), "p90": percentile(values, 90),
    }


def raw_ranking_rows(groups: list[dict], scores: np.ndarray, dataset_split: str) -> list[dict[str, str]]:
    rows = []
    for group, group_scores in zip(groups, scores):
        candidates = group["candidates"]
        row = group["row"]
        selected = int(np.argmax(group_scores))
        original = candidates[0]
        ranked = candidates[selected]
        rows.append({
            "dataset_split": dataset_split, "record_id": row["record_id"], "image_name": row["image_name"], "probe": row["probe"],
            "case_type": classify_group(group), "top1_x": f"{original['x']:.3f}", "top1_y": f"{original['y']:.3f}",
            "ranked_x": f"{ranked['x']:.3f}", "ranked_y": f"{ranked['y']:.3f}",
            "top1_error_px": f"{original['error']:.3f}", "ranked_error_px": f"{ranked['error']:.3f}",
            "top1_box48": "1" if original["box48"] else "0", "ranked_box48": "1" if ranked["box48"] else "0",
            "selected_original_rank": str(selected + 1), "gate_selected": "1" if selected != 0 else "0",
            "ranker_advantage": f"{float(group_scores[selected] - group_scores[0]):.6f}",
            "candidate_ranker_scores": ";".join(f"{value:.6f}" for value in group_scores),
        })
    return rows


def diagnostic_cases(groups: list[dict], scores: np.ndarray, dataset_split: str) -> list[dict[str, str]]:
    rows = []
    for group, group_scores in zip(groups, scores):
        candidates = group["candidates"]
        case_type = classify_group(group)
        if case_type != "rank_hard":
            continue
        positive = min((index for index, item in enumerate(candidates) if item["box48"]), key=lambda index: candidates[index]["error"])
        raw_best = int(np.argmax(group_scores))
        correct_rank = min(index for index, item in enumerate(candidates) if item["box48"])
        top1_score = candidates[0]["heatmap_score"]
        top2_score = candidates[1]["heatmap_score"] if len(candidates) > 1 else 0.0
        row = group["row"]
        rows.append({
            "dataset_split": dataset_split, "record_id": row["record_id"], "image_name": row["image_name"], "probe": row["probe"],
            "case_type": case_type, "c2_top1_rank": "1", "raw_ranker_rank": str(raw_best + 1), "positive_rank": str(positive + 1),
            "raw_ranker_box48": "1" if candidates[raw_best]["box48"] else "0", "c2_top1_box48": "0",
            "positive_score": f"{float(group_scores[positive]):.6f}", "c2_top1_score": f"{float(group_scores[0]):.6f}",
            "positive_minus_c2_score": f"{float(group_scores[positive] - group_scores[0]):.6f}",
            "raw_best_score": f"{float(group_scores[raw_best]):.6f}", "raw_best_minus_c2_score": f"{float(group_scores[raw_best] - group_scores[0]):.6f}",
            "correct_candidate_rank": str(correct_rank + 1), "top1_heatmap_score": f"{top1_score:.6f}",
            "top2_heatmap_score": f"{top2_score:.6f}", "top2_to_top1_ratio": f"{top2_score / max(top1_score, 1e-8):.6f}",
            "predicted_ttc": f"{group['predicted_ttc']:.6f}", "motion_speed": f"{group['motion_speed']:.6f}",
            "motion_stability": f"{group['motion_stability']:.6f}", "motion_cumulative": f"{group['motion_cumulative']:.6f}",
            "trajectory_padding_ratio": row.get("trajectory_padding_ratio", ""),
            "trajectory_real_point_count": row.get("trajectory_real_point_count", ""),
            "trajectory_history_span_frames": row.get("trajectory_history_span_frames", ""),
        })
    return rows


def raw_ranker_metrics(rows: list[dict[str, str]]) -> dict:
    far = [row for row in rows if int(row["probe"]) >= 75]
    far_hard = [row for row in far if row["case_type"] == "rank_hard"]
    far_easy = [row for row in far if row["case_type"] == "easy"]
    return {
        "far_samples": len(far),
        "far_box48": float(np.mean([row["ranked_box48"] == "1" for row in far])) if far else None,
        "far_rank_hard_total": len(far_hard),
        "far_rank_hard_corrected": sum(row["ranked_box48"] == "1" for row in far_hard),
        "far_easy_total": len(far_easy),
        "far_easy_corruption": sum(row["ranked_box48"] != "1" for row in far_easy),
        "far_p75_error_px": percentile([float(row["ranked_error_px"]) for row in far], 75),
        "far_p90_error_px": percentile([float(row["ranked_error_px"]) for row in far], 90),
    }


def build_diagnostics(
    train_groups: list[dict], train_scores: np.ndarray, val_groups: list[dict], val_scores: np.ndarray,
) -> tuple[dict, list[dict[str, str]], list[dict[str, str]]]:
    train_cases = diagnostic_cases(train_groups, train_scores, "train_oof")
    val_cases = diagnostic_cases(val_groups, val_scores, "val")
    raw_train_rows = raw_ranking_rows(train_groups, train_scores, "train_oof")
    raw_val_rows = raw_ranking_rows(val_groups, val_scores, "val")

    def pairwise_summary(cases: list[dict[str, str]]) -> dict:
        gaps = [float(row["positive_minus_c2_score"]) for row in cases]
        return {
            "hard_queries": len(cases),
            "positive_beats_c2_rate": float(np.mean([gap > 0.0 for gap in gaps])) if gaps else None,
            "positive_minus_c2": scalar_summary(gaps),
            "raw_best_box48_rate": float(np.mean([row["raw_ranker_box48"] == "1" for row in cases])) if cases else None,
        }

    numeric_context = [
        "top1_heatmap_score", "top2_to_top1_ratio", "correct_candidate_rank", "predicted_ttc",
        "motion_speed", "motion_stability", "motion_cumulative", "trajectory_padding_ratio",
        "trajectory_real_point_count", "trajectory_history_span_frames",
    ]
    context = {}
    for name, cases in (("train_oof", train_cases), ("val", val_cases)):
        context[name] = {
            field: scalar_summary([float(row[field]) for row in cases if row[field] != ""])
            for field in numeric_context
        }
    summary = {
        "diagnostic_policy": "Raw ranker scores are reported without a gate. These diagnostics use only OOF train and validation; no final holdout is loaded.",
        "pairwise_train_oof": pairwise_summary(train_cases),
        "pairwise_validation": pairwise_summary(val_cases),
        "raw_train_oof": raw_ranker_metrics(raw_train_rows),
        "raw_validation": raw_ranker_metrics(raw_val_rows),
        "rank_hard_candidate_context": context,
    }
    return summary, train_cases + val_cases, raw_train_rows + raw_val_rows


def draw_box(draw: ImageDraw.ImageDraw, x: float, y: float, color: str, width: int) -> None:
    draw.rectangle((x - 24, y - 24, x + 24, y + 24), outline=color, width=width)


def save_debug(rows: list[dict[str, str]], groups: list[dict], output_dir: Path, limit: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    by_name = {group["row"]["image_name"]: group for group in groups}
    ordered = sorted(rows, key=lambda row: float(row["ranked_error_px"]), reverse=True)[:limit]
    for result in ordered:
        row = by_name[result["image_name"]]["row"]
        image = Image.open(row["vision_path"]).convert("RGB")
        draw = ImageDraw.Draw(image)
        draw_box(draw, float(row["target_tip_x"]), float(row["target_tip_y"]), "lime", 4)
        draw_box(draw, float(result["top1_x"]), float(result["top1_y"]), "orange", 3)
        draw_box(draw, float(result["ranked_x"]), float(result["ranked_y"]), "cyan", 3)
        draw.text((8, 8), f"GT green | C2 orange | ranker cyan | rank {result['selected_original_rank']}", fill="white", stroke_width=2, stroke_fill="black")
        image.save(output_dir / f"{result['record_id']}_probe{int(result['probe']):03d}_{result['image_name']}")


def train(config_path: str, section: str) -> dict:
    config = load_config(config_path)
    cfg = config[section]
    set_seed(int(cfg.get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_groups = load_groups(config, cfg, cfg["train_oof_predictions_csv"], "train")
    val_groups = load_groups(config, cfg, cfg["validation_predictions_csv"], "val")
    feature_mode = str(cfg.get("feature_mode", "all"))
    select_feature_mode(train_groups, feature_mode)
    select_feature_mode(val_groups, feature_mode)
    train_features_raw = np.concatenate([group["features"] for group in train_groups], axis=0)
    mean = train_features_raw.mean(axis=0).astype(np.float32)
    std = train_features_raw.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    train_features = normalize(train_groups, mean, std).to(device)
    val_features = normalize(val_groups, mean, std)
    hard_indices, easy_indices = group_indices(train_groups)
    if len(hard_indices) < int(cfg.get("minimum_hard_queries", 50)):
        raise ValueError("Insufficient OOF rank-hard queries for pairwise ranker training")
    model = build_ranker(train_features.shape[-1], cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("learning_rate", 1e-3)), weight_decay=float(cfg.get("weight_decay", 1e-4)))
    checkpoint_dir = project_path(cfg["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    margin_candidates = [float(value) for value in cfg.get("rerank_margins", [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0])]
    best_key = None
    best_epoch = 0
    best_margin = float("inf")
    best_rows = []
    best_gate = None
    stale_epochs = 0
    history = []
    for epoch in range(1, int(cfg.get("epochs", 200)) + 1):
        model.train()
        scores = model(train_features)
        loss_hard = hard_loss(scores, train_groups, hard_indices, float(cfg.get("pairwise_margin", 0.25)), float(cfg.get("auxiliary_loss_weight", 0.25)))
        loss_easy = stability_loss(scores, easy_indices, float(cfg.get("stability_margin", 0.10)))
        loss = loss_hard + float(cfg.get("stability_loss_weight", 1.0)) * loss_easy
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        candidate_results = []
        for margin in margin_candidates:
            rows, gate = score_validation(model, val_features, val_groups, device, margin)
            if gate["passed"]:
                key = (gate["far_rank_hard_corrected"], gate["far_box48_after"], -gate["far_easy_corruption"], -gate["far_p90_error_px"], gate["gate_selected_queries"])
                candidate_results.append((key, margin, rows, gate))
        if candidate_results:
            key, margin, rows, gate = max(candidate_results, key=lambda item: item[0])
            if best_key is None or key > best_key:
                best_key, best_epoch, best_margin, best_rows, best_gate = key, epoch, margin, rows, gate
                stale_epochs = 0
                torch.save({"model": model.state_dict(), "feature_mean": mean, "feature_std": std, "feature_size": int(train_features.shape[-1]), "epoch": epoch, "rerank_margin": margin, "gate": gate, "config_section": section}, checkpoint_dir / "best.pt")
            else:
                stale_epochs += 1
        else:
            stale_epochs += 1
        history.append({"epoch": epoch, "loss": float(loss.item()), "hard_loss": float(loss_hard.item()), "stability_loss": float(loss_easy.item()), "best_margin": best_margin if math.isfinite(best_margin) else "inf", "gate_passed": best_gate is not None})
        if epoch == 1 or epoch % 10 == 0:
            print(f"epoch={epoch:03d} loss={loss.item():.5f} gate_passed={best_gate is not None} margin={best_margin if math.isfinite(best_margin) else 'inf'}")
        if best_gate is not None and stale_epochs >= int(cfg.get("patience", 40)):
            break
    if best_gate is None:
        baseline_rows, baseline_gate = score_validation(model, val_features, val_groups, device, float("inf"))
        best_rows, best_gate = baseline_rows, baseline_gate
    train_scores = model_scores(model, train_features, device)
    val_scores = model_scores(model, val_features, device)
    diagnostics, diagnostic_rows, raw_rows = build_diagnostics(train_groups, train_scores, val_groups, val_scores)
    torch.save({
        "model": model.state_dict(), "feature_mean": mean, "feature_std": std,
        "feature_size": int(train_features.shape[-1]), "epoch": len(history),
        "config_section": section, "note": "Final epoch checkpoint saved even when the validation gate rejects reranking.",
    }, checkpoint_dir / "last.pt")
    write_csv_rows(project_path(cfg["output_csv"]), best_rows, OUTPUT_FIELDS)
    write_csv_rows(project_path(cfg["raw_output_csv"]), raw_rows, OUTPUT_FIELDS)
    write_csv_rows(project_path(cfg["diagnostic_cases_csv"]), diagnostic_rows, DIAGNOSTIC_CASE_FIELDS)
    save_debug(best_rows, val_groups, project_path(cfg["debug_dir"]), int(cfg.get("debug_samples", 20)))
    summary = {
        "device": str(device), "policy": "OOF-only pairwise training; validation-only far gate; near/mid always preserve C2 Top1; no final holdout is loaded.",
        "train_queries": len(train_groups), "train_case_counts": dict(Counter(classify_group(group) for group in train_groups)),
        "validation_queries": len(val_groups), "validation_case_counts": dict(Counter(classify_group(group) for group in val_groups)),
        "ranker_architecture": str(cfg.get("ranker_architecture", "mlp")),
        "feature_mode": feature_mode, "feature_size": int(train_features.shape[-1]), "best_epoch": best_epoch,
        "rerank_margin": best_margin if math.isfinite(best_margin) else "inf", "validation_gate": best_gate,
        "gate_passed": bool(best_gate["passed"]), "diagnostics": diagnostics, "history": history,
    }
    write_json(project_path(cfg["metrics_json"]), summary)
    write_json(project_path(cfg["diagnostics_json"]), diagnostics)
    print(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Phase35 OOF pairwise Top-10 proposal ranker.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--section", default="proposal_ranker_pairwise_phase35")
    args = parser.parse_args()
    train(args.config, args.section)


if __name__ == "__main__":
    main()
