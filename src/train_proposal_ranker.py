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
from torch.utils.data import DataLoader, TensorDataset

from .build_cache_retrieval import visual_patch_feature
from .config import load_config, project_path
from .evaluate_proposal_recall import load_model_and_dataset
from .temporal_progress import DEFAULT_TTC_VALUES, masked_trajectory_features, motion_basis, read_trajectory_tracks
from .train_contact_region import collate_batch, forward_contact_model, parse_probe, topk_points, ttc_bucket_name
from .utils import read_csv_rows, write_csv_rows, write_json


OUTPUT_FIELDS = [
    "dataset_split", "record_id", "image_name", "probe", "ttc_bucket",
    "target_x", "target_y", "top1_x", "top1_y", "ranked_x", "ranked_y",
    "top1_error_px", "ranked_error_px", "top1_box48", "ranked_box48",
    "oracle_top10_box48", "selected_original_rank", "improved", "ranker_score",
    "candidate_points", "candidate_ranker_scores",
]


class ProposalRanker(nn.Module):
    def __init__(self, feature_size: int, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(feature_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.scorer(features).squeeze(-1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def percentile(values: list[float], q: float) -> float | None:
    return float(np.percentile(values, q)) if values else None


def summarize(rows: list[dict[str, str]], error_field: str, box_field: str) -> dict:
    errors = [float(row[error_field]) for row in rows]
    return {
        "samples": len(rows),
        "mean_error_px": float(np.mean(errors)) if errors else None,
        "median_error_px": float(np.median(errors)) if errors else None,
        "p75_error_px": percentile(errors, 75),
        "p90_error_px": percentile(errors, 90),
        "max_error_px": max(errors) if errors else None,
        "pck48": float(np.mean([error <= 48.0 for error in errors])) if errors else None,
        "failure_rate_gt48": float(np.mean([error > 48.0 for error in errors])) if errors else None,
        "box48_hit": float(np.mean([row[box_field] == "1" for row in rows])) if rows else None,
    }


def grouped_metrics(rows: list[dict[str, str]], error_field: str, box_field: str, field: str) -> dict:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row[field]].append(row)
    return {name: summarize(items, error_field, box_field) for name, items in sorted(groups.items())}


def recent_motion(trajectory: np.ndarray, mask: np.ndarray, fallback: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    valid = np.flatnonzero(mask > 0.5)
    if len(valid):
        compact = trajectory[valid]
        direction, speed = motion_basis(compact)
        stability = float(compact[-1, 14])
        cumulative = float(compact[-1, 12])
    else:
        direction, speed, stability, cumulative = fallback, 0.0, 0.0, 0.0
    if float(np.linalg.norm(direction)) <= 1e-6:
        direction = fallback
    norm = float(np.linalg.norm(direction))
    if norm > 1e-6:
        direction = direction / norm
    return direction.astype(np.float32), speed, stability, cumulative


def parse_points(value: str) -> list[tuple[float, float, float]]:
    points = []
    for item in value.split(";"):
        if not item:
            continue
        x, y, score = item.split(",")
        points.append((float(x), float(y), float(score)))
    return points


def soft_ttc_features(predicted_ttc: float, ttc_axis: np.ndarray) -> np.ndarray:
    weights = np.exp(-0.5 * ((ttc_axis - predicted_ttc) / 20.0) ** 2)
    return (weights / max(float(weights.sum()), 1e-6)).astype(np.float32)


def build_group(
    row: dict[str, str],
    points: list[tuple[float, float, float]],
    predicted_ttc: float,
    predicted_delta: np.ndarray,
    trajectory: np.ndarray,
    trajectory_mask: np.ndarray,
    topk: int,
    crop_size: int,
    include_heatmap_features: bool,
) -> dict:
    target_x = float(row["target_tip_x"])
    target_y = float(row["target_tip_y"])
    width = float(row["image_width"])
    height = float(row["image_height"])
    tip = np.asarray([float(row["tip_x"]), float(row["tip_y"])], dtype=np.float32)
    endpoint = tip + predicted_delta
    pose_direction = np.asarray([float(row["direction_x"]), float(row["direction_y"])], dtype=np.float32)
    pose_norm = float(np.linalg.norm(pose_direction))
    if pose_norm > 1e-6:
        pose_direction /= pose_norm
    direction, speed, stability, cumulative = recent_motion(trajectory, trajectory_mask, pose_direction)
    perpendicular = np.asarray([-direction[1], direction[0]], dtype=np.float32)
    pose_perpendicular = np.asarray([-pose_direction[1], pose_direction[0]], dtype=np.float32)
    top_score = max(float(points[0][2]), 1e-6)
    ttc_features = soft_ttc_features(predicted_ttc, np.asarray(DEFAULT_TTC_VALUES, dtype=np.float32))
    candidates = []
    for rank, (x, y, heatmap_score) in enumerate(points[:topk]):
        point = np.asarray([x, y], dtype=np.float32)
        tip_vector = point - tip
        endpoint_vector = point - endpoint
        values = [
            x / width, y / height,
            tip_vector[0] / 48.0, tip_vector[1] / 48.0, float(np.linalg.norm(tip_vector)) / 48.0,
            float(np.dot(tip_vector, direction)) / 48.0, float(np.dot(tip_vector, perpendicular)) / 48.0,
            endpoint_vector[0] / 48.0, endpoint_vector[1] / 48.0, float(np.linalg.norm(endpoint_vector)) / 48.0,
            float(np.dot(tip_vector, pose_direction)) / 48.0,
            float(np.dot(tip_vector, pose_perpendicular)) / 48.0,
            speed, stability, cumulative, predicted_ttc / 100.0,
            float(row.get("trajectory_padding_ratio", 0.0)),
        ]
        if include_heatmap_features:
            values = [
                heatmap_score, math.log(max(heatmap_score, 1e-8)), heatmap_score / top_score,
                float(rank) / max(topk - 1, 1),
            ] + values
        base_features = np.asarray(values, dtype=np.float32)
        local_visual = visual_patch_feature(row["vision_path"], x, y, crop_size).astype(np.float32)
        feature = np.concatenate([base_features, ttc_features, local_visual])
        error = float(np.linalg.norm(point - np.asarray([target_x, target_y], dtype=np.float32)))
        candidates.append({
            "x": float(x), "y": float(y), "heatmap_score": float(heatmap_score), "error": error,
            "box48": abs(x - target_x) <= 24.0 and abs(y - target_y) <= 24.0, "feature": feature,
        })
    return {
        "row": row, "target_x": target_x, "target_y": target_y, "candidates": candidates,
        "features": np.stack([candidate["feature"] for candidate in candidates]),
        "target_index": int(np.argmin([candidate["error"] for candidate in candidates])),
    }


def extract_groups(config: dict, cfg: dict, device: torch.device) -> list[dict]:
    model_section = str(cfg["contact_model_section"])
    model_cfg = config[model_section]
    model, dataset = load_model_and_dataset(config, model_section, device, {"train", "val", "test"})
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()
    loader = DataLoader(
        dataset, batch_size=int(cfg.get("extraction_batch_size", 16)), shuffle=False,
        num_workers=0, collate_fn=collate_batch,
    )
    k = int(cfg.get("topk", 10))
    suppression_radius = int(cfg.get("suppression_radius", 6))
    displacement_scale = float(model_cfg.get("displacement_scale_px", 48.0))
    ttc_axis = np.asarray(model_cfg.get("ttc_values", DEFAULT_TTC_VALUES), dtype=np.float32)
    groups = []

    with torch.no_grad():
        for batch in loader:
            output = forward_contact_model(model, batch, device)
            for index, row in enumerate(batch["rows"]):
                target_x, target_y, width, height = batch["coords"][index].cpu().numpy()
                raw_points = topk_points(output["heatmap"][index, 0].cpu(), k, suppression_radius)
                probabilities = output["ttc_probabilities"][index].cpu().numpy()
                predicted_ttc = float(np.sum(probabilities * ttc_axis))
                predicted_delta = output["displacement"][index].cpu().numpy() * displacement_scale
                trajectory = batch["trajectory"][index].cpu().numpy()
                trajectory_mask = batch["trajectory_mask"][index].cpu().numpy()
                scaled_points = [
                    (x / int(model_cfg["input_width"]) * float(width), y / int(model_cfg["input_height"]) * float(height), score)
                    for x, y, score in raw_points
                ]
                groups.append(build_group(
                    row, scaled_points, predicted_ttc, predicted_delta, trajectory, trajectory_mask, k,
                    int(cfg.get("cache_crop_size", 48)), bool(cfg.get("include_heatmap_features", True)),
                ))
    return groups


def extract_oof_groups(config: dict, cfg: dict) -> list[dict]:
    model_cfg = config[str(cfg["contact_model_section"])]
    source_rows = read_csv_rows(project_path(model_cfg["samples_csv"]))
    rows_by_name = {row["image_name"]: row for row in source_rows}
    tracks = read_trajectory_tracks(project_path(model_cfg["motion_tracks_csv"]))
    history = int(model_cfg.get("trajectory_history_frames", 16))
    spatial_scale = float(model_cfg.get("trajectory_spatial_scale_px", 48.0))
    speed_scale = float(model_cfg.get("trajectory_speed_scale_px", 4.0))
    groups = []
    for prediction in read_csv_rows(project_path(cfg["oof_predictions_csv"])):
        row = rows_by_name[prediction["image_name"]]
        trajectory, mask, _ = masked_trajectory_features(row, tracks, history, spatial_scale, speed_scale)
        predicted_delta = np.asarray([
            float(prediction.get("predicted_delta_x") or 0.0),
            float(prediction.get("predicted_delta_y") or 0.0),
        ], dtype=np.float32)
        groups.append(build_group(
            row, parse_points(prediction["topk_points"]), float(prediction.get("predicted_ttc") or 0.0),
            predicted_delta, trajectory, mask, int(cfg.get("topk", 10)), int(cfg.get("cache_crop_size", 48)),
            bool(cfg.get("include_heatmap_features", True)),
        ))
    return groups


def normalized_arrays(groups: list[dict], mean: np.ndarray, std: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    features = np.stack([group["features"] for group in groups])
    features = (features - mean[None, None]) / std[None, None]
    targets = np.asarray([
        [1.0 if candidate["box48"] else 0.0 for candidate in group["candidates"]]
        for group in groups
    ], dtype=np.float32)
    weights = np.asarray([
        2.0 if int(parse_probe(group["row"]) or 0) >= 75 else 1.25 if int(parse_probe(group["row"]) or 0) >= 30 else 1.0
        for group in groups
    ], dtype=np.float32)
    return torch.from_numpy(features.astype(np.float32)), torch.from_numpy(targets), torch.from_numpy(weights)


def score_groups(
    model: ProposalRanker,
    groups: list[dict],
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    rerank_margin: float = 0.0,
) -> list[dict[str, str]]:
    if not groups:
        return []
    features, _, _ = normalized_arrays(groups, mean, std)
    model.eval()
    rows = []
    with torch.no_grad():
        scores = model(features.to(device)).cpu().numpy()
    buckets = {"near": {5, 10, 20}, "mid": {30, 50}, "far": {75, 100}}
    for group, group_scores in zip(groups, scores):
        source = group["row"]
        candidates = group["candidates"]
        best_index = int(np.argmax(group_scores))
        advantage = float(group_scores[best_index] - group_scores[0])
        selected_index = best_index if int(parse_probe(group["row"]) or 0) >= 75 and advantage >= rerank_margin else 0
        original = candidates[0]
        selected = candidates[selected_index]
        rows.append({
            "dataset_split": source["dataset_split"], "record_id": source["record_id"],
            "image_name": source["image_name"], "probe": str(parse_probe(source) or ""),
            "ttc_bucket": ttc_bucket_name(parse_probe(source), buckets),
            "target_x": f"{group['target_x']:.3f}", "target_y": f"{group['target_y']:.3f}",
            "top1_x": f"{original['x']:.3f}", "top1_y": f"{original['y']:.3f}",
            "ranked_x": f"{selected['x']:.3f}", "ranked_y": f"{selected['y']:.3f}",
            "top1_error_px": f"{original['error']:.3f}", "ranked_error_px": f"{selected['error']:.3f}",
            "top1_box48": "1" if original["box48"] else "0",
            "ranked_box48": "1" if selected["box48"] else "0",
            "oracle_top10_box48": "1" if any(candidate["box48"] for candidate in candidates) else "0",
            "selected_original_rank": str(selected_index + 1),
            "improved": "1" if selected["error"] < original["error"] else "0",
            "ranker_score": f"{float(group_scores[selected_index]):.6f}",
            "candidate_points": ";".join(
                f"{candidate['x']:.3f},{candidate['y']:.3f},{candidate['heatmap_score']:.6f}"
                for candidate in candidates
            ),
            "candidate_ranker_scores": ";".join(f"{float(value):.6f}" for value in group_scores),
        })
    return rows


def selection_key(rows: list[dict[str, str]]) -> tuple[float, float, float, float]:
    far_rows = [row for row in rows if row["ttc_bucket"] == "far"]
    metrics = summarize(far_rows, "ranked_error_px", "ranked_box48")
    return (
        float(metrics["pck48"] or 0.0),
        float(metrics["box48_hit"] or 0.0),
        -float(metrics["p90_error_px"] or 1e9),
        float(summarize(rows, "ranked_error_px", "ranked_box48")["box48_hit"] or 0.0),
    )


def draw_box(draw: ImageDraw.ImageDraw, x: float, y: float, color: str, width: int) -> None:
    draw.rectangle((x - 24, y - 24, x + 24, y + 24), outline=color, width=width)


def save_debug(rows: list[dict[str, str]], rows_by_name: dict[str, dict[str, str]], output_dir: Path, limit: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=lambda row: float(row["ranked_error_px"]), reverse=True)[:limit]
    for row in ordered:
        source = rows_by_name[row["image_name"]]
        image = Image.open(source["vision_path"]).convert("RGB")
        draw = ImageDraw.Draw(image)
        draw_box(draw, float(row["target_x"]), float(row["target_y"]), "lime", 4)
        draw_box(draw, float(row["top1_x"]), float(row["top1_y"]), "orange", 3)
        draw_box(draw, float(row["ranked_x"]), float(row["ranked_y"]), "cyan", 3)
        draw.text((8, 8), f"GT green | C2 orange | ranker cyan | rank {row['selected_original_rank']}", fill="white", stroke_width=2, stroke_fill="black")
        name = f"{row['record_id']}_probe{int(row['probe']):03d}_{row['image_name']}"
        image.save(output_dir / name)


def train(config_path: str, section: str) -> dict:
    config = load_config(config_path)
    cfg = config[section]
    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    groups = extract_groups(config, cfg, device)
    by_split = {split: [group for group in groups if group["row"]["dataset_split"] == split] for split in ("train", "val", "test")}
    if cfg.get("oof_predictions_csv"):
        by_split["train"] = extract_oof_groups(config, cfg)
    by_split["train"] = [group for group in by_split["train"] if int(parse_probe(group["row"]) or 0) >= 75]
    train_matrix = np.concatenate([group["features"] for group in by_split["train"]], axis=0)
    mean = train_matrix.mean(axis=0).astype(np.float32)
    std = train_matrix.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    train_features, train_targets, train_weights = normalized_arrays(by_split["train"], mean, std)
    loader = DataLoader(
        TensorDataset(train_features, train_targets, train_weights),
        batch_size=int(cfg.get("batch_size", 64)), shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    model = ProposalRanker(train_features.shape[-1], int(cfg.get("hidden_size", 64)), float(cfg.get("dropout", 0.1))).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("learning_rate", 1e-3)), weight_decay=float(cfg.get("weight_decay", 1e-4)))
    checkpoint_dir = project_path(cfg["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_key = (-1.0, -1.0, -1e9, -1.0)
    best_epoch = 0
    best_margin = float("inf")
    epochs_without_improvement = 0
    history = []
    for epoch in range(1, int(cfg.get("epochs", 200)) + 1):
        model.train()
        losses = []
        for features, targets, weights in loader:
            logits = model(features.to(device))
            target_values = targets.to(device)
            positive_count = torch.sum(target_values)
            negative_count = target_values.numel() - positive_count
            positive_weight = torch.clamp(negative_count / torch.clamp(positive_count, min=1.0), max=8.0)
            element_loss = nn.functional.binary_cross_entropy_with_logits(
                logits, target_values, reduction="none", pos_weight=positive_weight,
            )
            sample_loss = torch.mean(element_loss, dim=1)
            loss = torch.mean(sample_loss * weights.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        margin_candidates = [float(value) for value in cfg.get("rerank_margins", [0.0, 0.1, 0.25, 0.5, 1.0, 2.0])]
        margin_candidates.append(float("inf"))
        candidate_results = [
            (selection_key(rows), margin, rows)
            for margin in margin_candidates
            for rows in [score_groups(model, by_split["val"], mean, std, device, margin)]
        ]
        key, selected_margin, val_rows = max(candidate_results, key=lambda item: item[0])
        val_metrics = summarize(val_rows, "ranked_error_px", "ranked_box48")
        history.append({
            "epoch": epoch, "train_loss": float(np.mean(losses)),
            "rerank_margin": selected_margin if math.isfinite(selected_margin) else "inf",
            "validation": val_metrics,
        })
        if key > best_key:
            best_key = key
            best_epoch = epoch
            best_margin = selected_margin
            epochs_without_improvement = 0
            torch.save({
                "model": model.state_dict(), "feature_mean": mean, "feature_std": std,
                "feature_size": int(train_features.shape[-1]), "epoch": epoch,
                "selection_key": key, "rerank_margin": selected_margin,
                "config_section": section,
            }, checkpoint_dir / "best.pt")
        else:
            epochs_without_improvement += 1
        if epoch == 1 or epoch % 10 == 0:
            print(f"epoch={epoch} loss={np.mean(losses):.5f} val_box48={val_metrics['box48_hit']:.4f} val_p90={val_metrics['p90_error_px']:.3f}")
        if epochs_without_improvement >= int(cfg.get("patience", 40)):
            break

    checkpoint = torch.load(checkpoint_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    best_margin = float(checkpoint.get("rerank_margin", float("inf")))
    output_rows = []
    split_metrics = {}
    for split, split_groups in by_split.items():
        rows = score_groups(model, split_groups, mean, std, device, best_margin)
        output_rows.extend(rows)
        split_metrics[split] = {
            "before": summarize(rows, "top1_error_px", "top1_box48"),
            "after": summarize(rows, "ranked_error_px", "ranked_box48"),
            "by_ttc_bucket": grouped_metrics(rows, "ranked_error_px", "ranked_box48", "ttc_bucket"),
            "by_probe": grouped_metrics(rows, "ranked_error_px", "ranked_box48", "probe"),
            "oracle_top10_box48": float(np.mean([row["oracle_top10_box48"] == "1" for row in rows])) if rows else None,
            "wins": sum(row["improved"] == "1" for row in rows),
            "losses": sum(float(row["ranked_error_px"]) > float(row["top1_error_px"]) for row in rows),
            "ties": sum(float(row["ranked_error_px"]) == float(row["top1_error_px"]) for row in rows),
            "selected_rank_counts": dict(Counter(row["selected_original_rank"] for row in rows)),
        }
    write_csv_rows(project_path(cfg["output_csv"]), output_rows, OUTPUT_FIELDS)
    summary = {
        "device": str(device), "seed": seed, "contact_model_section": cfg["contact_model_section"],
        "contact_model_frozen": True, "topk": int(cfg.get("topk", 10)),
        "feature_size": int(train_features.shape[-1]), "best_epoch": best_epoch,
        "rerank_margin": best_margin if math.isfinite(best_margin) else "inf",
        "selection_policy": "Far-only validation lexicographic: Box48 hit, PCK@48, then lower P90. Near/mid always preserve C2 Top1. Test is never used for selection.",
        "splits": split_metrics, "history": history,
    }
    write_json(project_path(cfg["metrics_json"]), summary)
    rows_by_name = {group["row"]["image_name"]: group["row"] for group in groups}
    save_debug([row for row in output_rows if row["dataset_split"] == "test"], rows_by_name, project_path(cfg["debug_dir"]), int(cfg.get("debug_samples", 20)))
    print(summary["splits"]["test"])
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze C2 and train a listwise Top-10 contact proposal ranker.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--section", default="proposal_ranker_masked_16")
    args = parser.parse_args()
    train(args.config, args.section)


if __name__ == "__main__":
    main()
