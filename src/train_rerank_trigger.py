from __future__ import annotations

import argparse
import math
import random
from collections import Counter

import numpy as np
import torch
from torch import nn

from .config import load_config, project_path
from .temporal_progress import masked_trajectory_features, read_trajectory_tracks
from .train_proposal_ranker import recent_motion
from .utils import ensure_dir, read_csv_rows, write_csv_rows, write_json


OUTPUT_FIELDS = [
    "dataset_split", "split", "record_id", "image_name", "probe", "case_type", "label_rank_hard",
    "risk_score", "gate_selected",
]
GRID_FIELDS = [
    "quantile", "threshold", "selected_queries", "selected_rank_hard", "selected_easy", "precision", "recall", "eligible",
]


class RerankRiskTrigger(nn.Module):
    def __init__(self, feature_size: int, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_points(value: str) -> list[tuple[float, float, float]]:
    points = []
    for item in value.split(";"):
        if item:
            x, y, score = item.split(",")
            points.append((float(x), float(y), float(score)))
    return points


def classify(row: dict[str, str]) -> str:
    target_x, target_y = float(row["target_x"]), float(row["target_y"])
    hits = [abs(x - target_x) <= 24.0 and abs(y - target_y) <= 24.0 for x, y, _ in parse_points(row["topk_points"])]
    if not hits:
        raise ValueError(f"Missing Top-10 candidates for {row['image_name']}")
    return "easy" if hits[0] else "rank_hard" if any(hits) else "proposal_miss"


def entropy(values: np.ndarray) -> float:
    total = float(values.sum())
    if total <= 1e-8:
        return 0.0
    probability = values / total
    return float(-(probability * np.log(np.maximum(probability, 1e-8))).sum() / math.log(len(values)))


def feature_row(
    prediction: dict[str, str],
    sample: dict[str, str],
    tracks: dict,
    history_frames: int,
    spatial_scale: float,
    speed_scale: float,
) -> np.ndarray:
    points = parse_points(prediction["topk_points"])
    if len(points) != 10:
        raise ValueError(f"Expected 10 candidates for {prediction['image_name']}, got {len(points)}")
    scores = np.asarray([point[2] for point in points], dtype=np.float32)
    top_score = max(float(scores[0]), 1e-8)
    tip = np.asarray([float(sample["tip_x"]), float(sample["tip_y"])], dtype=np.float32)
    pose_direction = np.asarray([float(sample["direction_x"]), float(sample["direction_y"])], dtype=np.float32)
    pose_norm = float(np.linalg.norm(pose_direction))
    if pose_norm > 1e-6:
        pose_direction /= pose_norm
    trajectory, mask, _ = masked_trajectory_features(sample, tracks, history_frames, spatial_scale, speed_scale)
    direction, speed, stability, cumulative = recent_motion(trajectory, mask, pose_direction)
    perpendicular = np.asarray([-direction[1], direction[0]], dtype=np.float32)
    top_vector = np.asarray(points[0][:2], dtype=np.float32) - tip
    second_vector = np.asarray(points[1][:2], dtype=np.float32) - tip
    top_to_second = np.asarray(points[1][:2], dtype=np.float32) - np.asarray(points[0][:2], dtype=np.float32)
    endpoint = tip + np.asarray([
        float(prediction.get("predicted_delta_x") or 0.0),
        float(prediction.get("predicted_delta_y") or 0.0),
    ], dtype=np.float32)
    endpoint_vector = np.asarray(points[0][:2], dtype=np.float32) - endpoint
    history_count = float(sample.get("trajectory_real_point_count") or 0.0)
    history_span = float(sample.get("trajectory_history_span_frames") or 0.0)
    padding_ratio = float(sample.get("trajectory_padding_ratio") or 0.0)
    return np.asarray([
        float(prediction["probe"]) / 100.0,
        float(prediction.get("predicted_ttc") or 0.0) / 100.0,
        top_score,
        math.log(top_score),
        float(scores[1] / top_score),
        float(scores[2] / top_score),
        entropy(scores),
        float(np.dot(top_vector, direction)) / spatial_scale,
        float(np.dot(top_vector, perpendicular)) / spatial_scale,
        float(np.linalg.norm(top_to_second)) / spatial_scale,
        float(np.dot(top_to_second, direction)) / spatial_scale,
        float(np.dot(second_vector, perpendicular)) / spatial_scale,
        float(np.linalg.norm(endpoint_vector)) / spatial_scale,
        speed,
        stability,
        cumulative,
        history_count / max(history_frames, 1),
        history_span / max(history_frames - 1, 1),
        padding_ratio,
        float(sample["tip_base_distance"]) / spatial_scale,
    ], dtype=np.float32)


def load_examples(config: dict, cfg: dict, prediction_csv: str, expected_split: str) -> tuple[list[dict[str, str]], np.ndarray, np.ndarray]:
    model_cfg = config[str(cfg["contact_model_section"])]
    samples = {row["image_name"]: row for row in read_csv_rows(project_path(model_cfg["samples_csv"]))}
    tracks = read_trajectory_tracks(project_path(model_cfg["motion_tracks_csv"]))
    history = int(model_cfg["trajectory_history_frames"])
    spatial_scale = float(model_cfg["trajectory_spatial_scale_px"])
    speed_scale = float(model_cfg["trajectory_speed_scale_px"])
    far_only = bool(cfg.get("far_only", True))
    rows, features, labels = [], [], []
    for prediction in read_csv_rows(project_path(prediction_csv)):
        if prediction["dataset_split"] != expected_split:
            continue
        if far_only and int(prediction["probe"]) < 75:
            continue
        sample = samples.get(prediction["image_name"])
        if sample is None or sample["dataset_split"] != expected_split:
            raise RuntimeError(f"Prediction/sample split mismatch for {prediction['image_name']}")
        record_number = int(prediction["record_id"].rsplit("_", 1)[1])
        if prediction["split"] == "0" and 950 <= record_number <= 999:
            raise RuntimeError("Final holdout entered trigger data")
        case_type = classify(prediction)
        if case_type == "proposal_miss":
            continue
        rows.append({**prediction, "case_type": case_type})
        features.append(feature_row(prediction, sample, tracks, history, spatial_scale, speed_scale))
        labels.append(1.0 if case_type == "rank_hard" else 0.0)
    if not rows:
        raise ValueError(f"No usable {expected_split} trigger examples")
    return rows, np.stack(features), np.asarray(labels, dtype=np.float32)


def score(model: nn.Module, features: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(features).to(device))
    return torch.sigmoid(logits).cpu().numpy()


def grid_rows(scores: np.ndarray, labels: np.ndarray, quantiles: list[float], minimum_precision: float, minimum_recall: float, minimum_queries: int) -> tuple[list[dict[str, str]], float | None]:
    rows = []
    best_key, selected_threshold = None, None
    for quantile in quantiles:
        threshold = float(np.quantile(scores, quantile))
        selected = scores >= threshold
        selected_count = int(selected.sum())
        selected_hard = int(labels[selected].sum())
        selected_easy = selected_count - selected_hard
        precision = selected_hard / selected_count if selected_count else 0.0
        recall = selected_hard / max(int(labels.sum()), 1)
        eligible = selected_count >= minimum_queries and precision >= minimum_precision and recall >= minimum_recall
        rows.append({
            "quantile": f"{quantile:.4f}", "threshold": f"{threshold:.6f}", "selected_queries": str(selected_count),
            "selected_rank_hard": str(selected_hard), "selected_easy": str(selected_easy),
            "precision": f"{precision:.6f}", "recall": f"{recall:.6f}", "eligible": "1" if eligible else "0",
        })
        if eligible:
            key = (precision, recall, -selected_easy, selected_hard)
            if best_key is None or key > best_key:
                best_key, selected_threshold = key, threshold
    return rows, selected_threshold


def train(config_path: str, section: str) -> dict:
    config = load_config(config_path)
    cfg = config[section]
    set_seed(int(cfg.get("seed", 20260730)))
    train_rows, train_features, train_labels = load_examples(config, cfg, cfg["train_oof_predictions_csv"], "train")
    validation_rows, validation_features, validation_labels = load_examples(config, cfg, cfg["validation_predictions_csv"], "val")
    mean = train_features.mean(axis=0)
    std = train_features.std(axis=0)
    std[std < 1e-6] = 1.0
    train_features = ((train_features - mean) / std).astype(np.float32)
    validation_features = ((validation_features - mean) / std).astype(np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RerankRiskTrigger(train_features.shape[1], int(cfg.get("hidden_size", 32)), float(cfg.get("dropout", 0.1))).to(device)
    positive_weight = float(cfg.get("positive_weight", max((len(train_labels) - train_labels.sum()) / max(train_labels.sum(), 1.0), 1.0)))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_weight, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("learning_rate", 0.001)), weight_decay=float(cfg.get("weight_decay", 1e-4)))
    checkpoint_dir = ensure_dir(project_path(cfg["checkpoint_dir"]))
    quantiles = [float(value) for value in cfg.get("gate_quantiles", [0.5, 0.75, 0.85, 0.9, 0.95, 0.97, 0.99])]
    min_precision = float(cfg.get("minimum_precision", 0.5))
    min_recall = float(cfg.get("minimum_recall", 0.2))
    min_queries = int(cfg.get("minimum_gate_queries", 4))
    best_key, best_epoch, best_threshold, stale = None, 0, None, 0
    history = []
    x_train = torch.from_numpy(train_features).to(device)
    y_train = torch.from_numpy(train_labels).to(device)
    for epoch in range(1, int(cfg.get("epochs", 160)) + 1):
        model.train()
        logits = model(x_train)
        loss = loss_fn(logits, y_train)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        validation_scores = score(model, validation_features, device)
        grid, threshold = grid_rows(validation_scores, validation_labels, quantiles, min_precision, min_recall, min_queries)
        eligible = [row for row in grid if row["eligible"] == "1"]
        if eligible:
            winner = max(eligible, key=lambda row: (float(row["precision"]), float(row["recall"]), -int(row["selected_easy"])))
            key = (float(winner["precision"]), float(winner["recall"]), -int(winner["selected_easy"]), int(winner["selected_rank_hard"]))
            if best_key is None or key > best_key:
                best_key, best_epoch, best_threshold, stale = key, epoch, threshold, 0
                torch.save({"model": model.state_dict(), "feature_mean": mean, "feature_std": std, "threshold": threshold, "epoch": epoch, "config_section": section}, checkpoint_dir / "best.pt")
            else:
                stale += 1
        else:
            stale += 1
        history.append({"epoch": epoch, "loss": float(loss.item()), "gate_found": bool(eligible)})
        if epoch == 1 or epoch % 20 == 0:
            print(f"epoch={epoch:03d} loss={loss.item():.5f} gate_found={bool(eligible)}")
        if best_key is not None and stale >= int(cfg.get("patience", 30)):
            break
    if best_threshold is not None:
        best_checkpoint = torch.load(checkpoint_dir / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(best_checkpoint["model"])
    validation_scores = score(model, validation_features, device)
    grid, _ = grid_rows(validation_scores, validation_labels, quantiles, min_precision, min_recall, min_queries)
    selected = validation_scores >= best_threshold if best_threshold is not None else np.zeros(len(validation_scores), dtype=bool)
    output_rows = [{
        "dataset_split": "val", "split": row["split"], "record_id": row["record_id"], "image_name": row["image_name"], "probe": row["probe"],
        "case_type": row["case_type"], "label_rank_hard": str(int(label)), "risk_score": f"{float(value):.6f}", "gate_selected": str(int(flag)),
    } for row, label, value, flag in zip(validation_rows, validation_labels, validation_scores, selected)]
    torch.save({"model": model.state_dict(), "feature_mean": mean, "feature_std": std, "epoch": len(history), "config_section": section}, checkpoint_dir / "last.pt")
    write_csv_rows(project_path(cfg["output_csv"]), output_rows, OUTPUT_FIELDS)
    write_csv_rows(project_path(cfg["grid_csv"]), grid, GRID_FIELDS)
    summary = {
        "device": str(device),
        "policy": "The trigger uses only online-safe C2/trajectory signals. It is trained from OOF train examples and selects a threshold only on development validation; proposal misses and final holdout are excluded.",
        "feature_size": int(train_features.shape[1]), "train_examples": len(train_rows), "validation_examples": len(validation_rows),
        "train_case_counts": dict(Counter(row["case_type"] for row in train_rows)), "validation_case_counts": dict(Counter(row["case_type"] for row in validation_rows)),
        "best_epoch": best_epoch, "selected_threshold": best_threshold, "gate_passed": best_threshold is not None,
        "minimum_precision": min_precision, "minimum_recall": min_recall, "minimum_gate_queries": min_queries, "history": history,
    }
    write_json(project_path(cfg["metrics_json"]), summary)
    print(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a high-precision query trigger before Top-K reranking.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--section", default="rerank_trigger_phase35_v4")
    args = parser.parse_args()
    train(args.config, args.section)


if __name__ == "__main__":
    main()
