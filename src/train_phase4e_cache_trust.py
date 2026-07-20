from __future__ import annotations

import argparse
import math
from collections import defaultdict

import numpy as np
import torch
from torch import nn

from .config import load_config, project_path
from .train_phase4b_predicted_box_cache_ranker import is_final_holdout, set_seed
from .utils import ensure_dir, read_csv_rows, write_csv_rows, write_json


TARGET_FIELDS = ("tactile_diff_mae", "tactile_ssim", "tactile_mask_iou")
OUTPUT_FIELDS = [
    "query_record_id", "query_image_name", "query_probe", "oof_fold", "status", "trust_score", "gate_threshold",
    "predicted_tactile_diff_mae", "predicted_tactile_ssim", "predicted_tactile_mask_iou", "selected_cache_record_id",
    "top3_cache_record_ids", "rejection_reasons", "ranker_best_score", "ranker_margin_normalized",
    "top3_tactile_embedding_disagreement", "c2_pred_score", "ranker_oracle_embedding_rank", *TARGET_FIELDS,
]


class CacheTrustPredictor(nn.Module):
    """Predicts retrieval quality using only signals present at cache-query time."""

    def __init__(self, feature_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_dim, 96), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(96, 48), nn.ReLU(inplace=True), nn.Linear(48, 3),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def finite(value: str, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def candidate_features(candidate_rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    """Aggregate online-only score and distance statistics from the fixed 32-item shortlist."""
    by_query: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in candidate_rows:
        by_query[row["query_image_name"]].append(row)
    output = {}
    for image_name, rows in by_query.items():
        ordered = sorted(rows, key=lambda row: int(row["candidate_rank"]))
        if len(ordered) < 3:
            raise RuntimeError(f"Expected at least three candidates for {image_name}, got {len(ordered)}")
        scores = np.asarray([finite(row["candidate_score"]) for row in ordered], dtype=np.float32)
        hand = np.asarray([finite(row["hand_score"]) for row in ordered], dtype=np.float32)
        geometry = np.asarray([finite(row["geometry_distance"]) for row in ordered], dtype=np.float32)
        detail = np.asarray([finite(row["detail_visual_distance"]) for row in ordered], dtype=np.float32)
        context = np.asarray([finite(row["context_visual_distance"]) for row in ordered], dtype=np.float32)
        hand_best_rank = int(np.argmin(hand))
        output[image_name] = {
            "candidate_score_top5_mean": float(scores[:5].mean()),
            "candidate_score_top5_std": float(scores[:5].std()),
            "candidate_score_all_std": float(scores.std()),
            "candidate_geometry_top3_mean": float(geometry[:3].mean()),
            "candidate_detail_top3_mean": float(detail[:3].mean()),
            "candidate_context_top3_mean": float(context[:3].mean()),
            "candidate_ranker_hand_disagreement": float(hand_best_rank != 0),
            "candidate_ranker_hand_rank_gap": float(hand_best_rank),
        }
    return output


def feature_names() -> list[str]:
    return [
        "query_probe", "c2_pred_score", "ranker_best_score", "ranker_margin", "ranker_margin_normalized",
        "hand_best_score", "hand_margin", "top3_tactile_embedding_disagreement", "top3_score_std",
        "geometry_distance", "detail_visual_distance", "context_visual_distance", "trajectory_real_point_count",
        "trajectory_history_span_frames", "trajectory_padding_ratio", "trajectory_cumulative_displacement",
        "candidate_score_top5_mean", "candidate_score_top5_std", "candidate_score_all_std", "candidate_geometry_top3_mean",
        "candidate_detail_top3_mean", "candidate_context_top3_mean", "candidate_ranker_hand_disagreement",
        "candidate_ranker_hand_rank_gap",
    ]


def make_matrix(rows: list[dict[str, str]], candidate_by_name: dict[str, dict[str, float]]) -> np.ndarray:
    names = feature_names()
    matrix = []
    for row in rows:
        aggregate = candidate_by_name.get(row["query_image_name"])
        if aggregate is None:
            raise RuntimeError(f"Candidate aggregate missing for {row['query_image_name']}")
        values = []
        for name in names:
            if name in aggregate:
                values.append(aggregate[name])
            else:
                values.append(finite(row.get(name, "")))
        matrix.append(values)
    return np.asarray(matrix, dtype=np.float32)


def make_targets(rows: list[dict[str, str]]) -> np.ndarray:
    return np.asarray([[finite(row[field]) for field in TARGET_FIELDS] for row in rows], dtype=np.float32)


def standardize(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (values - mean) / np.maximum(std, 1e-6)


def trust_scores(predicted_targets: np.ndarray, target_mean: np.ndarray, target_std: np.ndarray) -> np.ndarray:
    """A high score requires all three predicted quality dimensions to be good, not only their average."""
    quality = np.column_stack(
        [
            (target_mean[0] - predicted_targets[:, 0]) / max(float(target_std[0]), 1e-6),
            (predicted_targets[:, 1] - target_mean[1]) / max(float(target_std[1]), 1e-6),
            (predicted_targets[:, 2] - target_mean[2]) / max(float(target_std[2]), 1e-6),
        ]
    )
    return quality.min(axis=1)


def choose_gate(rows: list[dict[str, str]], scores: np.ndarray, min_coverage: float, max_coverage: float, target_coverage: float) -> dict:
    if not 0 < min_coverage <= target_coverage <= max_coverage <= 1:
        raise ValueError("Coverage configuration must satisfy 0 < min <= target <= max <= 1.")
    metrics = make_targets(rows)
    baseline = metrics.mean(axis=0)
    options = []
    for coverage in np.linspace(min_coverage, max_coverage, 81):
        count = max(1, int(round(len(rows) * coverage)))
        threshold = float(np.sort(scores)[-count])
        accepted = scores >= threshold
        selected = metrics[accepted].mean(axis=0)
        passes = bool(selected[0] < baseline[0] and selected[1] > baseline[1] and selected[2] >= baseline[2])
        utility = (baseline[0] - selected[0]) / max(float(baseline[0]), 1e-6) + (selected[1] - baseline[1]) + (selected[2] - baseline[2])
        options.append({
            "coverage": float(accepted.mean()), "threshold": threshold, "accepted_queries": int(accepted.sum()),
            "actual": {field: float(value) for field, value in zip(TARGET_FIELDS, selected, strict=True)},
            "passes_multimetric_guard": passes, "utility": float(utility),
        })
    valid = [option for option in options if option["passes_multimetric_guard"]]
    if not valid:
        return {"enabled": False, "baseline": {field: float(value) for field, value in zip(TARGET_FIELDS, baseline, strict=True)}, "options": options}
    chosen = min(valid, key=lambda option: (abs(option["coverage"] - target_coverage), -option["utility"]))
    return {
        "enabled": True, "baseline": {field: float(value) for field, value in zip(TARGET_FIELDS, baseline, strict=True)},
        "selected": chosen, "options": options,
    }


def rejection_reasons(prediction: np.ndarray, target_mean: np.ndarray, target_std: np.ndarray, score: float, threshold: float | None) -> str:
    if threshold is not None and score >= threshold:
        return ""
    reasons = ["low_trust_score"]
    if prediction[0] > target_mean[0]:
        reasons.append("predicted_high_mae")
    if prediction[1] < target_mean[1]:
        reasons.append("predicted_low_ssim")
    if prediction[2] < target_mean[2]:
        reasons.append("predicted_low_iou")
    return "|".join(reasons)


def metric_summary(rows: list[dict[str, str]], accepted: np.ndarray, baseline: np.ndarray) -> dict:
    targets = make_targets(rows)
    output = {"queries": len(rows), "cache_hit_coverage": float(accepted.mean()), "cache_miss_rate": float(1.0 - accepted.mean())}
    for label, mask in (("all", np.ones(len(rows), dtype=bool)), ("accepted", accepted), ("miss", ~accepted)):
        values = targets[mask]
        ranks = np.asarray([int(rows[index]["ranker_oracle_embedding_rank"]) for index in np.flatnonzero(mask)], dtype=np.int32)
        output[label] = {
            "queries": int(mask.sum()),
            **{field: float(values[:, index].mean()) if len(values) else None for index, field in enumerate(TARGET_FIELDS)},
            "tactile_best_top1_rate": float(np.mean(ranks == 1)) if len(ranks) else None,
            "tactile_best_top3_rate": float(np.mean(ranks <= 3)) if len(ranks) else None,
        }
    actual_good = (targets[:, 0] < baseline[0]) & (targets[:, 1] > baseline[1]) & (targets[:, 2] >= baseline[2])
    false_accept = accepted & ~actual_good
    false_reject = ~accepted & actual_good
    output["quality_confusion"] = {
        "quality_definition": "actual MAE below all-query baseline, SSIM above it, and IoU at least equal to it",
        "actual_good_queries": int(actual_good.sum()), "true_accepts": int((accepted & actual_good).sum()),
        "false_accepts": int(false_accept.sum()), "false_rejects": int(false_reject.sum()),
        "false_accept_rate_among_hits": float(false_accept.sum() / max(int(accepted.sum()), 1)),
        "false_reject_rate_among_actual_good": float(false_reject.sum() / max(int(actual_good.sum()), 1)),
    }
    return output


def train(config_path: str, section: str) -> dict:
    cfg = load_config(config_path)[section]
    set_seed(int(cfg["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples = read_csv_rows(project_path(cfg["samples_csv"]))
    source_by_name = {row["image_name"]: row for row in samples}
    query_rows = read_csv_rows(project_path(cfg["oof_query_csv"]))
    candidate_rows = read_csv_rows(project_path(cfg["oof_candidate_csv"]))
    if not query_rows:
        raise RuntimeError("OOF query table is empty.")
    for row in query_rows:
        source = source_by_name.get(row["query_image_name"])
        if source is None or source["dataset_split"] != "train":
            raise RuntimeError(f"Trust labels may only come from development-pool train samples: {row['query_image_name']}")
        if is_final_holdout(source):
            raise RuntimeError("Refusing to access sealed final-holdout samples.")
    if len({row["query_image_name"] for row in query_rows}) != len(query_rows):
        raise RuntimeError("OOF trust query rows must be unique.")
    candidate_by_name = candidate_features(candidate_rows)
    if set(candidate_by_name) != {row["query_image_name"] for row in query_rows}:
        raise RuntimeError("Candidate table must cover every OOF trust query exactly once.")

    validation_fold = str(cfg["validation_oof_fold"])
    train_rows = [row for row in query_rows if row["oof_fold"] != validation_fold]
    validation_rows = [row for row in query_rows if row["oof_fold"] == validation_fold]
    if not train_rows or not validation_rows:
        raise RuntimeError(f"Validation OOF fold {validation_fold!r} does not create a train/validation split.")
    if {row["query_record_id"] for row in train_rows} & {row["query_record_id"] for row in validation_rows}:
        raise RuntimeError("Trust train and validation records overlap.")
    train_features, validation_features = make_matrix(train_rows, candidate_by_name), make_matrix(validation_rows, candidate_by_name)
    train_targets, validation_targets = make_targets(train_rows), make_targets(validation_rows)
    feature_mean, feature_std = train_features.mean(axis=0), train_features.std(axis=0)
    target_mean, target_std = train_targets.mean(axis=0), train_targets.std(axis=0)
    train_features = standardize(train_features, feature_mean, feature_std)
    validation_features = standardize(validation_features, feature_mean, feature_std)
    train_targets_scaled = standardize(train_targets, target_mean, target_std)

    model = CacheTrustPredictor(train_features.shape[1], float(cfg["dropout"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]))
    batch_size, patience = int(cfg["batch_size"]), int(cfg["early_stopping_patience"])
    checkpoint_dir = project_path(cfg["checkpoint_dir"])
    ensure_dir(checkpoint_dir)
    best_loss, best_epoch, stale, history = float("inf"), 0, 0, []
    for epoch in range(1, int(cfg["epochs"]) + 1):
        model.train()
        order = np.random.permutation(len(train_rows))
        losses = []
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            prediction = model(torch.from_numpy(train_features[indices]).to(device))
            target = torch.from_numpy(train_targets_scaled[indices]).to(device)
            loss = nn.functional.smooth_l1_loss(prediction, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            validation_prediction = model(torch.from_numpy(validation_features).to(device))
            validation_loss = float(nn.functional.smooth_l1_loss(validation_prediction, torch.from_numpy(standardize(validation_targets, target_mean, target_std)).to(device)).cpu())
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_loss": validation_loss})
        if validation_loss < best_loss - 1e-6:
            best_loss, best_epoch, stale = validation_loss, epoch, 0
            torch.save({
                "model_state": model.state_dict(), "feature_names": feature_names(), "feature_mean": feature_mean,
                "feature_std": feature_std, "target_mean": target_mean, "target_std": target_std, "config_section": section,
            }, checkpoint_dir / "best.pt")
        else:
            stale += 1
        if stale >= patience:
            break
    checkpoint = torch.load(checkpoint_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    with torch.no_grad():
        predicted_targets = model(torch.from_numpy(validation_features).to(device)).cpu().numpy() * target_std + target_mean
    scores = trust_scores(predicted_targets, target_mean, target_std)
    gate = choose_gate(validation_rows, scores, float(cfg["min_coverage"]), float(cfg["max_coverage"]), float(cfg["target_coverage"]))
    threshold = float(gate["selected"]["threshold"]) if gate["enabled"] else None
    accepted = scores >= threshold if threshold is not None else np.zeros(len(validation_rows), dtype=bool)
    output_rows = []
    for index, row in enumerate(validation_rows):
        output_rows.append({
            "query_record_id": row["query_record_id"], "query_image_name": row["query_image_name"], "query_probe": row["query_probe"], "oof_fold": row["oof_fold"],
            "status": "cache_hit" if accepted[index] else "cache_miss", "trust_score": f"{scores[index]:.6f}", "gate_threshold": "" if threshold is None else f"{threshold:.6f}",
            "predicted_tactile_diff_mae": f"{predicted_targets[index, 0]:.6f}", "predicted_tactile_ssim": f"{predicted_targets[index, 1]:.6f}", "predicted_tactile_mask_iou": f"{predicted_targets[index, 2]:.6f}",
            "selected_cache_record_id": row["selected_cache_record_id"], "top3_cache_record_ids": row["top3_cache_record_ids"],
            "rejection_reasons": rejection_reasons(predicted_targets[index], target_mean, target_std, scores[index], threshold),
            **{field: row[field] for field in ("ranker_best_score", "ranker_margin_normalized", "top3_tactile_embedding_disagreement", "c2_pred_score", "ranker_oracle_embedding_rank", *TARGET_FIELDS)},
        })
    output_csv = project_path(cfg["validation_output_csv"])
    write_csv_rows(output_csv, output_rows, OUTPUT_FIELDS)
    gate_output = {
        "mode": "phase4e_cache_trust_gate", "enabled": bool(gate["enabled"]), "threshold": threshold,
        "trust_score": "minimum standardized quality among predicted MAE, SSIM, and IoU; higher is better",
        "target_coverage": float(cfg["target_coverage"]), "coverage_bounds": [float(cfg["min_coverage"]), float(cfg["max_coverage"])],
        "validation_oof_fold": validation_fold, "feature_names": feature_names(), "gate_selection": gate,
        "online_output_schema": {"status": "cache_hit | cache_miss", "selected_cache_id": "cache entry ID", "top3_cache_ids": "ranked cache entry IDs", "trust_score": "scalar", "rejection_reasons": "pipe-separated diagnostics"},
    }
    write_json(project_path(cfg["gate_output_json"]), gate_output)
    summary = {
        "mode": "phase4e_oof_cache_trust_predictor", "device": str(device), "train_queries": len(train_rows), "validation_queries": len(validation_rows),
        "validation_oof_fold": validation_fold, "best_epoch": best_epoch, "epochs_ran": len(history), "best_validation_loss": best_loss,
        "gate": gate_output,
        "validation_metrics": {
            "all": metric_summary(validation_rows, accepted, make_targets(validation_rows).mean(axis=0)),
            "far_probe75_100": metric_summary(
                [row for row in validation_rows if int(row["query_probe"]) >= 75],
                accepted[np.asarray([int(row["query_probe"]) >= 75 for row in validation_rows])],
                make_targets(validation_rows).mean(axis=0),
            ),
            "near_mid": metric_summary(
                [row for row in validation_rows if int(row["query_probe"]) < 75],
                accepted[np.asarray([int(row["query_probe"]) < 75 for row in validation_rows])],
                make_targets(validation_rows).mean(axis=0),
            ),
        },
        "integrity": {
            "base_cache_labels": "strict Phase4E OOF", "trust_train_validation": "record-disjoint OOF folds", "sealed_final_holdout_rows_read": 0,
            "query_tactile_usage": "offline target only; all model features are online cache/query signals",
        }, "history": history,
    }
    write_json(project_path(cfg["metrics_json"]), summary)
    print({key: value for key, value in summary.items() if key != "history"})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a strict-OOF cache trust and cache-miss predictor for Phase 4E.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--section", default="phase4e_cache_trust_v1")
    args = parser.parse_args()
    train(args.config, args.section)


if __name__ == "__main__":
    main()
