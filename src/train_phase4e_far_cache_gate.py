from __future__ import annotations

import argparse
import math

import numpy as np
import torch
from torch import nn

from .config import load_config, project_path
from .train_phase4b_predicted_box_cache_ranker import is_final_holdout, set_seed
from .train_phase4e_cache_trust import candidate_features, feature_names, finite, make_matrix, standardize
from .utils import ensure_dir, read_csv_rows, write_csv_rows, write_json


TARGET_FIELDS = ("tactile_diff_mae", "tactile_ssim", "tactile_mask_iou")
OUTPUT_FIELDS = [
    "query_record_id", "query_image_name", "query_probe", "oof_fold", "status", "far_quality_probability",
    "gate_threshold", "selected_cache_record_id", "top3_cache_record_ids", "rejection_reasons",
    "ranker_best_score", "ranker_margin_normalized", "top3_tactile_embedding_disagreement", "c2_pred_score",
    "ranker_oracle_embedding_rank", *TARGET_FIELDS,
]


class FarCacheGate(nn.Module):
    """A compact, far-only quality classifier fed exclusively by online cache signals."""

    def __init__(self, feature_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_dim, 96), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(96, 32), nn.ReLU(inplace=True), nn.Linear(32, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(1)


def targets(rows: list[dict[str, str]]) -> np.ndarray:
    return np.asarray([[finite(row[field]) for field in TARGET_FIELDS] for row in rows], dtype=np.float32)


def quality_labels(rows: list[dict[str, str]], baseline: np.ndarray) -> np.ndarray:
    values = targets(rows)
    return (
        (values[:, 0] < baseline[0])
        & (values[:, 1] > baseline[1])
        & (values[:, 2] >= baseline[2])
    ).astype(np.float32)


def tactile_best_rates(rows: list[dict[str, str]], mask: np.ndarray) -> dict:
    ranks = np.asarray([int(rows[index]["ranker_oracle_embedding_rank"]) for index in np.flatnonzero(mask)], dtype=np.int32)
    return {
        "tactile_best_top1_rate": float(np.mean(ranks == 1)) if len(ranks) else None,
        "tactile_best_top3_rate": float(np.mean(ranks <= 3)) if len(ranks) else None,
    }


def selected_summary(rows: list[dict[str, str]], accepted: np.ndarray, baseline: np.ndarray) -> dict:
    values = targets(rows)
    actual_good = quality_labels(rows, baseline).astype(bool)
    result = {
        "queries": len(rows), "coverage": float(accepted.mean()), "accepted_queries": int(accepted.sum()),
        "baseline": {field: float(value) for field, value in zip(TARGET_FIELDS, baseline, strict=True)},
    }
    for name, mask in (("all", np.ones(len(rows), dtype=bool)), ("accepted", accepted), ("miss", ~accepted)):
        subset = values[mask]
        result[name] = {
            "queries": int(mask.sum()),
            **{field: float(subset[:, index].mean()) if len(subset) else None for index, field in enumerate(TARGET_FIELDS)},
            **tactile_best_rates(rows, mask),
        }
    false_accept = accepted & ~actual_good
    false_reject = ~accepted & actual_good
    result["quality_confusion"] = {
        "quality_definition": "actual far MAE below far baseline, SSIM above it, and IoU at least equal to it",
        "actual_good_queries": int(actual_good.sum()), "true_accepts": int((accepted & actual_good).sum()),
        "false_accepts": int(false_accept.sum()), "false_rejects": int(false_reject.sum()),
        "false_accept_rate_among_hits": float(false_accept.sum() / max(int(accepted.sum()), 1)),
        "false_reject_rate_among_actual_good": float(false_reject.sum() / max(int(actual_good.sum()), 1)),
    }
    return result


def choose_threshold(rows: list[dict[str, str]], probabilities: np.ndarray, minimum: float, maximum: float, target: float) -> dict:
    if not 0 < minimum <= target <= maximum <= 1:
        raise ValueError("Coverage settings must satisfy 0 < minimum <= target <= maximum <= 1.")
    baseline = targets(rows).mean(axis=0)
    all_rates = tactile_best_rates(rows, np.ones(len(rows), dtype=bool))
    options = []
    for coverage in np.linspace(minimum, maximum, 81):
        count = max(1, int(round(len(rows) * coverage)))
        threshold = float(np.sort(probabilities)[-count])
        accepted = probabilities >= threshold
        summary = selected_summary(rows, accepted, baseline)
        selected = summary["accepted"]
        passes = bool(
            selected["tactile_diff_mae"] < baseline[0]
            and selected["tactile_ssim"] > baseline[1]
            and selected["tactile_mask_iou"] >= baseline[2]
            and selected["tactile_best_top1_rate"] >= all_rates["tactile_best_top1_rate"]
            and selected["tactile_best_top3_rate"] >= all_rates["tactile_best_top3_rate"]
        )
        precision = 1.0 - summary["quality_confusion"]["false_accept_rate_among_hits"]
        options.append({"threshold": threshold, "coverage": float(accepted.mean()), "passes_guard": passes, "precision": precision, "summary": summary})
    valid = [option for option in options if option["passes_guard"]]
    if not valid:
        return {"enabled": False, "baseline": {field: float(value) for field, value in zip(TARGET_FIELDS, baseline, strict=True)}, "options": options}
    chosen = min(valid, key=lambda option: (abs(option["coverage"] - target), -option["precision"]))
    return {"enabled": True, "baseline": {field: float(value) for field, value in zip(TARGET_FIELDS, baseline, strict=True)}, "selected": chosen, "options": options}


def train(config_path: str, section: str) -> dict:
    cfg = load_config(config_path)[section]
    set_seed(int(cfg["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples = read_csv_rows(project_path(cfg["samples_csv"]))
    source_by_name = {row["image_name"]: row for row in samples}
    queries = [row for row in read_csv_rows(project_path(cfg["oof_query_csv"])) if int(row["query_probe"]) >= 75]
    candidates = read_csv_rows(project_path(cfg["oof_candidate_csv"]))
    if not queries:
        raise RuntimeError("No far OOF cache queries found.")
    for row in queries:
        source = source_by_name.get(row["query_image_name"])
        if source is None or source["dataset_split"] != "train" or is_final_holdout(source):
            raise RuntimeError("Far gate may only use development-train strict OOF data, never final holdout.")
    aggregate = candidate_features(candidates)
    validation_fold = str(cfg["validation_oof_fold"])
    train_rows = [row for row in queries if row["oof_fold"] != validation_fold]
    calibration_rows = [row for row in queries if row["oof_fold"] == validation_fold]
    if {row["query_record_id"] for row in train_rows} & {row["query_record_id"] for row in calibration_rows}:
        raise RuntimeError("Far gate train and calibration records overlap.")
    train_features = make_matrix(train_rows, aggregate)
    calibration_features = make_matrix(calibration_rows, aggregate)
    feature_mean, feature_std = train_features.mean(axis=0), train_features.std(axis=0)
    train_features = standardize(train_features, feature_mean, feature_std)
    calibration_features = standardize(calibration_features, feature_mean, feature_std)
    train_baseline = targets(train_rows).mean(axis=0)
    train_labels = quality_labels(train_rows, train_baseline)
    positive_count = float(train_labels.sum())
    if positive_count < 8 or positive_count >= len(train_labels) - 8:
        raise RuntimeError(f"Far quality labels are too imbalanced for stable training: positives={positive_count} total={len(train_labels)}")
    pos_weight = torch.tensor([(len(train_labels) - positive_count) / positive_count], dtype=torch.float32, device=device)
    model = FarCacheGate(train_features.shape[1], float(cfg["dropout"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]))
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    checkpoint_dir = project_path(cfg["checkpoint_dir"])
    ensure_dir(checkpoint_dir)
    batch_size, patience = int(cfg["batch_size"]), int(cfg["early_stopping_patience"])
    best_loss, best_epoch, stale, history = float("inf"), 0, 0, []
    calibration_labels = quality_labels(calibration_rows, targets(calibration_rows).mean(axis=0))
    for epoch in range(1, int(cfg["epochs"]) + 1):
        model.train()
        order = np.random.permutation(len(train_rows))
        losses = []
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            logits = model(torch.from_numpy(train_features[indices]).to(device))
            loss = criterion(logits, torch.from_numpy(train_labels[indices]).to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            logits = model(torch.from_numpy(calibration_features).to(device))
            loss = float(criterion(logits, torch.from_numpy(calibration_labels).to(device)).cpu())
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "calibration_loss": loss})
        if loss < best_loss - 1e-6:
            best_loss, best_epoch, stale = loss, epoch, 0
            torch.save({"model_state": model.state_dict(), "feature_names": feature_names(), "feature_mean": feature_mean, "feature_std": feature_std, "config_section": section}, checkpoint_dir / "best.pt")
        else:
            stale += 1
        if stale >= patience:
            break
    checkpoint = torch.load(checkpoint_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(model(torch.from_numpy(calibration_features).to(device))).cpu().numpy()
    gate = choose_threshold(calibration_rows, probabilities, float(cfg["min_coverage"]), float(cfg["max_coverage"]), float(cfg["target_coverage"]))
    threshold = float(gate["selected"]["threshold"]) if gate["enabled"] else None
    accepted = probabilities >= threshold if threshold is not None else np.zeros(len(calibration_rows), dtype=bool)
    output_rows = []
    for index, row in enumerate(calibration_rows):
        reasons = "" if accepted[index] else "far_gate_low_quality_probability"
        output_rows.append({
            "query_record_id": row["query_record_id"], "query_image_name": row["query_image_name"], "query_probe": row["query_probe"], "oof_fold": row["oof_fold"],
            "status": "cache_hit" if accepted[index] else "cache_miss", "far_quality_probability": f"{probabilities[index]:.6f}", "gate_threshold": "" if threshold is None else f"{threshold:.6f}",
            "selected_cache_record_id": row["selected_cache_record_id"], "top3_cache_record_ids": row["top3_cache_record_ids"], "rejection_reasons": reasons,
            **{field: row[field] for field in ("ranker_best_score", "ranker_margin_normalized", "top3_tactile_embedding_disagreement", "c2_pred_score", "ranker_oracle_embedding_rank", *TARGET_FIELDS)},
        })
    write_csv_rows(project_path(cfg["calibration_output_csv"]), output_rows, OUTPUT_FIELDS)
    gate_output = {
        "mode": "phase4e_far_cache_gate", "enabled": bool(gate["enabled"]), "threshold": threshold,
        "coverage_bounds": [float(cfg["min_coverage"]), float(cfg["max_coverage"])], "target_coverage": float(cfg["target_coverage"]),
        "calibration_oof_fold": validation_fold, "feature_names": feature_names(), "selection": gate,
        "online_output_schema": {"status": "cache_hit | cache_miss", "far_quality_probability": "0..1", "rejection_reasons": "far gate diagnostics"},
    }
    write_json(project_path(cfg["gate_output_json"]), gate_output)
    summary = {
        "mode": "phase4e_far_cache_gate_strict_oof", "device": str(device), "train_queries": len(train_rows), "calibration_queries": len(calibration_rows),
        "positive_train_labels": int(positive_count), "best_epoch": best_epoch, "epochs_ran": len(history), "best_calibration_loss": best_loss,
        "gate": gate_output,
        "integrity": {"cache_ranker": "strict Phase4E OOF", "gate_train_calibration": "record-disjoint OOF folds", "sealed_final_holdout_rows_read": 0, "query_tactile_usage": "offline quality supervision only"},
        "history": history,
    }
    write_json(project_path(cfg["metrics_json"]), summary)
    print({key: value for key, value in summary.items() if key != "history"})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a far-only strict-OOF cache quality gate for Phase 4E.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--section", default="phase4e_far_cache_gate_v1")
    args = parser.parse_args()
    train(args.config, args.section)


if __name__ == "__main__":
    main()
