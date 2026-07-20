from __future__ import annotations

import argparse
import random
from collections import defaultdict

import numpy as np
import torch
from torch import nn

from .config import load_config, project_path
from .utils import ensure_dir, read_csv_rows, write_csv_rows, write_json


SCORE_FIELDS = [
    "split", "query_record_id", "query_image_name", "query_probe", "query_case", "candidate_rank",
    "candidate_box48_hit", "calibrator_score", "calibrator_rank", "calibrator_selects_candidate",
]
GRID_FIELDS = [
    "quantile", "advantage_threshold", "gate_coverage", "rank_hard_corrected", "easy_corrupted",
    "overall_box48_rate", "far_box48_rate", "eligible",
]
QUERY_FIELDS = [
    "query_record_id", "query_image_name", "query_probe", "query_case", "baseline_box48_hit",
    "calibrator_candidate_rank", "calibrator_box48_hit", "advantage", "gate_applied", "gated_candidate_rank", "gated_box48_hit",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ranks_desc(values: np.ndarray) -> np.ndarray:
    result = np.empty(len(values), dtype=np.int32)
    result[np.argsort(-values, kind="stable")] = np.arange(1, len(values) + 1)
    return result


class CandidateCalibrator(nn.Module):
    def __init__(self, input_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.ReLU(inplace=True), nn.Linear(32, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


def float_value(row: dict[str, str], name: str) -> float:
    return float(row.get(name, "0") or 0.0)


def case_from_group(rows: list[dict[str, str]]) -> str:
    hits = [row["candidate_box48_hit"] == "1" for row in rows]
    return "easy" if hits[0] else ("rank_hard" if any(hits) else "proposal_miss")


def enrich_groups(rows: list[dict[str, str]], split: str, samples_by_name: dict[str, dict[str, str]]) -> tuple[list[dict], np.ndarray, np.ndarray, list[str]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["query_image_name"]].append(row)
    groups, features, labels, cases = [], [], [], []
    for image_name, candidate_rows in grouped.items():
        candidate_rows.sort(key=lambda row: int(row["candidate_rank"]))
        if len(candidate_rows) != 10:
            raise RuntimeError(f"Expected 10 candidates for {image_name}, got {len(candidate_rows)}.")
        case = candidate_rows[0].get("query_case") or case_from_group(candidate_rows)
        heat_context = np.asarray([float_value(row, "heatmap_ratio") for row in candidate_rows], dtype=np.float32)
        group_features = []
        for row in candidate_rows:
            source = samples_by_name.get(row["query_image_name"])
            if source is None:
                raise RuntimeError(f"Missing source sample for {row['query_image_name']}.")
            x, y = float_value(row, "candidate_x"), float_value(row, "candidate_y")
            dx, dy = x - float(source["tip_x"]), y - float(source["tip_y"])
            width, height = float(source["image_width"]), float(source["image_height"])
            direction_x, direction_y = float(source["direction_x"]), float(source["direction_y"])
            scale = max(width, height)
            rel_tip_x = float_value(row, "candidate_rel_tip_x") if row.get("candidate_rel_tip_x") else dx / width
            rel_tip_y = float_value(row, "candidate_rel_tip_y") if row.get("candidate_rel_tip_y") else dy / height
            direction_projection = float_value(row, "candidate_direction_projection") if row.get("candidate_direction_projection") else (dx * direction_x + dy * direction_y) / scale
            lateral_offset = float_value(row, "candidate_lateral_offset") if row.get("candidate_lateral_offset") else (-dx * direction_y + dy * direction_x) / scale
            probe = float_value(row, "query_probe") / 100.0
            base = np.asarray([
                float_value(row, "heatmap_ratio"),
                float_value(row, "cache_ranker_score_normalized"),
                float_value(row, "cache_score_rank_within_top10") / 10.0,
                rel_tip_x, rel_tip_y, direction_projection, lateral_offset,
                probe,
                float_value(row, "candidate_rank") / 10.0,
            ], dtype=np.float32)
            group_features.append(np.concatenate([base, heat_context], axis=0))
        groups.append({"split": split, "rows": candidate_rows, "case": case})
        features.append(np.stack(group_features))
        labels.append(np.asarray([float(row["candidate_box48_hit"]) for row in candidate_rows], dtype=np.float32))
        cases.append(case)
    return groups, np.stack(features), np.stack(labels), cases


def predict(model: nn.Module, features: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            batch = torch.from_numpy(features[start:start + batch_size]).to(device)
            outputs.append(model(batch.reshape(-1, batch.shape[-1])).reshape(batch.shape[:2]).cpu().numpy())
    return np.concatenate(outputs)


def evaluate_gate(groups: list[dict], labels: np.ndarray, scores: np.ndarray, threshold: float) -> tuple[list[dict[str, str]], dict[str, float | int]]:
    output_rows: list[dict[str, str]] = []
    for index, group in enumerate(groups):
        baseline, selected = 0, int(np.argmax(scores[index]))
        advantage = float(scores[index, selected] - scores[index, baseline])
        gate = selected != baseline and advantage >= threshold
        gated = selected if gate else baseline
        rows = group["rows"]
        output_rows.append({
            "query_record_id": rows[0]["query_record_id"], "query_image_name": rows[0]["query_image_name"], "query_probe": rows[0]["query_probe"],
            "query_case": group["case"], "baseline_box48_hit": str(int(labels[index, baseline])),
            "calibrator_candidate_rank": str(selected + 1), "calibrator_box48_hit": str(int(labels[index, selected])),
            "advantage": f"{advantage:.6f}", "gate_applied": str(int(gate)), "gated_candidate_rank": str(gated + 1), "gated_box48_hit": str(int(labels[index, gated])),
        })
    return output_rows, {
        "gate_coverage": float(np.mean([row["gate_applied"] == "1" for row in output_rows])),
        "rank_hard_corrected": int(sum(row["query_case"] == "rank_hard" and row["gated_box48_hit"] == "1" for row in output_rows)),
        "easy_corrupted": int(sum(row["query_case"] == "easy" and row["gated_box48_hit"] == "0" for row in output_rows)),
        "overall_box48_rate": float(np.mean([row["gated_box48_hit"] == "1" for row in output_rows])),
        "far_box48_rate": float(np.mean([row["gated_box48_hit"] == "1" for row in output_rows if int(row["query_probe"]) >= 75])),
    }


def train(config_path: str, section: str) -> dict:
    cfg = load_config(config_path)[section]
    set_seed(int(cfg.get("seed", 20260728)))
    train_rows = read_csv_rows(project_path(cfg["train_csv"]))
    val_rows = read_csv_rows(project_path(cfg["validation_csv"]))
    samples_by_name = {row["image_name"]: row for row in read_csv_rows(project_path(cfg["samples_csv"]))}
    train_groups, train_features, train_labels, train_cases = enrich_groups(train_rows, "train", samples_by_name)
    val_groups, val_features, val_labels, val_cases = enrich_groups(val_rows, "val", samples_by_name)
    feature_mean = train_features.reshape(-1, train_features.shape[-1]).mean(axis=0)
    feature_std = train_features.reshape(-1, train_features.shape[-1]).std(axis=0)
    feature_std[feature_std < 1e-6] = 1.0
    train_features = ((train_features - feature_mean) / feature_std).astype(np.float32)
    val_features = ((val_features - feature_mean) / feature_std).astype(np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CandidateCalibrator(train_features.shape[-1], float(cfg.get("dropout", 0.1))).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("learning_rate", 0.001)), weight_decay=float(cfg.get("weight_decay", 1e-4)))
    hard_weight, far_multiplier, easy_weight = float(cfg.get("hard_weight", 4.0)), float(cfg.get("far_hard_multiplier", 1.5)), float(cfg.get("easy_weight", 0.25))
    group_weights = []
    for group in train_groups:
        if group["case"] == "rank_hard":
            group_weights.append(hard_weight * (far_multiplier if int(group["rows"][0]["query_probe"]) >= 75 else 1.0))
        elif group["case"] == "easy":
            group_weights.append(easy_weight)
        else:
            group_weights.append(0.0)
    group_weights = np.asarray(group_weights, dtype=np.float32)
    batch_size, epochs = int(cfg.get("batch_size", 64)), int(cfg.get("epochs", 200))
    checkpoint_dir = project_path(cfg["checkpoint_dir"])
    ensure_dir(checkpoint_dir)
    best_score, best_epoch, stale, history = -1.0, 0, 0, []
    for epoch in range(1, epochs + 1):
        model.train()
        order = np.random.permutation(len(train_groups))
        losses = []
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            active = group_weights[indices] > 0
            if not active.any():
                continue
            indices = indices[active]
            features = torch.from_numpy(train_features[indices]).to(device)
            labels = torch.from_numpy(train_labels[indices]).to(device)
            weights = torch.from_numpy(group_weights[indices]).to(device)
            logits = model(features.reshape(-1, features.shape[-1])).reshape(features.shape[:2])
            target_distribution = labels / labels.sum(dim=1, keepdim=True).clamp_min(1.0)
            listwise = -(target_distribution * torch.log_softmax(logits, dim=1)).sum(dim=1)
            bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none").mean(dim=1)
            loss = ((listwise + float(cfg.get("bce_weight", 0.25)) * bce) * weights).sum() / weights.sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_scores = predict(model, val_features, device, batch_size)
        raw_rows, raw_metrics = evaluate_gate(val_groups, val_labels, val_scores, threshold=0.0)
        score = raw_metrics["rank_hard_corrected"] - raw_metrics["easy_corrupted"]
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), **raw_metrics, "selection_score": score})
        if score > best_score:
            best_score, best_epoch, stale = score, epoch, 0
            torch.save({"model_state": model.state_dict(), "feature_mean": feature_mean, "feature_std": feature_std, "config_section": section}, checkpoint_dir / "best.pt")
        else:
            stale += 1
        if stale >= int(cfg.get("early_stopping_patience", 40)):
            break
    checkpoint = torch.load(checkpoint_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    val_scores = predict(model, val_features, device, batch_size)
    val_advantages = np.asarray([float(np.max(score) - score[0]) for score in val_scores if int(np.argmax(score)) != 0], dtype=np.float32)
    quantiles = [float(value) for value in cfg.get("gate_quantiles", [0.0, 0.25, 0.5, 0.75, 0.9])]
    thresholds = [float(np.quantile(val_advantages, quantile)) if len(val_advantages) else float("inf") for quantile in quantiles]
    grid_rows, experiments = [], []
    baseline_rows, baseline_metrics = evaluate_gate(val_groups, val_labels, val_scores, threshold=float("inf"))
    baseline_far = float(np.mean([row["gated_box48_hit"] == "1" for row in baseline_rows if int(row["query_probe"]) >= 75]))
    for grid_index, (quantile, threshold) in enumerate(zip(quantiles, thresholds)):
        gated_rows, metrics = evaluate_gate(val_groups, val_labels, val_scores, threshold)
        eligible = (
            metrics["rank_hard_corrected"] >= int(cfg.get("minimum_rank_hard_corrections", 2))
            and metrics["easy_corrupted"] <= int(cfg.get("maximum_easy_corruptions", 1))
            and metrics["overall_box48_rate"] >= baseline_metrics["overall_box48_rate"]
            and metrics["far_box48_rate"] >= baseline_far
        )
        grid_rows.append({"quantile": f"{quantile:.3f}", "advantage_threshold": f"{threshold:.6f}", **{key: f"{value:.6f}" if isinstance(value, float) else str(value) for key, value in metrics.items()}, "eligible": str(int(eligible))})
        experiments.append((metrics["rank_hard_corrected"], -metrics["easy_corrupted"], metrics["far_box48_rate"], grid_index, gated_rows, eligible))
    eligible = [item for item in experiments if item[5]]
    selected = max(eligible, key=lambda item: (item[0], item[1], item[2])) if eligible else None
    if selected is None:
        selected_rows, selected_grid = baseline_rows, None
    else:
        selected_rows, selected_grid = selected[4], grid_rows[selected[3]]
    score_rows = []
    for group_index, group in enumerate(val_groups):
        ranks = ranks_desc(val_scores[group_index])
        for candidate_index, row in enumerate(group["rows"]):
            score_rows.append({
                "split": "val", "query_record_id": row["query_record_id"], "query_image_name": row["query_image_name"], "query_probe": row["query_probe"],
                "query_case": group["case"], "candidate_rank": row["candidate_rank"], "candidate_box48_hit": row["candidate_box48_hit"],
                "calibrator_score": f"{val_scores[group_index, candidate_index]:.6f}", "calibrator_rank": str(int(ranks[candidate_index])),
                "calibrator_selects_candidate": str(int(ranks[candidate_index] == 1)),
            })
    summary = {
        "mode": "validation_only_topk_candidate_calibrator", "device": str(device), "train_queries": len(train_groups), "validation_queries": len(val_groups),
        "train_case_counts": {case: train_cases.count(case) for case in ("easy", "rank_hard", "proposal_miss")},
        "validation_case_counts": {case: val_cases.count(case) for case in ("easy", "rank_hard", "proposal_miss")},
        "best_epoch": best_epoch, "epochs_ran": len(history), "selection_policy": "maximize rank-hard corrections subject to easy corruption <= 1 and non-worse overall/far Box48",
        "selected_gate": selected_grid, "baseline": baseline_metrics, "gated": evaluate_gate(val_groups, val_labels, val_scores, float(selected_grid["advantage_threshold"]) if selected_grid else float("inf"))[1],
        "checkpoint": str(checkpoint_dir / "best.pt"), "history": history,
    }
    write_csv_rows(project_path(cfg["score_output_csv"]), score_rows, SCORE_FIELDS)
    write_csv_rows(project_path(cfg["query_output_csv"]), selected_rows, QUERY_FIELDS)
    write_csv_rows(project_path(cfg["grid_output_csv"]), grid_rows, GRID_FIELDS)
    write_json(project_path(cfg["metrics_json"]), summary)
    print({key: value for key, value in summary.items() if key != "history"})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Top-K candidate calibrator from OOF cache-aware features.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--section", default="topk_candidate_calibrator_phase35_v3")
    args = parser.parse_args()
    train(args.config, args.section)


if __name__ == "__main__":
    main()
