"""Retrain cache trust from Phase4H strict OOF after ranking validation passes."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict

import numpy as np
import torch
from torch import nn

from .config import load_config, project_path
from .phase4h_dino_adaptation import assert_development_only, record_hash_split
from .train_phase4b_predicted_box_cache_ranker import set_seed
from .train_phase4e_cache_trust import CacheTrustPredictor, choose_gate, trust_scores
from .utils import ensure_dir, read_csv_rows, write_csv_rows, write_json


TARGET_FIELDS = ("tactile_diff_mae", "tactile_ssim", "tactile_mask_iou")
OUTPUT_FIELDS = [
    "query_record_id", "query_image_name", "query_probe", "status", "trust_score",
    "gate_threshold", "final_selection_source", "selected_cache_record_id",
    "selected_cache_image_name", "predicted_tactile_diff_mae", "predicted_tactile_ssim",
    "predicted_tactile_mask_iou", "tactile_diff_mae", "tactile_ssim",
    "tactile_mask_iou", "rejection_reasons",
]


def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def finite(value: str | float, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def grouped(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    output: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        output[row["query_image_name"]].append(row)
    for group in output.values():
        group.sort(key=lambda row: int(row["candidate_rank"]))
    return output


def feature_names() -> list[str]:
    return [
        "dino_accept_probability", "aligned_source", "predicted_ttc", "ttc_entropy",
        "trajectory_stability", "query_padding_ratio", "aligned_ranker_margin_normalized",
        "aligned_ranker_entropy", "selected_best_score", "selected_margin_normalized",
        "candidate_score_std", "top3_predicted_latent_mean", "top3_predicted_latent_std",
        "top1_scale_std",
    ]


def make_features(
    query_rows: list[dict[str, str]],
    aligned_by_name: dict[str, dict[str, str]],
    candidates_by_name: dict[str, list[dict[str, str]]],
) -> np.ndarray:
    values = []
    for row in query_rows:
        name = row["query_image_name"]
        aligned = aligned_by_name[name]
        group = candidates_by_name[name]
        scores = np.asarray([finite(item["candidate_score"]) for item in group], dtype=np.float32)
        predicted = np.asarray(
            [finite(item.get("predicted_tactile_latent_distance", "")) for item in group[:3]],
            dtype=np.float32,
        )
        scale = np.asarray(
            [
                finite(group[0].get(field, ""))
                for field in ("detail_patch_score", "context_patch_score", "wide_patch_score")
                if group[0].get(field, "") != ""
            ],
            dtype=np.float32,
        )
        values.append(
            [
                finite(row["dino_accept_probability"]),
                float(row["final_selection_source"] == "aligned_dino"),
                finite(aligned["predicted_ttc"]) / 100.0,
                finite(aligned["ttc_entropy"]),
                finite(aligned["trajectory_stability"]),
                finite(aligned["query_padding_ratio"]),
                finite(aligned["ranker_margin_normalized"]),
                finite(aligned["ranker_entropy"]),
                float(scores[0]),
                float((scores[1] - scores[0]) / max(float(scores.std()), 1e-6)),
                float(scores.std()),
                float(predicted.mean()),
                float(predicted.std()),
                float(scale.std()) if len(scale) else 0.0,
            ]
        )
    return np.asarray(values, dtype=np.float32)


def targets(rows: list[dict[str, str]]) -> np.ndarray:
    return np.asarray(
        [[finite(row[field]) for field in TARGET_FIELDS] for row in rows],
        dtype=np.float32,
    )


def train(config_path: str, section: str) -> dict:
    cfg = load_config(config_path)[section]
    validation_report = load_json(project_path(cfg["phase4h_validation_metrics_json"]))
    if not validation_report.get("accepted", False):
        raise RuntimeError("Phase4H ranking validation did not pass; cache-trust retraining is blocked")
    set_seed(int(cfg["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples = read_csv_rows(project_path(cfg["samples_csv"]))
    assert_development_only(samples, project_path(cfg["final_partition_csv"]))
    oof_rows = read_csv_rows(project_path(cfg["oof_gated_query_csv"]))
    oof_aligned = {
        row["query_image_name"]: row for row in read_csv_rows(project_path(cfg["oof_aligned_query_csv"]))
    }
    oof_candidates = grouped(read_csv_rows(project_path(cfg["oof_gated_candidate_csv"])))
    validation_rows = read_csv_rows(project_path(cfg["validation_gated_query_csv"]))
    validation_aligned = {
        row["query_image_name"]: row for row in read_csv_rows(project_path(cfg["validation_aligned_query_csv"]))
    }
    validation_candidates = grouped(read_csv_rows(project_path(cfg["validation_gated_candidate_csv"])))
    raw_train = make_features(oof_rows, oof_aligned, oof_candidates)
    raw_validation = make_features(validation_rows, validation_aligned, validation_candidates)
    train_targets = targets(oof_rows)
    validation_targets = targets(validation_rows)
    inner_validation = np.asarray(
        [
            index for index, row in enumerate(oof_rows)
            if record_hash_split(
                row["query_record_id"],
                float(cfg["inner_validation_fraction"]),
                int(cfg["inner_split_seed"]),
            )
        ],
        dtype=np.int32,
    )
    inner_set = set(inner_validation.tolist())
    fit = np.asarray([index for index in range(len(oof_rows)) if index not in inner_set], dtype=np.int32)
    feature_mean, feature_std = raw_train[fit].mean(axis=0), raw_train[fit].std(axis=0)
    feature_std[feature_std < 1e-6] = 1.0
    target_mean, target_std = train_targets[fit].mean(axis=0), train_targets[fit].std(axis=0)
    target_std[target_std < 1e-6] = 1.0
    train_features = (raw_train - feature_mean) / feature_std
    validation_features = (raw_validation - feature_mean) / feature_std
    scaled_targets = (train_targets - target_mean) / target_std
    model = CacheTrustPredictor(train_features.shape[1], float(cfg["dropout"])).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]),
    )
    checkpoint_path = project_path(cfg["checkpoint"])
    ensure_dir(checkpoint_path.parent)
    best_loss, best_epoch, stale, history = float("inf"), 0, 0, []
    batch_size = int(cfg["batch_size"])
    for epoch in range(1, int(cfg["epochs"]) + 1):
        model.train()
        order = np.random.permutation(fit)
        losses = []
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            prediction = model(torch.from_numpy(train_features[batch]).to(device))
            loss = nn.functional.smooth_l1_loss(
                prediction, torch.from_numpy(scaled_targets[batch]).to(device),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            prediction = model(torch.from_numpy(train_features[inner_validation]).to(device))
            current = float(
                nn.functional.smooth_l1_loss(
                    prediction, torch.from_numpy(scaled_targets[inner_validation]).to(device),
                ).cpu()
            )
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "inner_validation_loss": current})
        if current < best_loss - 1e-6:
            best_loss, best_epoch, stale = current, epoch, 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "feature_names": feature_names(),
                    "feature_mean": feature_mean,
                    "feature_std": feature_std,
                    "target_mean": target_mean,
                    "target_std": target_std,
                    "source": "Phase4H strict OOF only",
                },
                checkpoint_path,
            )
        else:
            stale += 1
        if stale >= int(cfg["early_stopping_patience"]):
            break
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    with torch.no_grad():
        scaled_prediction = model(torch.from_numpy(validation_features).to(device)).cpu().numpy()
    predicted = scaled_prediction * target_std + target_mean
    scores = trust_scores(predicted, target_mean, target_std)
    gate = choose_gate(
        validation_rows,
        scores,
        float(cfg["min_coverage"]),
        float(cfg["max_coverage"]),
        float(cfg["target_coverage"]),
    )
    threshold = float(gate["selected"]["threshold"]) if gate["enabled"] else None
    accepted = scores >= threshold if threshold is not None else np.zeros(len(validation_rows), dtype=bool)
    output = []
    for index, row in enumerate(validation_rows):
        output.append(
            {
                "query_record_id": row["query_record_id"],
                "query_image_name": row["query_image_name"],
                "query_probe": row["query_probe"],
                "status": "cache_hit" if accepted[index] else "cache_miss",
                "trust_score": f"{scores[index]:.6f}",
                "gate_threshold": "" if threshold is None else f"{threshold:.6f}",
                "final_selection_source": row["final_selection_source"],
                "selected_cache_record_id": row["selected_cache_record_id"],
                "selected_cache_image_name": row["selected_cache_image_name"],
                **{
                    f"predicted_{field}": f"{predicted[index, metric_index]:.6f}"
                    for metric_index, field in enumerate(TARGET_FIELDS)
                },
                **{field: row[field] for field in TARGET_FIELDS},
                "rejection_reasons": "" if accepted[index] else "phase4h_low_cache_trust",
            }
        )
    write_csv_rows(project_path(cfg["validation_output_csv"]), output, OUTPUT_FIELDS)
    gate_output = {
        "mode": "phase4h_cache_trust_v1",
        "enabled": bool(gate["enabled"]),
        "threshold": threshold,
        "threshold_selection_source": "Phase4H development validation only",
        "feature_names": feature_names(),
        "selection": gate,
    }
    write_json(project_path(cfg["gate_json"]), gate_output)
    summary = {
        "mode": "phase4h_cache_trust_retrained_from_strict_oof_v1",
        "device": str(device),
        "train_queries": len(oof_rows),
        "validation_queries": len(validation_rows),
        "best_epoch": best_epoch,
        "best_inner_validation_loss": best_loss,
        "gate": gate_output,
        "history": history,
        "integrity": {
            "old_v1_trust_checkpoint_reused": False,
            "ranker_training_source": "strict Phase4H OOF",
            "threshold_selection_source": "Phase4H development validation",
            "query_tactile_input": False,
            "sealed_final_holdout_rows_read": 0,
        },
    }
    write_json(project_path(cfg["metrics_json"]), summary)
    print({key: value for key, value in summary.items() if key != "history"})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrain cache trust after Phase4H validation passes.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--section", default="phase4h_cache_trust_v1")
    args = parser.parse_args()
    train(args.config, args.section)


if __name__ == "__main__":
    main()
