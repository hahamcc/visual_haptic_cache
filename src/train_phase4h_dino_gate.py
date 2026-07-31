"""Strict-OOF query-level safety gate between frozen V1 and aligned DINO."""
from __future__ import annotations

import argparse
import math
from collections import defaultdict

import numpy as np
import torch
from torch import nn

from .config import load_config, project_path
from .phase4h_dino_adaptation import DinoSafetyGate, assert_development_only, record_hash_split
from .train_phase4b_predicted_box_cache_ranker import set_seed
from .utils import ensure_dir, read_csv_rows, write_csv_rows, write_json


METRICS = ("tactile_diff_mae", "tactile_ssim", "tactile_mask_iou")
OUTPUT_QUERY_FIELDS = [
    "query_record_id", "query_image_name", "query_probe", "oof_fold",
    "dino_accept_probability", "gate_threshold", "dino_accepted", "final_selection_source",
    "selected_cache_record_id", "selected_cache_image_name", "ranker_oracle_embedding_rank",
    "tactile_diff_mae", "tactile_ssim", "tactile_mask_iou",
    "v1_selected_cache_image_name", "dino_selected_cache_image_name",
    "v1_tactile_diff_mae", "v1_tactile_ssim", "v1_tactile_mask_iou",
    "dino_tactile_diff_mae", "dino_tactile_ssim", "dino_tactile_mask_iou",
    "strict_triple_win_label",
]
OUTPUT_CANDIDATE_FIELDS = [
    "query_record_id", "query_image_name", "query_probe", "oof_fold", "candidate_rank",
    "candidate_score", "candidate_record_id", "candidate_image_name",
    "predicted_tactile_latent_distance", "candidate_tactile_latent_distance",
    "detail_patch_score", "context_patch_score", "wide_patch_score",
    "position_aware_match_score", "hard_negative_flag", "candidate_oracle_embedding_rank",
    "dino_accept_probability", "final_selection_source",
]


def finite(value: str | float, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def grouped(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    result: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        result[row["query_image_name"]].append(row)
    for values in result.values():
        values.sort(key=lambda row: int(row["candidate_rank"]))
    return result


def normalized_entropy(scores: np.ndarray) -> float:
    logits = -scores
    logits -= logits.max()
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum()
    return float(-np.sum(probabilities * np.log(np.maximum(probabilities, 1e-12))) / math.log(len(scores)))


def feature_names() -> list[str]:
    return [
        "dino_best_score", "dino_margin", "dino_margin_normalized", "dino_entropy",
        "v1_best_score", "v1_margin", "v1_margin_normalized", "rank_disagreement",
        "predicted_tactile_nearest_distance", "predicted_tactile_margin",
        "ttc_entropy", "trajectory_stability", "crop_padding_ratio",
        "scale_score_std", "scale_margin_mean",
    ]


def make_features(
    queries: list[dict[str, str]],
    v1_by_name: dict[str, dict[str, str]],
    v1_candidates: dict[str, list[dict[str, str]]],
    dino_candidates: dict[str, list[dict[str, str]]],
) -> np.ndarray:
    output = []
    for query in queries:
        name = query["query_image_name"]
        v1 = v1_by_name[name]
        v1_group, dino_group = v1_candidates[name], dino_candidates[name]
        dino_scores = np.asarray([finite(row["candidate_score"]) for row in dino_group], dtype=np.float32)
        v1_scores = np.asarray([finite(row["candidate_score"]) for row in v1_group], dtype=np.float32)
        scale_top = np.asarray(
            [
                finite(dino_group[0].get(field, ""))
                for field in ("detail_patch_score", "context_patch_score", "wide_patch_score")
                if dino_group[0].get(field, "") != ""
            ],
            dtype=np.float32,
        )
        scale_margins = []
        for field in ("detail_patch_score", "context_patch_score", "wide_patch_score"):
            if dino_group[0].get(field, "") != "" and dino_group[1].get(field, "") != "":
                scale_margins.append(finite(dino_group[0][field]) - finite(dino_group[1][field]))
        dino_std = max(float(dino_scores.std()), 1e-6)
        v1_std = max(float(v1_scores.std()), 1e-6)
        output.append(
            [
                dino_scores[0],
                dino_scores[1] - dino_scores[0],
                (dino_scores[1] - dino_scores[0]) / dino_std,
                normalized_entropy(dino_scores),
                v1_scores[0],
                v1_scores[1] - v1_scores[0],
                (v1_scores[1] - v1_scores[0]) / v1_std,
                float(v1["selected_cache_image_name"] != query["selected_cache_image_name"]),
                finite(dino_group[0]["predicted_tactile_latent_distance"]),
                finite(dino_group[1]["predicted_tactile_latent_distance"]) - finite(dino_group[0]["predicted_tactile_latent_distance"]),
                finite(query["ttc_entropy"]),
                finite(query["trajectory_stability"]),
                finite(query["query_padding_ratio"]),
                float(scale_top.std()) if len(scale_top) else 0.0,
                float(np.mean(scale_margins)) if scale_margins else 0.0,
            ]
        )
    return np.asarray(output, dtype=np.float32)


def labels(v1_rows: list[dict[str, str]], dino_rows: list[dict[str, str]]) -> np.ndarray:
    v1 = {row["query_image_name"]: row for row in v1_rows}
    return np.asarray(
        [
            float(
                finite(row["tactile_diff_mae"]) < finite(v1[row["query_image_name"]]["tactile_diff_mae"])
                and finite(row["tactile_ssim"]) >= finite(v1[row["query_image_name"]]["tactile_ssim"])
                and finite(row["tactile_mask_iou"]) >= finite(v1[row["query_image_name"]]["tactile_mask_iou"])
            )
            for row in dino_rows
        ],
        dtype=np.float32,
    )


def standardize(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (values - mean) / np.maximum(std, 1e-6)


def train_model(
    features: np.ndarray,
    targets: np.ndarray,
    fit: np.ndarray,
    validation: np.ndarray,
    cfg: dict,
    device: torch.device,
    checkpoint_path,
    metadata: dict,
) -> tuple[DinoSafetyGate, dict]:
    positive = float(targets[fit].sum())
    if positive < 4 or positive >= len(fit) - 4:
        raise RuntimeError(f"Phase4H gate labels are too imbalanced: positives={positive} total={len(fit)}")
    model = DinoSafetyGate(features.shape[1], float(cfg["dropout"])).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]),
    )
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([(len(fit) - positive) / positive], device=device)
    )
    best_loss, best_epoch, stale, history = float("inf"), 0, 0, []
    batch_size = int(cfg["batch_size"])
    for epoch in range(1, int(cfg["epochs"]) + 1):
        model.train()
        order = np.random.permutation(fit)
        losses = []
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            logits = model(torch.from_numpy(features[batch]).to(device))
            loss = criterion(logits, torch.from_numpy(targets[batch]).to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            val_logits = model(torch.from_numpy(features[validation]).to(device))
            val_loss = float(criterion(val_logits, torch.from_numpy(targets[validation]).to(device)).cpu())
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_loss": val_loss})
        if val_loss < best_loss - 1e-6:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            ensure_dir(checkpoint_path.parent)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "feature_names": feature_names(),
                    "feature_mean": metadata["feature_mean"],
                    "feature_std": metadata["feature_std"],
                    "metadata": metadata,
                },
                checkpoint_path,
            )
        else:
            stale += 1
        if stale >= int(cfg["early_stopping_patience"]):
            break
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    return model, {
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "epochs_ran": len(history),
        "history": history,
    }


def logits_for(model: DinoSafetyGate, features: np.ndarray, indices: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(features[indices]).to(device)).cpu().numpy()


def calibrate_temperature(logits: np.ndarray, targets: np.ndarray) -> float:
    value = torch.tensor([0.0], requires_grad=True)
    source = torch.from_numpy(logits.astype(np.float32))
    labels_tensor = torch.from_numpy(targets.astype(np.float32))
    optimizer = torch.optim.LBFGS([value], lr=0.1, max_iter=100)

    def closure():
        optimizer.zero_grad()
        temperature = value.exp().clamp(0.25, 4.0)
        loss = nn.functional.binary_cross_entropy_with_logits(source / temperature, labels_tensor)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(value.detach().exp().clamp(0.25, 4.0))


def selected_rows(
    v1_rows: list[dict[str, str]],
    dino_rows: list[dict[str, str]],
    accepted: np.ndarray,
) -> list[dict[str, str]]:
    v1 = {row["query_image_name"]: row for row in v1_rows}
    return [row if accepted[index] else v1[row["query_image_name"]] for index, row in enumerate(dino_rows)]


def metric_summary(rows: list[dict[str, str]], predicate=lambda _: True) -> dict:
    subset = [row for row in rows if predicate(row)]
    return {
        "queries": len(subset),
        "tactile_diff_mae": float(np.mean([finite(row["tactile_diff_mae"]) for row in subset])) if subset else None,
        "tactile_ssim": float(np.mean([finite(row["tactile_ssim"]) for row in subset])) if subset else None,
        "tactile_mask_iou": float(np.mean([finite(row["tactile_mask_iou"]) for row in subset])) if subset else None,
        "oracle_top1": float(np.mean([int(row["ranker_oracle_embedding_rank"]) == 1 for row in subset])) if subset else None,
        "oracle_top3": float(np.mean([int(row["ranker_oracle_embedding_rank"]) <= 3 for row in subset])) if subset else None,
    }


def passes_point_guard(v1_rows: list[dict[str, str]], selected: list[dict[str, str]]) -> bool:
    for predicate in (lambda _: True, lambda row: int(row["query_probe"]) >= 75):
        base, current = metric_summary(v1_rows, predicate), metric_summary(selected, predicate)
        if not (
            current["tactile_diff_mae"] < base["tactile_diff_mae"]
            and current["tactile_ssim"] >= base["tactile_ssim"]
            and current["tactile_mask_iou"] >= base["tactile_mask_iou"]
            and current["oracle_top1"] >= base["oracle_top1"]
        ):
            return False
    return True


def choose_threshold(
    probabilities: np.ndarray,
    targets: np.ndarray,
    v1_rows: list[dict[str, str]],
    dino_rows: list[dict[str, str]],
    minimum_coverage: float,
    minimum_precision: float,
) -> dict:
    options = []
    thresholds = np.unique(np.quantile(probabilities, np.linspace(0.0, 1.0, 201)))
    for threshold in sorted((float(value) for value in thresholds), reverse=True):
        accepted = probabilities >= threshold
        coverage = float(accepted.mean())
        if coverage < minimum_coverage:
            continue
        precision = float(targets[accepted].mean()) if accepted.any() else 0.0
        guard = passes_point_guard(v1_rows, selected_rows(v1_rows, dino_rows, accepted))
        options.append(
            {
                "threshold": threshold,
                "coverage": coverage,
                "accepted_queries": int(accepted.sum()),
                "strict_triple_win_precision": precision,
                "passes_point_guard": guard,
            }
        )
    valid = [
        item for item in options
        if item["strict_triple_win_precision"] >= minimum_precision and item["passes_point_guard"]
    ]
    if not valid:
        return {"enabled": False, "options": options}
    chosen = max(valid, key=lambda item: (item["coverage"], item["strict_triple_win_precision"]))
    return {"enabled": True, "selected": chosen, "options": options}


def train(config_path: str, section: str) -> dict:
    cfg = load_config(config_path)[section]
    set_seed(int(cfg["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples = read_csv_rows(project_path(cfg["samples_csv"]))
    assert_development_only(samples, project_path(cfg["final_partition_csv"]))
    source_by_name = {row["image_name"]: row for row in samples if row["dataset_split"] == "train"}
    v1_rows = read_csv_rows(project_path(cfg["v1_query_csv"]))
    dino_rows = read_csv_rows(project_path(cfg["aligned_query_csv"]))
    if {row["query_image_name"] for row in v1_rows} != {row["query_image_name"] for row in dino_rows}:
        raise RuntimeError("V1 and aligned-DINO query sets differ")
    order = {row["query_image_name"]: index for index, row in enumerate(v1_rows)}
    dino_rows.sort(key=lambda row: order[row["query_image_name"]])
    if any(row["query_image_name"] not in source_by_name for row in dino_rows):
        raise RuntimeError("Gate inputs must be strict development-train OOF rows")
    v1_by_name = {row["query_image_name"]: row for row in v1_rows}
    v1_candidate_groups = grouped(read_csv_rows(project_path(cfg["v1_candidate_csv"])))
    dino_candidate_groups = grouped(read_csv_rows(project_path(cfg["aligned_candidate_csv"])))
    if set(v1_candidate_groups) != set(dino_candidate_groups):
        raise RuntimeError("V1 and aligned-DINO candidate query sets differ")
    for name in v1_candidate_groups:
        if {row["candidate_image_name"] for row in v1_candidate_groups[name]} != {
            row["candidate_image_name"] for row in dino_candidate_groups[name]
        }:
            raise RuntimeError(f"V1 and aligned-DINO Top-32 differ for {name}")
    raw_features = make_features(dino_rows, v1_by_name, v1_candidate_groups, dino_candidate_groups)
    targets = labels(v1_rows, dino_rows)
    folds = sorted({row["oof_fold"] for row in dino_rows})
    logits = np.zeros(len(dino_rows), dtype=np.float32)
    checkpoint_dir = project_path(cfg["checkpoint_dir"])
    ensure_dir(checkpoint_dir)
    reports = []
    for fold in folds:
        held_out = np.asarray([i for i, row in enumerate(dino_rows) if row["oof_fold"] == fold], dtype=np.int32)
        outer_fit = np.asarray([i for i, row in enumerate(dino_rows) if row["oof_fold"] != fold], dtype=np.int32)
        inner_validation = np.asarray(
            [
                i for i in outer_fit
                if record_hash_split(
                    dino_rows[int(i)]["query_record_id"],
                    float(cfg["inner_validation_fraction"]),
                    int(cfg["inner_split_seed"]) + int(fold),
                )
            ],
            dtype=np.int32,
        )
        inner_validation_set = set(inner_validation.tolist())
        inner_fit = np.asarray([i for i in outer_fit if i not in inner_validation_set], dtype=np.int32)
        mean, std = raw_features[inner_fit].mean(axis=0), raw_features[inner_fit].std(axis=0)
        std[std < 1e-6] = 1.0
        features = standardize(raw_features, mean, std)
        model, report = train_model(
            features, targets, inner_fit, inner_validation, cfg, device,
            checkpoint_dir / f"fold_{fold}.pt",
            {"scope": "strict_oof", "fold": fold, "feature_mean": mean, "feature_std": std},
        )
        logits[held_out] = logits_for(model, features, held_out, device)
        reports.append({"fold": fold, **report})

    temperature = calibrate_temperature(logits, targets)
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits / temperature, -60.0, 60.0)))
    gate = choose_threshold(
        probabilities,
        targets,
        v1_rows,
        dino_rows,
        float(cfg["minimum_coverage"]),
        float(cfg["minimum_precision"]),
    )
    threshold = float(gate["selected"]["threshold"]) if gate["enabled"] else None
    accepted = probabilities >= threshold if threshold is not None else np.zeros(len(dino_rows), dtype=bool)
    final_rows = selected_rows(v1_rows, dino_rows, accepted)

    full_validation = np.asarray(
        [
            i for i, row in enumerate(dino_rows)
            if record_hash_split(
                row["query_record_id"],
                float(cfg["inner_validation_fraction"]),
                int(cfg["inner_split_seed"]) + 1000,
            )
        ],
        dtype=np.int32,
    )
    full_validation_set = set(full_validation.tolist())
    full_fit = np.asarray([i for i in range(len(dino_rows)) if i not in full_validation_set], dtype=np.int32)
    full_mean, full_std = raw_features[full_fit].mean(axis=0), raw_features[full_fit].std(axis=0)
    full_std[full_std < 1e-6] = 1.0
    full_features = standardize(raw_features, full_mean, full_std)
    _, full_report = train_model(
        full_features, targets, full_fit, full_validation, cfg, device,
        checkpoint_dir / "full.pt",
        {
            "scope": "full_development_for_frozen_validation",
            "feature_mean": full_mean,
            "feature_std": full_std,
            "temperature": temperature,
            "threshold": threshold,
        },
    )
    reports.append({"fold": "full", **full_report})

    query_output = []
    for index, (v1, dino, final) in enumerate(zip(v1_rows, dino_rows, final_rows, strict=True)):
        use_dino = bool(accepted[index])
        query_output.append(
            {
                "query_record_id": dino["query_record_id"],
                "query_image_name": dino["query_image_name"],
                "query_probe": dino["query_probe"],
                "oof_fold": dino["oof_fold"],
                "dino_accept_probability": f"{probabilities[index]:.6f}",
                "gate_threshold": "" if threshold is None else f"{threshold:.6f}",
                "dino_accepted": str(int(use_dino)),
                "final_selection_source": "aligned_dino" if use_dino else "v1",
                "selected_cache_record_id": final["selected_cache_record_id"],
                "selected_cache_image_name": final["selected_cache_image_name"],
                "ranker_oracle_embedding_rank": final["ranker_oracle_embedding_rank"],
                **{metric: final[metric] for metric in METRICS},
                "v1_selected_cache_image_name": v1["selected_cache_image_name"],
                "dino_selected_cache_image_name": dino["selected_cache_image_name"],
                **{f"v1_{metric}": v1[metric] for metric in METRICS},
                **{f"dino_{metric}": dino[metric] for metric in METRICS},
                "strict_triple_win_label": str(int(targets[index])),
            }
        )
    candidate_output = []
    for index, query in enumerate(dino_rows):
        name = query["query_image_name"]
        source = dino_candidate_groups[name] if accepted[index] else v1_candidate_groups[name]
        for rank, row in enumerate(source, start=1):
            dino_source = next(
                item for item in dino_candidate_groups[name]
                if item["candidate_image_name"] == row["candidate_image_name"]
            )
            candidate_output.append(
                {
                    "query_record_id": query["query_record_id"],
                    "query_image_name": name,
                    "query_probe": query["query_probe"],
                    "oof_fold": query["oof_fold"],
                    "candidate_rank": str(rank),
                    "candidate_score": row["candidate_score"],
                    "candidate_record_id": row["candidate_record_id"],
                    "candidate_image_name": row["candidate_image_name"],
                    **{
                        field: dino_source.get(field, "")
                        for field in (
                            "predicted_tactile_latent_distance",
                            "candidate_tactile_latent_distance",
                            "detail_patch_score",
                            "context_patch_score",
                            "wide_patch_score",
                            "position_aware_match_score",
                            "hard_negative_flag",
                            "candidate_oracle_embedding_rank",
                        )
                    },
                    "dino_accept_probability": f"{probabilities[index]:.6f}",
                    "final_selection_source": "aligned_dino" if accepted[index] else "v1",
                }
            )
    write_csv_rows(project_path(cfg["query_output_csv"]), query_output, OUTPUT_QUERY_FIELDS)
    write_csv_rows(project_path(cfg["candidate_output_csv"]), candidate_output, OUTPUT_CANDIDATE_FIELDS)
    gate_json = {
        "mode": "phase4h_query_level_safety_gate_v1",
        "enabled": bool(gate["enabled"]),
        "temperature": temperature,
        "threshold": threshold,
        "feature_names": feature_names(),
        "selection": gate,
        "oof_positive_labels": int(targets.sum()),
        "oof_queries": len(targets),
    }
    write_json(project_path(cfg["gate_json"]), gate_json)
    summary = {
        "mode": "phase4h_strict_oof_query_level_gated_v1_dino",
        "device": str(device),
        "gate": gate_json,
        "v1": {
            "all": metric_summary(v1_rows),
            "far_probe75_100": metric_summary(v1_rows, lambda row: int(row["query_probe"]) >= 75),
        },
        "aligned_dino": {
            "all": metric_summary(dino_rows),
            "far_probe75_100": metric_summary(dino_rows, lambda row: int(row["query_probe"]) >= 75),
        },
        "gated": {
            "all": metric_summary(query_output),
            "far_probe75_100": metric_summary(query_output, lambda row: int(row["query_probe"]) >= 75),
        },
        "training": reports,
        "integrity": {
            "query_tactile_input": False,
            "gate_features": "online-only",
            "gate_probabilities": "strict outer-fold OOF",
            "sealed_final_holdout_rows_read": 0,
        },
    }
    write_json(project_path(cfg["metrics_json"]), summary)
    print({key: value for key, value in summary.items() if key != "training"})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the strict-OOF Phase4H query-level safety gate.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--section", default="phase4h_dino_gate_oof_v1")
    args = parser.parse_args()
    train(args.config, args.section)


if __name__ == "__main__":
    main()
