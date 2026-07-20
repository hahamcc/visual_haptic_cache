from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from torch import nn

from .build_cache_retrieval import crop_contact_patch, motion_geometry_feature, standardize, visual_patch_feature_from_patch
from .build_phase4e_oof_multiscale_cache import (
    CANDIDATE_FIELDS,
    QUERY_FIELDS,
    MultiScaleTactileRanker,
    build_shortlists,
    normalized_features,
    score_model,
)
from .config import load_config, project_path
from .evaluate_oracle_tactile_retrieval import tactile_difference, tactile_embedding, tactile_metrics
from .train_phase4b_predicted_box_cache_ranker import is_final_holdout, prediction_map, set_seed
from .train_phase4e_cache_trust import (
    CacheTrustPredictor,
    OUTPUT_FIELDS,
    candidate_features,
    feature_names,
    make_matrix,
    metric_summary,
    rejection_reasons,
    trust_scores,
)
from .train_phase4e_far_cache_gate import FarCacheGate
from .train_soft_tactile_cache_ranker import image_tensor, ranks
from .utils import ensure_dir, read_csv_rows, write_csv_rows, write_json


COMBINED_TRUST_FIELDS = [*OUTPUT_FIELDS, "gate_source", "far_quality_probability"]


def mean_metrics(rows: list[dict[str, str]], prefix: str) -> dict:
    metrics = ("tactile_diff_mae", "tactile_ssim", "tactile_mask_iou")
    return {
        "queries": len(rows),
        **{metric: float(np.mean([float(row[f"{prefix}_{metric}"]) for row in rows])) if rows else None for metric in metrics},
    }


def train_and_predict(cfg: dict, device: torch.device) -> tuple[list[dict[str, str]], list[dict[str, str]], dict]:
    rows = read_csv_rows(project_path(cfg["samples_csv"]))
    if any(is_final_holdout(row) for row in rows):
        raise RuntimeError("Refusing to access sealed final-holdout samples.")
    train_rows = [row for row in rows if row["dataset_split"] == "train"]
    validation_rows = [row for row in rows if row["dataset_split"] == "val"]
    if not train_rows or not validation_rows:
        raise RuntimeError("Expected non-empty development train and validation rows.")
    if {row["record_id"] for row in train_rows} & {row["record_id"] for row in validation_rows}:
        raise RuntimeError("Cache train and validation query records overlap.")

    oof_predictions = read_csv_rows(project_path(cfg["train_oof_predictions_csv"]))
    validation_predictions = read_csv_rows(project_path(cfg["validation_predictions_csv"]))
    train_prediction_by_name = prediction_map(oof_predictions, train_rows, "train", "Phase4E train OOF")
    validation_prediction_by_name = prediction_map(validation_predictions, validation_rows, "val", "Phase4E validation C2")
    detail_size, context_size = int(cfg["detail_crop_size"]), int(cfg["context_crop_size"])
    filter_k, batch_size = min(int(cfg["geometry_filter_k"]), len(train_rows)), int(cfg["batch_size"])
    tactile_size, threshold = int(cfg["tactile_size"]), float(cfg["tactile_mask_threshold"])

    train_gt_coordinates = [(float(row["target_tip_x"]), float(row["target_tip_y"])) for row in train_rows]
    raw_geometry = np.stack([motion_geometry_feature(row, x, y) for row, (x, y) in zip(train_rows, train_gt_coordinates, strict=True)])
    _, geometry_mean, geometry_std = standardize(raw_geometry, raw_geometry)
    raw_detail = np.stack([visual_patch_feature_from_patch(crop_contact_patch(row["vision_path"], x, y, detail_size)) for row, (x, y) in zip(train_rows, train_gt_coordinates, strict=True)])
    raw_context = np.stack([visual_patch_feature_from_patch(crop_contact_patch(row["vision_path"], x, y, context_size)) for row, (x, y) in zip(train_rows, train_gt_coordinates, strict=True)])
    _, detail_mean, detail_std = standardize(raw_detail, raw_detail)
    _, context_mean, context_std = standardize(raw_context, raw_context)
    cache_detail, cache_context, cache_geometry, cache_detail_hand, cache_context_hand = normalized_features(
        train_rows, train_gt_coordinates, geometry_mean, geometry_std, detail_mean, detail_std, context_mean, context_std, detail_size, context_size,
    )
    train_coordinates = [(float(train_prediction_by_name[row["image_name"]]["pred_x"]), float(train_prediction_by_name[row["image_name"]]["pred_y"])) for row in train_rows]
    validation_coordinates = [(float(validation_prediction_by_name[row["image_name"]]["pred_x"]), float(validation_prediction_by_name[row["image_name"]]["pred_y"])) for row in validation_rows]
    train_detail, train_context, train_geometry, train_detail_hand, train_context_hand = normalized_features(
        train_rows, train_coordinates, geometry_mean, geometry_std, detail_mean, detail_std, context_mean, context_std, detail_size, context_size,
    )
    validation_detail, validation_context, validation_geometry, validation_detail_hand, validation_context_hand = normalized_features(
        validation_rows, validation_coordinates, geometry_mean, geometry_std, detail_mean, detail_std, context_mean, context_std, detail_size, context_size,
    )
    tactile_cache: dict[str, np.ndarray] = {}

    def touch(row: dict[str, str]) -> np.ndarray:
        return tactile_difference(row["touch_path"], tactile_cache, tactile_size)

    cache_embeddings = np.stack([tactile_embedding(touch(row)) for row in train_rows]).astype(np.float32)
    train_embeddings = np.stack([tactile_embedding(touch(row)) for row in train_rows]).astype(np.float32)
    validation_embeddings = np.stack([tactile_embedding(touch(row)) for row in validation_rows]).astype(np.float32)
    train_groups = build_shortlists(
        train_rows, train_geometry, train_detail_hand, train_context_hand, train_embeddings,
        train_rows, cache_geometry, cache_detail_hand, cache_context_hand, cache_embeddings, filter_k, True,
    )
    validation_groups = build_shortlists(
        validation_rows, validation_geometry, validation_detail_hand, validation_context_hand, validation_embeddings,
        train_rows, cache_geometry, cache_detail_hand, cache_context_hand, cache_embeddings, filter_k, False,
    )
    train_candidates, train_targets, train_hand, *_ = train_groups
    candidates, targets, hand, geometry_values, detail_values, context_values = validation_groups
    model = MultiScaleTactileRanker(cache_geometry.shape[1], float(cfg["dropout"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]))
    target_std = max(float(train_targets.std()), 1e-6)
    for _ in range(int(cfg["epochs"])):
        model.train()
        order = np.random.permutation(len(train_rows))
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            local_candidates = train_candidates[indices]
            batch, local_k = local_candidates.shape
            scores = model(
                image_tensor(train_detail[indices]).to(device), image_tensor(cache_detail[local_candidates].reshape(-1, *cache_detail.shape[1:])).reshape(batch, local_k, 3, *cache_detail.shape[1:3]).to(device),
                image_tensor(train_context[indices]).to(device), image_tensor(cache_context[local_candidates].reshape(-1, *cache_context.shape[1:])).reshape(batch, local_k, 3, *cache_context.shape[1:3]).to(device),
                torch.from_numpy(train_geometry[indices]).to(device), torch.from_numpy(cache_geometry[local_candidates]).to(device), torch.from_numpy(train_hand[indices]).to(device),
            )
            target = torch.from_numpy(train_targets[indices]).to(device)
            regression = nn.functional.smooth_l1_loss((scores - target) / target_std, torch.zeros_like(scores))
            distribution = torch.softmax(-target / float(cfg["target_temperature"]), dim=1)
            listwise = -(distribution * torch.log_softmax(-scores / float(cfg["target_temperature"]), dim=1)).sum(dim=1).mean()
            optimizer.zero_grad(set_to_none=True)
            (regression + float(cfg["listwise_weight"]) * listwise).backward()
            optimizer.step()
    checkpoint_dir = project_path(cfg["checkpoint_dir"])
    ensure_dir(checkpoint_dir)
    torch.save({"model_state": model.state_dict(), "config_section": cfg["_section"], "epochs": int(cfg["epochs"]), "training_queries": len(train_rows)}, checkpoint_dir / "ranker.pt")
    scores = score_model(model, candidates, hand, validation_detail, validation_context, validation_geometry, cache_detail, cache_context, cache_geometry, device, batch_size)

    query_output, candidate_output, metric_cache = [], [], {}
    for index, query in enumerate(validation_rows):
        order = np.argsort(scores[index], kind="stable")
        hand_order = np.argsort(hand[index], kind="stable")
        selected = int(order[0])
        top3 = order[:3]
        cache = train_rows[int(candidates[index, selected])]
        tactile = tactile_metrics(touch(query), tactile_difference(cache["touch_path"], metric_cache, tactile_size), threshold)
        top3_embeddings = cache_embeddings[candidates[index, top3]]
        prediction = validation_prediction_by_name[query["image_name"]]
        best, second = float(scores[index, order[0]]), float(scores[index, order[1]])
        hand_best, hand_second = float(hand[index, hand_order[0]]), float(hand[index, hand_order[1]])
        query_output.append({
            "query_record_id": query["record_id"], "query_image_name": query["image_name"], "query_probe": query["probe"], "oof_fold": "validation",
            "pred_x": f"{validation_coordinates[index][0]:.3f}", "pred_y": f"{validation_coordinates[index][1]:.3f}", "c2_pred_score": prediction.get("pred_score", ""),
            "ranker_best_score": f"{best:.6f}", "ranker_second_score": f"{second:.6f}", "ranker_margin": f"{second - best:.6f}", "ranker_margin_normalized": f"{(second - best) / max(float(scores[index].std()), 1e-6):.6f}",
            "hand_best_score": f"{hand_best:.6f}", "hand_second_score": f"{hand_second:.6f}", "hand_margin": f"{hand_second - hand_best:.6f}",
            "ranker_oracle_embedding_rank": str(int(ranks(scores[index])[int(np.argmin(targets[index]))])),
            "selected_cache_record_id": cache["record_id"], "selected_cache_image_name": cache["image_name"],
            "top3_cache_record_ids": "|".join(train_rows[int(candidates[index, item])]["record_id"] for item in top3),
            "top3_cache_image_names": "|".join(train_rows[int(candidates[index, item])]["image_name"] for item in top3),
            "top3_tactile_embedding_disagreement": f"{float(np.mean(np.linalg.norm(top3_embeddings[:, None] - top3_embeddings[None, :], axis=2))):.6f}",
            "top3_score_std": f"{float(scores[index, top3].std()):.6f}", "geometry_distance": f"{float(geometry_values[index, selected]):.6f}",
            "detail_visual_distance": f"{float(detail_values[index, selected]):.6f}", "context_visual_distance": f"{float(context_values[index, selected]):.6f}",
            "trajectory_real_point_count": query["trajectory_real_point_count"], "trajectory_history_span_frames": query["trajectory_history_span_frames"],
            "trajectory_padding_ratio": query["trajectory_padding_ratio"], "trajectory_cumulative_displacement": query["trajectory_cumulative_displacement"],
            **{key: f"{value:.6f}" for key, value in tactile.items()},
        })
        for rank, item in enumerate(order, start=1):
            candidate = train_rows[int(candidates[index, item])]
            candidate_output.append({
                "query_record_id": query["record_id"], "query_image_name": query["image_name"], "query_probe": query["probe"], "oof_fold": "validation",
                "candidate_rank": str(rank), "candidate_score": f"{float(scores[index, item]):.6f}", "hand_score": f"{float(hand[index, item]):.6f}",
                "geometry_distance": f"{float(geometry_values[index, item]):.6f}", "detail_visual_distance": f"{float(detail_values[index, item]):.6f}", "context_visual_distance": f"{float(context_values[index, item]):.6f}",
                "candidate_record_id": candidate["record_id"], "candidate_image_name": candidate["image_name"], "candidate_tactile_embedding_distance": f"{float(targets[index, item]):.6f}",
            })
    return query_output, candidate_output, {"train_queries": len(train_rows), "validation_queries": len(validation_rows), "train_records": len({row['record_id'] for row in train_rows}), "validation_records": len({row['record_id'] for row in validation_rows})}


def apply_frozen_trust(cfg: dict, query_rows: list[dict[str, str]], candidate_rows: list[dict[str, str]], device: torch.device) -> tuple[list[dict[str, str]], dict]:
    checkpoint = torch.load(project_path(cfg["trust_checkpoint"]), map_location=device, weights_only=False)
    expected_names = feature_names()
    if checkpoint["feature_names"] != expected_names:
        raise RuntimeError("Frozen trust checkpoint feature schema differs from the current evaluation schema.")
    aggregate = candidate_features(candidate_rows)
    features = make_matrix(query_rows, aggregate)
    features = (features - checkpoint["feature_mean"]) / np.maximum(checkpoint["feature_std"], 1e-6)
    model = CacheTrustPredictor(features.shape[1], float(cfg["trust_dropout"])).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    with torch.no_grad():
        predicted = model(torch.from_numpy(features.astype(np.float32)).to(device)).cpu().numpy()
    predicted = predicted * checkpoint["target_std"] + checkpoint["target_mean"]
    scores = trust_scores(predicted, checkpoint["target_mean"], checkpoint["target_std"])
    gate_config = json.loads(project_path(cfg["trust_gate_json"]).read_text())
    if not gate_config.get("enabled") or gate_config.get("threshold") is None:
        raise RuntimeError("Frozen trust gate is disabled or lacks a threshold.")
    threshold = float(gate_config["threshold"])
    accepted = scores >= threshold
    output = []
    for index, row in enumerate(query_rows):
        output.append({
            "query_record_id": row["query_record_id"], "query_image_name": row["query_image_name"], "query_probe": row["query_probe"], "oof_fold": "validation",
            "status": "cache_hit" if accepted[index] else "cache_miss", "trust_score": f"{scores[index]:.6f}", "gate_threshold": f"{threshold:.6f}",
            "predicted_tactile_diff_mae": f"{predicted[index, 0]:.6f}", "predicted_tactile_ssim": f"{predicted[index, 1]:.6f}", "predicted_tactile_mask_iou": f"{predicted[index, 2]:.6f}",
            "selected_cache_record_id": row["selected_cache_record_id"], "top3_cache_record_ids": row["top3_cache_record_ids"],
            "rejection_reasons": rejection_reasons(predicted[index], checkpoint["target_mean"], checkpoint["target_std"], scores[index], threshold),
            **{field: row[field] for field in ("ranker_best_score", "ranker_margin_normalized", "top3_tactile_embedding_disagreement", "c2_pred_score", "ranker_oracle_embedding_rank", "tactile_diff_mae", "tactile_ssim", "tactile_mask_iou")},
        })
    baseline = np.asarray([np.mean([float(row[field]) for row in query_rows]) for field in ("tactile_diff_mae", "tactile_ssim", "tactile_mask_iou")], dtype=np.float32)
    metrics = {
        "all": metric_summary(query_rows, accepted, baseline),
        "far_probe75_100": metric_summary([row for row in query_rows if int(row["query_probe"]) >= 75], accepted[np.asarray([int(row["query_probe"]) >= 75 for row in query_rows])], baseline),
        "near_mid": metric_summary([row for row in query_rows if int(row["query_probe"]) < 75], accepted[np.asarray([int(row["query_probe"]) < 75 for row in query_rows])], baseline),
    }
    return output, {"threshold": threshold, "validation_metrics": metrics, "trust_score_mean": float(scores.mean()), "trust_score_std": float(scores.std())}


def apply_frozen_far_gate(
    cfg: dict, query_rows: list[dict[str, str]], candidate_rows: list[dict[str, str]], unified_rows: list[dict[str, str]], device: torch.device,
) -> tuple[list[dict[str, str]], dict]:
    """Use the conservative far-only gate without changing the selected contact box or cache entry."""
    if len(query_rows) != len(unified_rows):
        raise RuntimeError("Unified trust rows must align one-to-one with validation queries.")
    far_indices = np.asarray([index for index, row in enumerate(query_rows) if int(row["query_probe"]) >= 75], dtype=np.int32)
    if not len(far_indices):
        raise RuntimeError("Validation set has no far query for the far-gate evaluation.")
    checkpoint = torch.load(project_path(cfg["far_gate_checkpoint"]), map_location=device, weights_only=False)
    if checkpoint["feature_names"] != feature_names():
        raise RuntimeError("Frozen far-gate feature schema differs from the evaluation schema.")
    aggregate = candidate_features(candidate_rows)
    far_rows = [query_rows[index] for index in far_indices]
    features = make_matrix(far_rows, aggregate)
    features = (features - checkpoint["feature_mean"]) / np.maximum(checkpoint["feature_std"], 1e-6)
    model = FarCacheGate(features.shape[1], float(cfg["far_gate_dropout"])).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(model(torch.from_numpy(features.astype(np.float32)).to(device))).cpu().numpy()
    far_gate = json.loads(project_path(cfg["far_gate_json"]).read_text())
    if not far_gate.get("enabled") or far_gate.get("threshold") is None:
        raise RuntimeError("Frozen far gate is disabled or lacks a threshold.")
    threshold = float(far_gate["threshold"])
    probability_by_name = {row["query_image_name"]: float(probability) for row, probability in zip(far_rows, probabilities, strict=True)}
    output = []
    for query, unified in zip(query_rows, unified_rows, strict=True):
        row = dict(unified)
        if int(query["query_probe"]) >= 75:
            probability = probability_by_name[query["query_image_name"]]
            row["status"] = "cache_hit" if probability >= threshold else "cache_miss"
            row["gate_threshold"] = f"{threshold:.6f}"
            row["rejection_reasons"] = "" if row["status"] == "cache_hit" else "far_gate_low_quality_probability"
            row["gate_source"] = "far_gate"
            row["far_quality_probability"] = f"{probability:.6f}"
        else:
            row["gate_source"] = "unified_gate"
            row["far_quality_probability"] = ""
        output.append(row)
    accepted = np.asarray([row["status"] == "cache_hit" for row in output], dtype=bool)
    all_baseline = np.asarray([np.mean([float(row[field]) for row in query_rows]) for field in ("tactile_diff_mae", "tactile_ssim", "tactile_mask_iou")], dtype=np.float32)
    far_mask = np.asarray([int(row["query_probe"]) >= 75 for row in query_rows])
    near_mid_mask = ~far_mask
    far_rows = [row for row in query_rows if int(row["query_probe"]) >= 75]
    near_mid_rows = [row for row in query_rows if int(row["query_probe"]) < 75]
    far_baseline = np.asarray([np.mean([float(row[field]) for row in far_rows]) for field in ("tactile_diff_mae", "tactile_ssim", "tactile_mask_iou")], dtype=np.float32)
    near_mid_baseline = np.asarray([np.mean([float(row[field]) for row in near_mid_rows]) for field in ("tactile_diff_mae", "tactile_ssim", "tactile_mask_iou")], dtype=np.float32)
    metrics = {
        "all": metric_summary(query_rows, accepted, all_baseline),
        "far_probe75_100": metric_summary(far_rows, accepted[far_mask], far_baseline),
        "near_mid": metric_summary(near_mid_rows, accepted[near_mid_mask], near_mid_baseline),
    }
    return output, {
        "unified_gate": "applied only to near/mid", "far_gate_threshold": threshold,
        "far_probability_mean": float(probabilities.mean()), "far_probability_std": float(probabilities.std()), "validation_metrics": metrics,
    }


def evaluate(config_path: str, section: str) -> dict:
    cfg = dict(load_config(config_path)[section])
    cfg["_section"] = section
    set_seed(int(cfg["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    query_rows, candidate_rows, split_summary = train_and_predict(cfg, device)
    unified_rows, unified_summary = apply_frozen_trust(cfg, query_rows, candidate_rows, device)
    if "far_gate_checkpoint" in cfg:
        trust_rows, trust_summary = apply_frozen_far_gate(cfg, query_rows, candidate_rows, unified_rows, device)
        trust_output_fields = COMBINED_TRUST_FIELDS
    else:
        trust_rows, trust_summary = unified_rows, unified_summary
        trust_output_fields = OUTPUT_FIELDS
    write_csv_rows(project_path(cfg["query_output_csv"]), query_rows, QUERY_FIELDS)
    write_csv_rows(project_path(cfg["candidate_output_csv"]), candidate_rows, CANDIDATE_FIELDS)
    write_csv_rows(project_path(cfg["trust_output_csv"]), trust_rows, trust_output_fields)
    phase4b_rows = read_csv_rows(project_path(cfg["phase4b_validation_csv"]))
    phase4b_by_name = {row["query_image_name"]: row for row in phase4b_rows}
    matching_phase4b = [phase4b_by_name[row["query_image_name"]] for row in query_rows if row["query_image_name"] in phase4b_by_name]
    summary = {
        "mode": "phase4e_frozen_multiscale_ranker_and_trust_validation", "device": str(device), "split": split_summary,
        "ranker": {"epochs": int(cfg["epochs"]), "checkpoint": str(project_path(cfg["checkpoint_dir"]) / "ranker.pt"), "contact_boxes": "frozen C2 refit validation predictions"},
        "trust": trust_summary, "unified_gate_reference": unified_summary,
        "phase4b_reference": {"current_hand_key": mean_metrics(matching_phase4b, "current"), "single_scale_ranker": mean_metrics(matching_phase4b, "ranker")},
        "integrity": {
            "c2": "frozen; cache evaluation does not modify predicted contact boxes", "ranker_train_boxes": "strict C2 OOF", "ranker_validation_boxes": "C2 refit validation",
            "trust": "frozen checkpoint and threshold; no threshold selection on validation", "same_record_cache_for_validation": "impossible because record splits are disjoint", "sealed_final_holdout_rows_read": 0,
        },
    }
    write_json(project_path(cfg["metrics_json"]), summary)
    print(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Independently validate the frozen Phase4E multi-scale cache trust recipe.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--section", default="phase4e_multiscale_trust_validation_v2")
    args = parser.parse_args()
    evaluate(args.config, args.section)


if __name__ == "__main__":
    main()
