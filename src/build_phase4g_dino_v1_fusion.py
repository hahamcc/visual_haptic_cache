"""Phase4G: strict convex fusion of V1 cache scores and frozen DINO patch scores.

The module never uses tactile data to form an online score.  Tactile images are
read only for offline listwise supervision, metric reporting, and bootstrap
evaluation on development data.
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import torch

from .build_cache_retrieval import crop_contact_patch, motion_geometry_feature, standardize, visual_patch_feature_from_patch
from .build_phase4e_oof_multiscale_cache import CANDIDATE_FIELDS as V1_CANDIDATE_FIELDS
from .build_phase4e_oof_multiscale_cache import QUERY_FIELDS as V1_QUERY_FIELDS
from .build_phase4e_oof_multiscale_cache import TACTILE_METRICS, build_shortlists, normalized_features
from .build_phase4f_dino_cross_attention_cache import embedding_matrix, encode, tactile_targets
from .build_phase4f_dino_patch_similarity_cache import patch_scores
from .config import load_config, project_path
from .evaluate_oracle_tactile_retrieval import tactile_difference, tactile_embedding, tactile_metrics
from .evaluate_phase4e_multiscale_trust_validation import train_and_predict
from .phase4f_dino_cross_attention import FrozenDinoV2
from .train_phase4b_predicted_box_cache_ranker import is_final_holdout, prediction_map, set_seed
from .train_soft_tactile_cache_ranker import ranks
from .utils import read_csv_rows, write_csv_rows, write_json


FUSION_CANDIDATE_FIELDS = [
    *V1_CANDIDATE_FIELDS, "v1_score", "v1_score_z", "dino_detail_patch_similarity",
    "dino_context_patch_similarity", "dino_detail_distance_z", "dino_context_distance_z", "fusion_score",
]
WEIGHT_FIELDS = ["scope", "oof_fold", "v1_weight", "dino_detail_weight", "dino_context_weight", "training_queries"]


def finite(value: str | float | int, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def grouped(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    output: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        output[row["query_image_name"]].append(row)
    return output


def merge_candidates(v1_rows: list[dict[str, str]], dino_rows: list[dict[str, str]], expected_k: int) -> dict[str, list[dict[str, object]]]:
    """Join candidate tables by image identity and reject protocol drift early."""
    v1_by_query, dino_by_query = grouped(v1_rows), grouped(dino_rows)
    if set(v1_by_query) != set(dino_by_query):
        raise RuntimeError("V1 and DINO candidate query sets differ; refusing fusion.")
    output: dict[str, list[dict[str, object]]] = {}
    for query_name in sorted(v1_by_query):
        v1_items, dino_items = v1_by_query[query_name], dino_by_query[query_name]
        v1_by_candidate = {item["candidate_image_name"]: item for item in v1_items}
        dino_by_candidate = {item["candidate_image_name"]: item for item in dino_items}
        if len(v1_by_candidate) != expected_k or len(dino_by_candidate) != expected_k:
            raise RuntimeError(f"{query_name}: expected exactly {expected_k} unique candidates.")
        if set(v1_by_candidate) != set(dino_by_candidate):
            raise RuntimeError(f"{query_name}: V1 and DINO Top-{expected_k} candidate sets differ.")
        merged = []
        for name in sorted(v1_by_candidate):
            v1, dino = v1_by_candidate[name], dino_by_candidate[name]
            if v1["candidate_record_id"] != dino["candidate_record_id"]:
                raise RuntimeError(f"{query_name}/{name}: candidate record identity differs.")
            if abs(finite(v1["candidate_tactile_embedding_distance"]) - finite(dino["candidate_tactile_embedding_distance"])) > 1e-5:
                raise RuntimeError(f"{query_name}/{name}: offline tactile target differs.")
            merged.append({"v1": v1, "dino": dino})
        output[query_name] = merged
    return output


def normalize_per_query(values: np.ndarray) -> np.ndarray:
    return (values - values.mean()) / max(float(values.std()), 1e-6)


def group_arrays(items: list[dict[str, object]]) -> tuple[np.ndarray, np.ndarray]:
    v1 = np.asarray([finite(item["v1"]["candidate_score"]) for item in items], dtype=np.float32)
    detail_distance = -np.asarray([finite(item["dino"]["detail_patch_similarity"]) for item in items], dtype=np.float32)
    context_distance = -np.asarray([finite(item["dino"]["context_patch_similarity"]) for item in items], dtype=np.float32)
    targets = np.asarray([finite(item["v1"]["candidate_tactile_embedding_distance"]) for item in items], dtype=np.float32)
    return np.stack((normalize_per_query(v1), normalize_per_query(detail_distance), normalize_per_query(context_distance)), axis=1), targets


def stack_groups(groups: dict[str, list[dict[str, object]]], query_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    features, targets = zip(*(group_arrays(groups[name]) for name in query_names), strict=True)
    return np.stack(features).astype(np.float32), np.stack(targets).astype(np.float32)


def fit_weights(features: np.ndarray, targets: np.ndarray, cfg: dict) -> np.ndarray:
    """Fit three non-negative convex weights with the existing soft listwise target."""
    x, y = torch.from_numpy(features), torch.from_numpy(targets)
    logits = torch.zeros(3, requires_grad=True)
    optimizer = torch.optim.Adam([logits], lr=float(cfg["fusion_learning_rate"]))
    target_temperature, score_temperature = float(cfg["target_temperature"]), float(cfg["fusion_score_temperature"])
    for _ in range(int(cfg["fusion_epochs"])):
        weights = torch.softmax(logits, dim=0)
        scores = (x * weights).sum(dim=2)
        target_distribution = torch.softmax(-y / target_temperature, dim=1)
        loss = -(target_distribution * torch.log_softmax(-scores / score_temperature, dim=1)).sum(dim=1).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return torch.softmax(logits.detach(), dim=0).cpu().numpy().astype(np.float32)


def score_groups(groups: dict[str, list[dict[str, object]]], query_names: list[str], weights: np.ndarray) -> dict[str, np.ndarray]:
    output = {}
    for name in query_names:
        features, _ = group_arrays(groups[name])
        output[name] = (features * weights[None]).sum(axis=1)
    return output


def build_outputs(
    groups: dict[str, list[dict[str, object]]], query_rows: list[dict[str, str]], scores: dict[str, np.ndarray],
    samples_by_name: dict[str, dict[str, str]], tactile_size: int, threshold: float,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    tactile_cache: dict[str, np.ndarray] = {}
    def touch(row: dict[str, str]) -> np.ndarray:
        return tactile_difference(row["touch_path"], tactile_cache, tactile_size)
    output_queries, output_candidates = [], []
    for query in query_rows:
        name, items, group_score = query["query_image_name"], groups[query["query_image_name"]], scores[query["query_image_name"]]
        order = np.argsort(group_score, kind="stable")
        selected_index, selected_item = int(order[0]), items[int(order[0])]
        selected_cache = samples_by_name[selected_item["v1"]["candidate_image_name"]]
        tactile = tactile_metrics(touch(samples_by_name[name]), touch(selected_cache), threshold)
        _, targets = group_arrays(items)
        top3 = order[:3]
        top3_embeddings = np.stack([tactile_embedding(touch(samples_by_name[items[int(index)]["v1"]["candidate_image_name"]])) for index in top3])
        hand = np.asarray([finite(item["v1"].get("hand_score", "")) for item in items], dtype=np.float32)
        hand_order = np.argsort(hand, kind="stable")
        selected_v1 = selected_item["v1"]
        row = dict(query)
        row.update({
            "selected_cache_record_id": selected_cache["record_id"], "selected_cache_image_name": selected_cache["image_name"],
            "ranker_best_score": f"{group_score[selected_index]:.6f}", "ranker_second_score": f"{group_score[int(order[1])]:.6f}",
            "ranker_margin": f"{group_score[int(order[1])] - group_score[selected_index]:.6f}",
            "ranker_margin_normalized": f"{(group_score[int(order[1])] - group_score[selected_index]) / max(float(group_score.std()), 1e-6):.6f}",
            "hand_best_score": f"{hand[int(hand_order[0])]:.6f}", "hand_second_score": f"{hand[int(hand_order[1])]:.6f}",
            "hand_margin": f"{hand[int(hand_order[1])] - hand[int(hand_order[0])]:.6f}",
            "ranker_oracle_embedding_rank": str(int(ranks(group_score)[int(np.argmin(targets))])),
            "top3_cache_record_ids": "|".join(items[int(index)]["v1"]["candidate_record_id"] for index in top3),
            "top3_cache_image_names": "|".join(items[int(index)]["v1"]["candidate_image_name"] for index in top3),
            "top3_tactile_embedding_disagreement": f"{float(np.mean(np.linalg.norm(top3_embeddings[:, None] - top3_embeddings[None, :], axis=2))):.6f}",
            "top3_score_std": f"{float(group_score[top3].std()):.6f}",
            "geometry_distance": selected_v1["geometry_distance"], "detail_visual_distance": selected_v1["detail_visual_distance"],
            "context_visual_distance": selected_v1["context_visual_distance"],
            **{metric: f"{tactile[metric]:.6f}" for metric in TACTILE_METRICS},
        })
        output_queries.append(row)
        features, _ = group_arrays(items)
        for rank, index in enumerate(order, start=1):
            v1, dino = items[int(index)]["v1"], items[int(index)]["dino"]
            candidate = dict(v1)
            candidate.update({
                "candidate_rank": str(rank), "candidate_score": f"{group_score[int(index)]:.6f}",
                "v1_score": v1["candidate_score"], "v1_score_z": f"{features[int(index), 0]:.6f}",
                "dino_detail_patch_similarity": dino["detail_patch_similarity"], "dino_context_patch_similarity": dino["context_patch_similarity"],
                "dino_detail_distance_z": f"{features[int(index), 1]:.6f}", "dino_context_distance_z": f"{features[int(index), 2]:.6f}",
                "fusion_score": f"{group_score[int(index)]:.6f}",
            })
            output_candidates.append(candidate)
    return output_queries, output_candidates


def dino_validation_outputs(cfg: dict, device: torch.device) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Score frozen DINO local correspondence on the fixed V4 validation protocol."""
    rows = read_csv_rows(project_path(cfg["samples_csv"]))
    if any(is_final_holdout(row) for row in rows):
        raise RuntimeError("Phase4G refuses sealed final-holdout samples.")
    train_rows = [row for row in rows if row["dataset_split"] == "train"]
    validation_rows = [row for row in rows if row["dataset_split"] == "val"]
    if {row["record_id"] for row in train_rows} & {row["record_id"] for row in validation_rows}:
        raise RuntimeError("Phase4G validation query and cache records overlap.")
    prediction_map(read_csv_rows(project_path(cfg["train_oof_predictions_csv"])), train_rows, "train", "Phase4G train OOF")
    validation_predictions = prediction_map(read_csv_rows(project_path(cfg["validation_predictions_csv"])), validation_rows, "val", "Phase4G validation C2")
    detail_size, context_size = int(cfg["detail_crop_size"]), int(cfg["context_crop_size"])
    train_gt = [(float(row["target_tip_x"]), float(row["target_tip_y"])) for row in train_rows]
    raw_geometry = np.stack([motion_geometry_feature(row, x, y) for row, (x, y) in zip(train_rows, train_gt, strict=True)])
    _, geometry_mean, geometry_std = standardize(raw_geometry, raw_geometry)
    raw_detail = np.stack([visual_patch_feature_from_patch(crop_contact_patch(row["vision_path"], x, y, detail_size)) for row, (x, y) in zip(train_rows, train_gt, strict=True)])
    raw_context = np.stack([visual_patch_feature_from_patch(crop_contact_patch(row["vision_path"], x, y, context_size)) for row, (x, y) in zip(train_rows, train_gt, strict=True)])
    _, detail_mean, detail_std = standardize(raw_detail, raw_detail)
    _, context_mean, context_std = standardize(raw_context, raw_context)
    cache_detail, cache_context, cache_geometry, cache_detail_hand, cache_context_hand = normalized_features(
        train_rows, train_gt, geometry_mean, geometry_std, detail_mean, detail_std, context_mean, context_std, detail_size, context_size,
    )
    validation_xy = [(float(validation_predictions[row["image_name"]]["pred_x"]), float(validation_predictions[row["image_name"]]["pred_y"])) for row in validation_rows]
    query_detail, query_context, query_geometry, query_detail_hand, query_context_hand = normalized_features(
        validation_rows, validation_xy, geometry_mean, geometry_std, detail_mean, detail_std, context_mean, context_std, detail_size, context_size,
    )
    tactile_cache: dict[str, np.ndarray] = {}
    def touch(row: dict[str, str]) -> np.ndarray:
        return tactile_difference(row["touch_path"], tactile_cache, int(cfg["tactile_size"]))
    cache_embeddings = embedding_matrix(train_rows, touch, "Phase4G validation cache")
    validation_embeddings = embedding_matrix(validation_rows, touch, "Phase4G validation query")
    candidates, targets, _, _, _, _ = build_shortlists(
        validation_rows, query_geometry, query_detail_hand, query_context_hand, validation_embeddings,
        train_rows, cache_geometry, cache_detail_hand, cache_context_hand, cache_embeddings,
        min(int(cfg["geometry_filter_k"]), len(train_rows)), False,
    )
    print(f"phase4g validation: loading frozen {cfg['dino_model']} and encoding visual patches", flush=True)
    backbone = FrozenDinoV2(str(cfg["dino_model"]), int(cfg["dino_image_size"])).to(device)
    cache_detail_tokens = encode(backbone, cache_detail, device, int(cfg["batch_size"]))
    cache_context_tokens = encode(backbone, cache_context, device, int(cfg["batch_size"]))
    query_detail_tokens = encode(backbone, query_detail, device, int(cfg["batch_size"]))
    query_context_tokens = encode(backbone, query_context, device, int(cfg["batch_size"]))
    detail_similarity, context_similarity, scores = patch_scores(
        query_detail_tokens, cache_detail_tokens, query_context_tokens, cache_context_tokens, candidates,
        float(cfg["detail_similarity_weight"]), int(cfg["local_match_radius"]), device, int(cfg["batch_size"]),
    )
    order = np.argsort(scores, axis=1, kind="stable")
    eval_ssim, eval_iou = tactile_targets(validation_rows, train_rows, candidates, touch, float(cfg["tactile_mask_threshold"]), "Phase4G validation")
    queries, candidate_rows = [], []
    for index, query in enumerate(validation_rows):
        choice = int(order[index, 0])
        selected = train_rows[int(candidates[index, choice])]
        metric = tactile_metrics(touch(query), touch(selected), float(cfg["tactile_mask_threshold"]))
        queries.append({
            "query_record_id": query["record_id"], "query_image_name": query["image_name"], "query_probe": query["probe"], "oof_fold": "validation",
            "pred_x": f"{validation_xy[index][0]:.3f}", "pred_y": f"{validation_xy[index][1]:.3f}",
            "selected_cache_record_id": selected["record_id"], "selected_cache_image_name": selected["image_name"],
            "ranker_best_score": f"{scores[index, choice]:.6f}", "ranker_margin": f"{scores[index, int(order[index, 1])] - scores[index, choice]:.6f}",
            "ranker_oracle_embedding_rank": str(int(ranks(scores[index])[int(np.argmin(targets[index]))])),
            **{metric_name: f"{metric[metric_name]:.6f}" for metric_name in ("tactile_diff_mae", "tactile_ssim", "tactile_mask_iou")},
        })
        for rank, item in enumerate(order[index], start=1):
            cache = train_rows[int(candidates[index, item])]
            candidate_rows.append({
                "query_record_id": query["record_id"], "query_image_name": query["image_name"], "query_probe": query["probe"], "oof_fold": "validation",
                "candidate_rank": str(rank), "candidate_score": f"{scores[index, item]:.6f}",
                "detail_patch_similarity": f"{detail_similarity[index, item]:.6f}", "context_patch_similarity": f"{context_similarity[index, item]:.6f}",
                "patch_similarity_score": f"{-scores[index, item]:.6f}", "candidate_record_id": cache["record_id"], "candidate_image_name": cache["image_name"],
                "candidate_tactile_embedding_distance": f"{targets[index, item]:.6f}", "candidate_tactile_ssim": f"{eval_ssim[index, item]:.6f}",
                "candidate_tactile_mask_iou": f"{eval_iou[index, item]:.6f}", "candidate_oracle_embedding_rank": str(int(ranks(targets[index])[item])),
            })
    return queries, candidate_rows


def v1_validation_outputs(cfg: dict, device: torch.device) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Rebuild the frozen V1 ranker under the exact independent validation protocol."""
    v1_cfg = {
        "_section": "phase4g_v1_validation_reference", "samples_csv": cfg["samples_csv"],
        "train_oof_predictions_csv": cfg["train_oof_predictions_csv"], "validation_predictions_csv": cfg["validation_predictions_csv"],
        "checkpoint_dir": cfg["v1_validation_checkpoint_dir"], "detail_crop_size": cfg["detail_crop_size"], "context_crop_size": cfg["context_crop_size"],
        "tactile_size": cfg["tactile_size"], "tactile_mask_threshold": cfg["tactile_mask_threshold"], "geometry_filter_k": cfg["geometry_filter_k"],
        "batch_size": cfg["v1_batch_size"], "epochs": cfg["v1_epochs"], "learning_rate": cfg["v1_learning_rate"], "weight_decay": cfg["v1_weight_decay"],
        "dropout": cfg["v1_dropout"], "target_temperature": cfg["target_temperature"], "listwise_weight": cfg["v1_listwise_weight"],
    }
    set_seed(int(cfg["v1_ranker_seed"]))
    query_rows, candidate_rows, _ = train_and_predict(v1_cfg, device)
    return query_rows, candidate_rows


def metric_values(rows: list[dict[str, str]]) -> dict[str, float]:
    return {
        "queries": float(len(rows)), "tactile_diff_mae": float(np.mean([finite(row["tactile_diff_mae"]) for row in rows])),
        "tactile_ssim": float(np.mean([finite(row["tactile_ssim"]) for row in rows])),
        "tactile_mask_iou": float(np.mean([finite(row["tactile_mask_iou"]) for row in rows])),
        "oracle_top1": float(np.mean([int(row["ranker_oracle_embedding_rank"]) == 1 for row in rows])),
    }


def bootstrap_comparison(v1_rows: list[dict[str, str]], fusion_rows: list[dict[str, str]], cfg: dict) -> dict:
    v1_by_name, fusion_by_name = {row["query_image_name"]: row for row in v1_rows}, {row["query_image_name"]: row for row in fusion_rows}
    if set(v1_by_name) != set(fusion_by_name):
        raise RuntimeError("Bootstrap requires one-to-one V1 and fusion validation queries.")
    names_by_record: dict[str, list[str]] = defaultdict(list)
    for name, row in v1_by_name.items(): names_by_record[row["query_record_id"]].append(name)
    record_ids = sorted(names_by_record)
    rng = np.random.default_rng(int(cfg["bootstrap_seed"]))
    specs = {"tactile_diff_mae": -1, "tactile_ssim": 1, "tactile_mask_iou": 1, "oracle_top1": 1}
    output = {}
    for regime, predicate in (("all", lambda _: True), ("far_probe75_100", lambda row: int(row["query_probe"]) >= 75)):
        selected_names = [name for name in v1_by_name if predicate(v1_by_name[name])]
        base = metric_values([v1_by_name[name] for name in selected_names])
        fused = metric_values([fusion_by_name[name] for name in selected_names])
        samples = {metric: [] for metric in specs}
        for _ in range(int(cfg["bootstrap_iterations"])):
            draw = rng.choice(record_ids, len(record_ids), replace=True)
            names = [name for record_id in draw for name in names_by_record[record_id] if name in selected_names]
            for metric in specs:
                if metric == "oracle_top1":
                    value = np.mean([int(fusion_by_name[name]["ranker_oracle_embedding_rank"]) == 1 for name in names]) - np.mean([int(v1_by_name[name]["ranker_oracle_embedding_rank"]) == 1 for name in names])
                else:
                    value = np.mean([finite(fusion_by_name[name][metric]) for name in names]) - np.mean([finite(v1_by_name[name][metric]) for name in names])
                samples[metric].append(float(value))
        deltas = {metric: fused[metric] - base[metric] for metric in specs}
        ci = {metric: [float(np.quantile(samples[metric], 0.025)), float(np.quantile(samples[metric], 0.975))] for metric in specs}
        point_pass = fused["tactile_diff_mae"] < base["tactile_diff_mae"] and fused["tactile_ssim"] >= base["tactile_ssim"] and fused["tactile_mask_iou"] >= base["tactile_mask_iou"] and fused["oracle_top1"] >= base["oracle_top1"]
        ci_pass = ci["tactile_diff_mae"][1] < 0 and ci["tactile_ssim"][0] >= 0 and ci["tactile_mask_iou"][0] >= 0
        output[regime] = {"v1": base, "fusion": fused, "delta_fusion_minus_v1": deltas, "bootstrap_95_ci": ci, "point_pass": point_pass, "ci_pass": ci_pass}
    output["accepted"] = bool(output["all"]["point_pass"] and output["all"]["ci_pass"] and output["far_probe75_100"]["point_pass"] and output["far_probe75_100"]["ci_pass"])
    return output


def main_build(config_path: str, section: str) -> dict:
    cfg = load_config(config_path)[section]
    set_seed(int(cfg["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples = read_csv_rows(project_path(cfg["samples_csv"]))
    if any(is_final_holdout(row) for row in samples):
        raise RuntimeError("Phase4G refuses sealed final-holdout samples.")
    samples_by_name = {row["image_name"]: row for row in samples}
    v1_oof_queries = read_csv_rows(project_path(cfg["v1_oof_query_csv"]))
    dino_oof_queries = read_csv_rows(project_path(cfg["dino_oof_query_csv"]))
    if {row["query_image_name"] for row in v1_oof_queries} != {row["query_image_name"] for row in dino_oof_queries}:
        raise RuntimeError("V1 and DINO OOF query sets differ; refusing fusion.")
    dino_oof_by_name = {row["query_image_name"]: row for row in dino_oof_queries}
    if any(dino_oof_by_name[row["query_image_name"]]["oof_fold"] != row["oof_fold"] for row in v1_oof_queries):
        raise RuntimeError("V1 and DINO OOF fold assignments differ; refusing fusion.")
    v1_oof_candidates = read_csv_rows(project_path(cfg["v1_oof_candidate_csv"]))
    dino_oof_candidates = read_csv_rows(project_path(cfg["dino_oof_candidate_csv"]))
    oof_groups = merge_candidates(v1_oof_candidates, dino_oof_candidates, int(cfg["geometry_filter_k"]))
    if any(item["v1"]["candidate_record_id"] == query["query_record_id"] for query in v1_oof_queries for item in oof_groups[query["query_image_name"]]):
        raise RuntimeError("Phase4G OOF candidate audit found a same-record cache entry.")
    folds = sorted({row["oof_fold"] for row in v1_oof_queries})
    scores, weight_rows = {}, []
    for fold in folds:
        held_out = [row["query_image_name"] for row in v1_oof_queries if row["oof_fold"] == fold]
        fit_names = [row["query_image_name"] for row in v1_oof_queries if row["oof_fold"] != fold]
        weights = fit_weights(*stack_groups(oof_groups, fit_names), cfg)
        scores.update(score_groups(oof_groups, held_out, weights))
        weight_rows.append({"scope": "nested_oof", "oof_fold": fold, "v1_weight": f"{weights[0]:.8f}", "dino_detail_weight": f"{weights[1]:.8f}", "dino_context_weight": f"{weights[2]:.8f}", "training_queries": str(len(fit_names))})
    oof_output_queries, oof_output_candidates = build_outputs(
        oof_groups, v1_oof_queries, scores, samples_by_name, int(cfg["tactile_size"]), float(cfg["tactile_mask_threshold"]),
    )
    write_csv_rows(project_path(cfg["fusion_oof_query_csv"]), oof_output_queries, V1_QUERY_FIELDS)
    write_csv_rows(project_path(cfg["fusion_oof_candidate_csv"]), oof_output_candidates, FUSION_CANDIDATE_FIELDS)
    write_csv_rows(project_path(cfg["fusion_weights_csv"]), weight_rows, WEIGHT_FIELDS)

    v1_validation_queries, v1_validation_candidates = v1_validation_outputs(cfg, device)
    write_csv_rows(project_path(cfg["v1_validation_query_csv"]), v1_validation_queries, V1_QUERY_FIELDS)
    write_csv_rows(project_path(cfg["v1_validation_candidate_csv"]), v1_validation_candidates, V1_CANDIDATE_FIELDS)
    dino_validation_queries, dino_validation_candidates = dino_validation_outputs(cfg, device)
    write_csv_rows(project_path(cfg["dino_validation_query_csv"]), dino_validation_queries, [
        "query_record_id", "query_image_name", "query_probe", "oof_fold", "pred_x", "pred_y", "selected_cache_record_id", "selected_cache_image_name", "ranker_best_score", "ranker_margin", "ranker_oracle_embedding_rank", "tactile_diff_mae", "tactile_ssim", "tactile_mask_iou",
    ])
    write_csv_rows(project_path(cfg["dino_validation_candidate_csv"]), dino_validation_candidates, [
        "query_record_id", "query_image_name", "query_probe", "oof_fold", "candidate_rank", "candidate_score", "detail_patch_similarity", "context_patch_similarity", "patch_similarity_score", "candidate_record_id", "candidate_image_name", "candidate_tactile_embedding_distance", "candidate_tactile_ssim", "candidate_tactile_mask_iou", "candidate_oracle_embedding_rank",
    ])
    validation_groups = merge_candidates(v1_validation_candidates, dino_validation_candidates, int(cfg["geometry_filter_k"]))
    all_oof_names = [row["query_image_name"] for row in v1_oof_queries]
    validation_weights = fit_weights(*stack_groups(oof_groups, all_oof_names), cfg)
    validation_scores = score_groups(validation_groups, [row["query_image_name"] for row in v1_validation_queries], validation_weights)
    fusion_validation_queries, fusion_validation_candidates = build_outputs(
        validation_groups, v1_validation_queries, validation_scores, samples_by_name, int(cfg["tactile_size"]), float(cfg["tactile_mask_threshold"]),
    )
    write_csv_rows(project_path(cfg["fusion_validation_query_csv"]), fusion_validation_queries, V1_QUERY_FIELDS)
    write_csv_rows(project_path(cfg["fusion_validation_candidate_csv"]), fusion_validation_candidates, FUSION_CANDIDATE_FIELDS)
    weight_rows.append({"scope": "validation", "oof_fold": "all", "v1_weight": f"{validation_weights[0]:.8f}", "dino_detail_weight": f"{validation_weights[1]:.8f}", "dino_context_weight": f"{validation_weights[2]:.8f}", "training_queries": str(len(all_oof_names))})
    write_csv_rows(project_path(cfg["fusion_weights_csv"]), weight_rows, WEIGHT_FIELDS)
    acceptance = bootstrap_comparison(v1_validation_queries, fusion_validation_queries, cfg)
    summary = {
        "mode": "phase4g_strict_oof_v1_dino_convex_fusion", "device": str(device), "candidate_audit": {"oof_queries": len(oof_groups), "validation_queries": len(validation_groups), "top_k": int(cfg["geometry_filter_k"]), "passed": True},
        "weights": {"validation": [float(value) for value in validation_weights]}, "validation": {
            "v1": {"all": metric_values(v1_validation_queries), "far_probe75_100": metric_values([row for row in v1_validation_queries if int(row["query_probe"]) >= 75])},
            "dino": {"all": metric_values(dino_validation_queries), "far_probe75_100": metric_values([row for row in dino_validation_queries if int(row["query_probe"]) >= 75])},
            "fusion": {"all": metric_values(fusion_validation_queries), "far_probe75_100": metric_values([row for row in fusion_validation_queries if int(row["query_probe"]) >= 75])},
        }, "acceptance": acceptance,
        "integrity": {"c2_contact_box": "unchanged", "cache": "fixed development-train cache", "same_record_cache": "excluded by disjoint validation records", "final_holdout_rows_read": 0, "query_tactile_usage": "offline supervision and evaluation only"},
        "next_action": "retrain trust/cache-miss from fusion OOF only" if acceptance["accepted"] else "retain V1; do not run V2 final holdout or retrain trust",
    }
    write_json(project_path(cfg["metrics_json"]), summary)
    print(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build strict Phase4G V1+DINO convex fusion and validation report.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--section", default="phase4g_dino_v1_fusion_v1")
    args = parser.parse_args()
    main_build(args.config, args.section)


if __name__ == "__main__":
    main()
