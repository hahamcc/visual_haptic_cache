"""One-shot development-validation evaluation for the frozen Phase4H recipe."""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from .build_phase4g_dino_v1_fusion import bootstrap_comparison
from .config import load_config, project_path
from .evaluate_oracle_tactile_retrieval import tactile_difference, tactile_metrics
from .phase4h_dino_adaptation import (
    TACTILE_LATENT_DIM,
    TactileLatentProjector,
    assert_development_only,
    candidate_groups,
    deployable_motion_feature,
    tactile_latent,
)
from .temporal_progress import DEFAULT_TTC_VALUES, masked_trajectory_features, read_trajectory_tracks
from .train_phase4b_predicted_box_cache_ranker import prediction_map
from .train_phase4h_dino_gate import (
    OUTPUT_CANDIDATE_FIELDS as GATED_CANDIDATE_FIELDS,
    OUTPUT_QUERY_FIELDS as GATED_QUERY_FIELDS,
    DinoSafetyGate,
    feature_names as gate_feature_names,
    grouped,
    labels as gate_labels,
    make_features as make_gate_features,
)
from .train_phase4h_dino_tactile_alignment import (
    CANDIDATE_FIELDS as ALIGNED_CANDIDATE_FIELDS,
    QUERY_FIELDS as ALIGNED_QUERY_FIELDS,
    load_feature_cache,
    load_tactile_index,
    predict_scores,
    softmax_entropy,
    standardize,
)
from .train_soft_tactile_cache_ranker import ranks
from .utils import read_csv_rows, write_csv_rows, write_json


def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validation_aligned_outputs(cfg: dict, device: torch.device):
    samples = read_csv_rows(project_path(cfg["samples_csv"]))
    assert_development_only(samples, project_path(cfg["final_partition_csv"]))
    cache_rows = [row for row in samples if row["dataset_split"] == "train"]
    query_rows = [row for row in samples if row["dataset_split"] == "val"]
    cache_names, query_names = [row["image_name"] for row in cache_rows], [row["image_name"] for row in query_rows]
    cache_index = {name: index for index, name in enumerate(cache_names)}
    predictions = prediction_map(
        read_csv_rows(project_path(cfg["validation_predictions_csv"])),
        query_rows,
        "val",
        "Phase4H frozen validation C2",
    )
    visual, padding = load_feature_cache(project_path(cfg["validation_feature_cache_npz"]), query_names)
    tactile_index = load_tactile_index(project_path(cfg["tactile_index_npz"]), cache_names)
    cache_latents = standardize(tactile_index["raw"], tactile_index["full_mean"], tactile_index["full_std"])
    v1_groups = candidate_groups(
        read_csv_rows(project_path(cfg["v1_validation_candidate_csv"])),
        int(cfg["geometry_filter_k"]),
    )
    candidates = np.stack(
        [
            np.asarray([cache_index[item["candidate_image_name"]] for item in v1_groups[name]], dtype=np.int32)
            for name in query_names
        ]
    )
    recipe_candidate_rows = read_csv_rows(project_path(cfg["validation_recipe_candidate_csv"]))
    recipe_map = {
        (row["query_image_name"], row["candidate_image_name"]): row
        for row in recipe_candidate_rows
    }
    tracks = read_trajectory_tracks(project_path(cfg["motion_tracks_csv"]))
    ttc_by_name = {}
    ttc_value = str(cfg.get("ttc_predictions_csv", "")).strip()
    ttc_path = project_path(ttc_value) if ttc_value else None
    if ttc_path is not None and ttc_path.is_file():
        ttc_by_name = {row["image_name"]: row for row in read_csv_rows(ttc_path) if row["image_name"] in set(query_names)}
    online_motion = []
    for row in query_rows:
        trajectory, mask, quality = masked_trajectory_features(
            row, tracks, int(cfg["trajectory_history_frames"]),
            float(cfg["trajectory_spatial_scale_px"]), float(cfg["trajectory_speed_scale_px"]),
        )
        prediction = predictions[row["image_name"]]
        online_motion.append(
            deployable_motion_feature(
                row, float(prediction["pred_x"]), float(prediction["pred_y"]),
                trajectory, mask, quality, ttc_by_name.get(row["image_name"], prediction),
                cfg.get("ttc_values", DEFAULT_TTC_VALUES),
            )
        )
    online_motion = np.stack(online_motion).astype(np.float32)
    seed_scores, seed_predictions = [], []
    for seed in [int(value) for value in cfg["seeds"]]:
        checkpoint = torch.load(
            project_path(cfg["alignment_checkpoint_dir"]) / f"full_seed_{seed}.pt",
            map_location=device,
            weights_only=False,
        )
        metadata = checkpoint["metadata"]
        motion = standardize(
            online_motion,
            np.asarray(metadata["motion_mean"], dtype=np.float32),
            np.asarray(metadata["motion_std"], dtype=np.float32),
        )
        features = np.concatenate((visual, motion), axis=1).astype(np.float32)
        model = TactileLatentProjector(
            int(checkpoint["input_dim"]), int(checkpoint["latent_dim"]), float(cfg["dropout"]),
        ).to(device)
        model.load_state_dict(checkpoint["model_state"])
        scores, predicted = predict_scores(
            model, features, candidates, cache_latents,
            np.arange(len(query_rows), dtype=np.int32), device, int(cfg["batch_size"]),
        )
        seed_scores.append(scores)
        seed_predictions.append(predicted)
    scores = np.mean(seed_scores, axis=0)
    predicted_latents = np.mean(seed_predictions, axis=0)

    touch_cache: dict[str, np.ndarray] = {}
    query_output, candidate_output = [], []
    recipe_name = recipe_candidate_rows[0]["recipe_name"] if recipe_candidate_rows else ""
    for index, query in enumerate(query_rows):
        query_diff = tactile_difference(query["touch_path"], touch_cache, int(cfg["tactile_size"]))
        query_raw = tactile_latent(query_diff, float(cfg["tactile_mask_threshold"]))
        query_latent = standardize(query_raw[None], tactile_index["full_mean"], tactile_index["full_std"])[0]
        target = ((cache_latents[candidates[index]] - query_latent[None]) ** 2).mean(axis=1)
        order = np.argsort(scores[index], kind="stable")
        model_ranks, oracle_ranks = ranks(scores[index]), ranks(target)
        choice = int(order[0])
        selected = cache_rows[int(candidates[index, choice])]
        metric = tactile_metrics(
            query_diff,
            tactile_difference(selected["touch_path"], touch_cache, int(cfg["tactile_size"])),
            float(cfg["tactile_mask_threshold"]),
        )
        best, second = float(scores[index, order[0]]), float(scores[index, order[1]])
        top3 = order[:3]
        prediction = predictions[query["image_name"]]
        query_output.append(
            {
                "query_record_id": query["record_id"],
                "query_image_name": query["image_name"],
                "query_probe": query["probe"],
                "oof_fold": "validation",
                "recipe_name": recipe_name,
                "pred_x": prediction["pred_x"],
                "pred_y": prediction["pred_y"],
                "query_padding_ratio": f"{padding[index]:.6f}",
                "predicted_ttc": f"{online_motion[index, 25] * 100.0:.6f}",
                "ttc_entropy": f"{online_motion[index, 26]:.6f}",
                "trajectory_stability": f"{online_motion[index, 18]:.6f}",
                "ranker_best_score": f"{best:.6f}",
                "ranker_second_score": f"{second:.6f}",
                "ranker_margin": f"{second - best:.6f}",
                "ranker_margin_normalized": f"{(second - best) / max(float(scores[index].std()), 1e-6):.6f}",
                "ranker_entropy": f"{softmax_entropy(scores[index]):.6f}",
                "ranker_oracle_embedding_rank": str(int(model_ranks[int(np.argmin(target))])),
                "selected_cache_record_id": selected["record_id"],
                "selected_cache_image_name": selected["image_name"],
                "top3_cache_record_ids": "|".join(cache_rows[int(candidates[index, item])]["record_id"] for item in top3),
                "top3_cache_image_names": "|".join(cache_rows[int(candidates[index, item])]["image_name"] for item in top3),
                "predicted_tactile_latent_error": f"{float(np.mean((predicted_latents[index] - query_latent) ** 2)):.6f}",
                **{name: f"{metric[name]:.6f}" for name in ("tactile_diff_mae", "tactile_ssim", "tactile_mask_iou")},
            }
        )
        for rank, item in enumerate(order, start=1):
            item = int(item)
            cache = cache_rows[int(candidates[index, item])]
            source = recipe_map[(query["image_name"], cache["image_name"])]
            candidate_output.append(
                {
                    "query_record_id": query["record_id"],
                    "query_image_name": query["image_name"],
                    "query_probe": query["probe"],
                    "oof_fold": "validation",
                    "recipe_name": recipe_name,
                    "candidate_rank": str(rank),
                    "candidate_score": f"{scores[index, item]:.6f}",
                    "predicted_tactile_latent_distance": f"{scores[index, item]:.6f}",
                    "candidate_tactile_latent_distance": f"{target[item]:.6f}",
                    **{
                        field: source.get(field, "")
                        for field in (
                            "detail_patch_score", "context_patch_score", "wide_patch_score",
                            "position_aware_match_score",
                        )
                    },
                    "hard_negative_flag": str(int(int(source["candidate_rank"]) <= 8 and oracle_ranks[item] >= 17)),
                    "candidate_record_id": cache["record_id"],
                    "candidate_image_name": cache["image_name"],
                    "candidate_oracle_embedding_rank": str(int(oracle_ranks[item])),
                }
            )
    return query_output, candidate_output


def gated_outputs(cfg: dict, aligned_queries, aligned_candidates, device: torch.device):
    v1_queries = read_csv_rows(project_path(cfg["v1_validation_query_csv"]))
    v1_by_name = {row["query_image_name"]: row for row in v1_queries}
    order = {row["query_image_name"]: index for index, row in enumerate(v1_queries)}
    aligned_queries.sort(key=lambda row: order[row["query_image_name"]])
    v1_groups = grouped(read_csv_rows(project_path(cfg["v1_validation_candidate_csv"])))
    aligned_groups = grouped(aligned_candidates)
    raw = make_gate_features(aligned_queries, v1_by_name, v1_groups, aligned_groups)
    checkpoint = torch.load(project_path(cfg["gate_checkpoint"]), map_location=device, weights_only=False)
    if checkpoint["feature_names"] != gate_feature_names():
        raise RuntimeError("Phase4H gate checkpoint feature schema differs from validation implementation")
    mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["feature_std"], dtype=np.float32)
    features = standardize(raw, mean, std)
    model = DinoSafetyGate(features.shape[1], float(cfg["gate_dropout"])).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(features).to(device)).cpu().numpy()
    gate = load_json(project_path(cfg["gate_json"]))
    temperature = float(gate["temperature"])
    threshold = gate["threshold"]
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits / temperature, -60.0, 60.0)))
    accepted = probabilities >= float(threshold) if gate["enabled"] and threshold is not None else np.zeros(len(aligned_queries), dtype=bool)
    targets = gate_labels(v1_queries, aligned_queries)
    output_queries, output_candidates = [], []
    for index, (v1, dino) in enumerate(zip(v1_queries, aligned_queries, strict=True)):
        use_dino = bool(accepted[index])
        final = dino if use_dino else v1
        output_queries.append(
            {
                "query_record_id": dino["query_record_id"],
                "query_image_name": dino["query_image_name"],
                "query_probe": dino["query_probe"],
                "oof_fold": "validation",
                "dino_accept_probability": f"{probabilities[index]:.6f}",
                "gate_threshold": "" if threshold is None else f"{float(threshold):.6f}",
                "dino_accepted": str(int(use_dino)),
                "final_selection_source": "aligned_dino" if use_dino else "v1",
                "selected_cache_record_id": final["selected_cache_record_id"],
                "selected_cache_image_name": final["selected_cache_image_name"],
                "ranker_oracle_embedding_rank": final["ranker_oracle_embedding_rank"],
                **{metric: final[metric] for metric in ("tactile_diff_mae", "tactile_ssim", "tactile_mask_iou")},
                "v1_selected_cache_image_name": v1["selected_cache_image_name"],
                "dino_selected_cache_image_name": dino["selected_cache_image_name"],
                **{f"v1_{metric}": v1[metric] for metric in ("tactile_diff_mae", "tactile_ssim", "tactile_mask_iou")},
                **{f"dino_{metric}": dino[metric] for metric in ("tactile_diff_mae", "tactile_ssim", "tactile_mask_iou")},
                "strict_triple_win_label": str(int(targets[index])),
            }
        )
        source = aligned_groups[dino["query_image_name"]] if use_dino else v1_groups[dino["query_image_name"]]
        for rank, row in enumerate(source, start=1):
            dino_source = next(
                item for item in aligned_groups[dino["query_image_name"]]
                if item["candidate_image_name"] == row["candidate_image_name"]
            )
            output_candidates.append(
                {
                    "query_record_id": dino["query_record_id"],
                    "query_image_name": dino["query_image_name"],
                    "query_probe": dino["query_probe"],
                    "oof_fold": "validation",
                    "candidate_rank": str(rank),
                    "candidate_score": row["candidate_score"],
                    "candidate_record_id": row["candidate_record_id"],
                    "candidate_image_name": row["candidate_image_name"],
                    **{
                        field: dino_source.get(field, "")
                        for field in (
                            "predicted_tactile_latent_distance", "candidate_tactile_latent_distance",
                            "detail_patch_score", "context_patch_score", "wide_patch_score",
                            "position_aware_match_score", "hard_negative_flag",
                            "candidate_oracle_embedding_rank",
                        )
                    },
                    "dino_accept_probability": f"{probabilities[index]:.6f}",
                    "final_selection_source": "aligned_dino" if use_dino else "v1",
                }
            )
    return v1_queries, output_queries, output_candidates, accepted, targets


def evaluate(config_path: str, section: str) -> dict:
    cfg = load_config(config_path)[section]
    oof = load_json(project_path(cfg["oof_evaluation_json"]))
    if not oof.get("ready_for_development_validation", False):
        raise RuntimeError("Phase4H OOF acceptance did not pass; validation is blocked")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    aligned_queries, aligned_candidates = validation_aligned_outputs(cfg, device)
    v1_queries, gated_queries, gated_candidates, accepted, targets = gated_outputs(
        cfg, aligned_queries, aligned_candidates, device,
    )
    write_csv_rows(project_path(cfg["aligned_query_output_csv"]), aligned_queries, ALIGNED_QUERY_FIELDS)
    write_csv_rows(project_path(cfg["aligned_candidate_output_csv"]), aligned_candidates, ALIGNED_CANDIDATE_FIELDS)
    write_csv_rows(project_path(cfg["gated_query_output_csv"]), gated_queries, GATED_QUERY_FIELDS)
    write_csv_rows(project_path(cfg["gated_candidate_output_csv"]), gated_candidates, GATED_CANDIDATE_FIELDS)
    comparison_cfg = {
        "bootstrap_iterations": int(cfg["bootstrap_iterations"]),
        "bootstrap_seed": int(cfg["bootstrap_seed"]),
    }
    aligned_comparison = bootstrap_comparison(v1_queries, aligned_queries, comparison_cfg)
    gated_comparison = bootstrap_comparison(v1_queries, gated_queries, comparison_cfg)
    coverage = float(accepted.mean())
    precision = float(targets[accepted].mean()) if accepted.any() else 0.0
    accepted_final = bool(
        gated_comparison["accepted"]
        and coverage >= float(cfg["minimum_coverage"])
        and precision >= float(cfg["minimum_precision"])
    )
    report = {
        "mode": "phase4h_frozen_development_validation_v1",
        "aligned_dino_minus_v1": aligned_comparison,
        "gated_minus_v1": gated_comparison,
        "gate_validation": {
            "coverage": coverage,
            "accepted_queries": int(accepted.sum()),
            "strict_triple_win_precision": precision,
        },
        "accepted": accepted_final,
        "allowed_to_run_v2_final_holdout": accepted_final,
        "next_action": (
            "freeze Phase4H artifacts, retrain trust/cache-miss from strict Phase4H OOF, then run V2 final holdout once"
            if accepted_final
            else "retain V1; do not run V2 final holdout"
        ),
        "integrity": {
            "c2_contact_box": "unchanged",
            "top32_candidates": "frozen V1 identity",
            "sealed_final_holdout_rows_read": 0,
            "query_tactile_usage": "offline validation labels and metrics only",
        },
    }
    write_json(project_path(cfg["metrics_json"]), report)
    print(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one-shot frozen Phase4H development validation.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--section", default="phase4h_validation_v1")
    args = parser.parse_args()
    evaluate(args.config, args.section)


if __name__ == "__main__":
    main()
