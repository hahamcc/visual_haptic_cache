"""Build frozen Phase4H DINO features for development validation only after OOF passes."""
from __future__ import annotations

import argparse
import gc
import json

import numpy as np
import torch

from .build_phase4h_dino_ablation import encode_rows, score_scale
from .config import load_config, project_path
from .phase4f_dino_cross_attention import FrozenDinoV2
from .phase4h_dino_adaptation import assert_development_only, candidate_groups
from .temporal_progress import read_trajectory_tracks
from .train_phase4b_predicted_box_cache_ranker import prediction_map, set_seed
from .train_soft_tactile_cache_ranker import ranks
from .utils import ensure_dir, read_csv_rows, write_csv_rows, write_json


CANDIDATE_FIELDS = [
    "query_record_id", "query_image_name", "query_probe", "oof_fold", "recipe_name",
    "candidate_rank", "candidate_score", "detail_patch_score", "context_patch_score",
    "wide_patch_score", "position_aware_match_score", "hard_negative_flag",
    "candidate_record_id", "candidate_image_name", "candidate_tactile_embedding_distance",
    "candidate_oracle_embedding_rank",
]


def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build(config_path: str, section: str) -> dict:
    cfg = load_config(config_path)[section]
    oof_report = load_json(project_path(cfg["oof_evaluation_json"]))
    if not oof_report.get("ready_for_development_validation", False):
        raise RuntimeError("Phase4H OOF acceptance did not pass; validation feature generation is blocked")
    frontier = load_json(project_path(cfg["frontier_json"]))
    recipe_name = frontier["primary_recipe"]
    recipe = next(item["recipe"] for item in frontier["selected"] if item["recipe"]["name"] == recipe_name)
    set_seed(int(cfg["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples = read_csv_rows(project_path(cfg["samples_csv"]))
    assert_development_only(samples, project_path(cfg["final_partition_csv"]))
    cache_rows = [row for row in samples if row["dataset_split"] == "train"]
    query_rows = [row for row in samples if row["dataset_split"] == "val"]
    if {row["record_id"] for row in cache_rows} & {row["record_id"] for row in query_rows}:
        raise RuntimeError("Phase4H validation cache and query records overlap")
    predictions = prediction_map(
        read_csv_rows(project_path(cfg["validation_predictions_csv"])),
        query_rows,
        "val",
        "Phase4H frozen validation C2",
    )
    groups = candidate_groups(read_csv_rows(project_path(cfg["v1_validation_candidate_csv"])), int(cfg["geometry_filter_k"]))
    if set(groups) != {row["image_name"] for row in query_rows}:
        raise RuntimeError("Frozen V1 validation candidate table does not cover validation queries")
    cache_index = {row["image_name"]: index for index, row in enumerate(cache_rows)}
    candidates = np.stack(
        [
            np.asarray([cache_index[item["candidate_image_name"]] for item in groups[row["image_name"]]], dtype=np.int32)
            for row in query_rows
        ]
    )
    query_xy = [
        (float(predictions[row["image_name"]]["pred_x"]), float(predictions[row["image_name"]]["pred_y"]))
        for row in query_rows
    ]
    cache_xy = [(float(row["target_tip_x"]), float(row["target_tip_y"])) for row in cache_rows]
    tracks = read_trajectory_tracks(project_path(cfg["motion_tracks_csv"]))
    backbone = FrozenDinoV2(str(cfg["dino_model"]), int(cfg["dino_image_size"])).to(device)
    scale_scores, pooled_parts, padding_parts = [], [], []
    token_cache_dir = project_path(cfg["token_cache_dir"])
    for size in [int(value) for value in recipe["scales"]]:
        cache_key = "_".join(
            (
                str(recipe["padding_mode"]),
                str(recipe["canonicalization_mode"]),
                str(recipe["layer_recipe"]),
                str(size),
            )
        )
        query_tokens, query_pooled, query_padding = encode_rows(
            backbone, query_rows, query_xy, size, str(recipe["padding_mode"]),
            str(recipe["canonicalization_mode"]), str(recipe["layer_recipe"]), tracks,
            device, int(cfg["batch_size"]), float(cfg["center_sigma"]),
            f"phase4h validation {recipe_name} query scale={size}",
            token_cache_dir / f"{cache_key}_validation_query",
        )
        cache_tokens, _, _ = encode_rows(
            backbone, cache_rows, cache_xy, size, str(recipe["padding_mode"]),
            str(recipe["canonicalization_mode"]), str(recipe["layer_recipe"]), tracks,
            device, int(cfg["batch_size"]), float(cfg["center_sigma"]),
            f"phase4h validation {recipe_name} cache scale={size}",
            token_cache_dir / f"{cache_key}_development_cache",
        )
        scale_scores.append(
            score_scale(
                query_tokens, cache_tokens, candidates, str(recipe["matcher"]), cfg,
                device, int(cfg["score_batch_size"]),
            )
        )
        pooled_parts.append(query_pooled)
        padding_parts.append(query_padding)
        del query_tokens, cache_tokens
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    weights = np.asarray(recipe["scale_weights"], dtype=np.float32)
    combined = np.einsum("s,sqk->qk", weights, np.stack(scale_scores))
    scores = -combined
    candidate_output = []
    for index, query in enumerate(query_rows):
        order = np.argsort(scores[index], kind="stable")
        targets = np.asarray(
            [float(item["candidate_tactile_embedding_distance"]) for item in groups[query["image_name"]]],
            dtype=np.float32,
        )
        oracle_ranks = ranks(targets)
        dino_ranks = ranks(scores[index])
        for rank, item in enumerate(order, start=1):
            item = int(item)
            source = groups[query["image_name"]][item]
            per_scale = [float(values[index, item]) for values in scale_scores]
            candidate_output.append(
                {
                    "query_record_id": query["record_id"],
                    "query_image_name": query["image_name"],
                    "query_probe": query["probe"],
                    "oof_fold": "validation",
                    "recipe_name": recipe_name,
                    "candidate_rank": str(rank),
                    "candidate_score": f"{scores[index, item]:.6f}",
                    "detail_patch_score": f"{per_scale[0]:.6f}",
                    "context_patch_score": f"{per_scale[1]:.6f}" if len(per_scale) > 1 else "",
                    "wide_patch_score": f"{per_scale[2]:.6f}" if len(per_scale) > 2 else "",
                    "position_aware_match_score": f"{combined[index, item]:.6f}" if recipe["matcher"] == "position_soft" else "",
                    "hard_negative_flag": str(int(dino_ranks[item] <= 8 and oracle_ranks[item] >= 17)),
                    "candidate_record_id": source["candidate_record_id"],
                    "candidate_image_name": source["candidate_image_name"],
                    "candidate_tactile_embedding_distance": source["candidate_tactile_embedding_distance"],
                    "candidate_oracle_embedding_rank": str(int(oracle_ranks[item])),
                }
            )
    feature_path = project_path(cfg["feature_cache_npz"])
    ensure_dir(feature_path.parent)
    np.savez_compressed(
        feature_path,
        image_names=np.asarray([row["image_name"] for row in query_rows]),
        query_features=np.concatenate(pooled_parts, axis=1).astype(np.float32),
        query_padding_ratio=np.stack(padding_parts, axis=1).mean(axis=1).astype(np.float32),
        recipe_json=np.asarray([json.dumps(recipe, sort_keys=True)]),
    )
    write_csv_rows(project_path(cfg["candidate_output_csv"]), candidate_output, CANDIDATE_FIELDS)
    summary = {
        "mode": "phase4h_frozen_development_validation_features_v1",
        "recipe": recipe,
        "validation_queries": len(query_rows),
        "cache_entries": len(cache_rows),
        "feature_cache_npz": str(feature_path),
        "integrity": {
            "oof_acceptance_required": True,
            "c2_contact_box": "unchanged",
            "top32_candidates": "frozen V1 validation identity",
            "cache_query_record_overlap": 0,
            "sealed_final_holdout_rows_read": 0,
            "query_tactile_usage": "none",
        },
    }
    write_json(project_path(cfg["metrics_json"]), summary)
    print(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build frozen Phase4H development-validation DINO features.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--section", default="phase4h_validation_features_v1")
    args = parser.parse_args()
    build(args.config, args.section)


if __name__ == "__main__":
    main()
