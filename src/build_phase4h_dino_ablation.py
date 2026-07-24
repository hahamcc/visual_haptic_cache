"""Strict-OOF frozen-DINO ablations on the exact V1 Top-32 candidate sets."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .build_cache_retrieval import crop_contact_patch
from .build_phase4f_dino_patch_similarity_cache import local_patch_similarity
from .config import load_config, project_path
from .evaluate_oracle_tactile_retrieval import tactile_difference, tactile_metrics
from .phase4f_dino_cross_attention import FrozenDinoV2
from .phase4h_dino_adaptation import (
    assert_candidate_identity,
    assert_development_only,
    candidate_groups,
    candidate_set_fingerprint,
    canonical_rotation_degrees,
    combine_layer_tokens,
    contact_crop_reflect,
    pooled_token_features,
    position_aware_soft_similarity,
)
from .temporal_progress import read_trajectory_tracks
from .train_phase4b_predicted_box_cache_ranker import prediction_map, set_seed
from .train_soft_tactile_cache_ranker import ranks
from .utils import ensure_dir, read_csv_rows, write_csv_rows, write_json


QUERY_FIELDS = [
    "recipe_name", "query_record_id", "query_image_name", "query_probe", "oof_fold",
    "pred_x", "pred_y", "dino_layer_recipe", "canonicalization_mode", "padding_mode",
    "matcher", "scales", "query_padding_ratio", "selected_cache_record_id",
    "selected_cache_image_name", "ranker_best_score", "ranker_margin",
    "ranker_oracle_embedding_rank", "tactile_diff_mae", "tactile_ssim", "tactile_mask_iou",
]
CANDIDATE_FIELDS = [
    "recipe_name", "query_record_id", "query_image_name", "query_probe", "oof_fold",
    "candidate_rank", "candidate_score", "detail_patch_score", "context_patch_score",
    "wide_patch_score", "position_aware_match_score", "candidate_record_id",
    "candidate_image_name", "candidate_tactile_embedding_distance",
    "candidate_tactile_ssim", "candidate_tactile_mask_iou",
    "candidate_oracle_embedding_rank", "hard_negative_flag",
]


def image_batch(crops: list[np.ndarray]) -> torch.Tensor:
    return torch.from_numpy(np.stack(crops)).permute(0, 3, 1, 2).contiguous().float()


def recipe_layers(recipe: str) -> tuple[int, ...]:
    return (8, 10, 12) if recipe == "mean_8_10_12" else (int(recipe.removeprefix("layer")),)


def crop_for_row(
    row: dict[str, str],
    xy: tuple[float, float],
    size: int,
    padding_mode: str,
    canonicalization: str,
    tracks: dict,
) -> tuple[np.ndarray, float]:
    angle, _ = canonical_rotation_degrees(row, canonicalization, tracks)
    if padding_mode == "black":
        if canonicalization != "raw":
            raise ValueError("Black padding is supported only for the frozen raw baseline")
        return crop_contact_patch(row["vision_path"], xy[0], xy[1], size), 1.0 - (
            max(0.0, min(float(row["image_width"]), xy[0] + size / 2) - max(0.0, xy[0] - size / 2))
            * max(0.0, min(float(row["image_height"]), xy[1] + size / 2) - max(0.0, xy[1] - size / 2))
            / float(size * size)
        )
    if padding_mode != "reflect":
        raise ValueError(f"Unsupported padding mode: {padding_mode}")
    return contact_crop_reflect(row["vision_path"], xy[0], xy[1], size, angle)


def encode_rows(
    backbone: FrozenDinoV2,
    rows: list[dict[str, str]],
    coordinates: list[tuple[float, float]],
    size: int,
    padding_mode: str,
    canonicalization: str,
    layer_recipe: str,
    tracks: dict,
    device: torch.device,
    batch_size: int,
    center_sigma: float,
    label: str,
    cache_prefix: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    token_path = Path(f"{cache_prefix}.tokens.npy") if cache_prefix is not None else None
    pooled_path = Path(f"{cache_prefix}.pooled.npy") if cache_prefix is not None else None
    padding_path = Path(f"{cache_prefix}.padding.npy") if cache_prefix is not None else None
    metadata_path = Path(f"{cache_prefix}.json") if cache_prefix is not None else None
    names_hash = hashlib.sha256("\n".join(row["image_name"] for row in rows).encode("utf-8")).hexdigest()
    cache_identity = {
        "rows": len(rows),
        "names_sha256": names_hash,
        "size": int(size),
        "padding_mode": padding_mode,
        "canonicalization": canonicalization,
        "layer_recipe": layer_recipe,
        "center_sigma": float(center_sigma),
    }
    if (
        token_path is not None
        and pooled_path is not None
        and padding_path is not None
        and metadata_path is not None
        and all(path.is_file() for path in (token_path, pooled_path, padding_path, metadata_path))
    ):
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        if any(metadata.get(key) != value for key, value in cache_identity.items()):
            raise RuntimeError(f"Phase4H token cache identity mismatch: {cache_prefix}")
        print(f"{label}: reusing {cache_prefix}", flush=True)
        return (
            np.load(token_path, mmap_mode="r"),
            np.load(pooled_path, mmap_mode="r"),
            np.load(padding_path, mmap_mode="r"),
        )
    token_store = pooled_store = padding_store = None
    token_batches, pooled_batches, padding_batches = [], [], []
    requested = recipe_layers(layer_recipe)
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            end = min(start + batch_size, len(rows))
            crops, ratios = [], []
            for row, xy in zip(rows[start:end], coordinates[start:end], strict=True):
                crop, ratio = crop_for_row(
                    row, xy, size, padding_mode, canonicalization, tracks,
                )
                crops.append(crop)
                ratios.append(ratio)
            layer_tokens = backbone.forward_layers(image_batch(crops).to(device), requested)
            tokens = combine_layer_tokens(layer_tokens, layer_recipe)
            token_array = tokens.detach().cpu().to(torch.float16).numpy()
            pooled_array = pooled_token_features(tokens, center_sigma).detach().cpu().numpy().astype(np.float32)
            padding_array = np.asarray(ratios, dtype=np.float32)
            if token_path is not None and token_store is None:
                ensure_dir(token_path.parent)
                token_store = np.lib.format.open_memmap(
                    token_path, mode="w+", dtype=np.float16,
                    shape=(len(rows), token_array.shape[1], token_array.shape[2]),
                )
                pooled_store = np.lib.format.open_memmap(
                    pooled_path, mode="w+", dtype=np.float32,
                    shape=(len(rows), pooled_array.shape[1]),
                )
                padding_store = np.lib.format.open_memmap(
                    padding_path, mode="w+", dtype=np.float32, shape=(len(rows),),
                )
            if token_store is not None:
                token_store[start:end] = token_array
                pooled_store[start:end] = pooled_array
                padding_store[start:end] = padding_array
            else:
                token_batches.append(token_array)
                pooled_batches.append(pooled_array)
                padding_batches.append(padding_array)
            if end % 200 == 0 or end == len(rows):
                print(f"{label}: {end}/{len(rows)}", flush=True)
    if token_store is not None:
        token_store.flush()
        pooled_store.flush()
        padding_store.flush()
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {**cache_identity, "label": label},
                handle,
                indent=2,
            )
            handle.write("\n")
        del token_store, pooled_store, padding_store
        return (
            np.load(token_path, mmap_mode="r"),
            np.load(pooled_path, mmap_mode="r"),
            np.load(padding_path, mmap_mode="r"),
        )
    return (
        np.concatenate(token_batches),
        np.concatenate(pooled_batches).astype(np.float32),
        np.concatenate(padding_batches).astype(np.float32),
    )


def score_scale(
    query_tokens: np.ndarray,
    cache_tokens: np.ndarray,
    candidates: np.ndarray,
    matcher: str,
    cfg: dict,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    output = []
    with torch.no_grad():
        for start in range(0, len(query_tokens), batch_size):
            end = min(start + batch_size, len(query_tokens))
            indices = candidates[start:end]
            query = torch.from_numpy(query_tokens[start:end]).to(device=device, dtype=torch.float32)
            cache = torch.from_numpy(cache_tokens[indices]).to(device=device, dtype=torch.float32)
            if matcher == "hard_local":
                similarity = local_patch_similarity(query, cache, int(cfg["hard_local_radius"]))
            elif matcher == "position_soft":
                similarity = position_aware_soft_similarity(
                    query,
                    cache,
                    radius=int(cfg["soft_local_radius"]),
                    temperature=float(cfg["soft_temperature"]),
                    position_penalty=float(cfg["position_penalty"]),
                    center_sigma=float(cfg["center_sigma"]),
                )
            else:
                raise ValueError(f"Unsupported matcher: {matcher}")
            output.append(similarity.cpu().numpy())
    return np.concatenate(output).astype(np.float32)


def summarize(rows: list[dict[str, str]]) -> dict:
    result = {}
    regimes = (
        ("all", rows),
        ("near_probe5_20", [row for row in rows if int(row["query_probe"]) <= 20]),
        ("mid_probe30_50", [row for row in rows if 30 <= int(row["query_probe"]) <= 50]),
        ("far_probe75_100", [row for row in rows if int(row["query_probe"]) >= 75]),
    )
    for name, selected in regimes:
        result[name] = {
            "queries": len(selected),
            "tactile_diff_mae": float(np.mean([float(row["tactile_diff_mae"]) for row in selected])) if selected else None,
            "tactile_ssim": float(np.mean([float(row["tactile_ssim"]) for row in selected])) if selected else None,
            "tactile_mask_iou": float(np.mean([float(row["tactile_mask_iou"]) for row in selected])) if selected else None,
            "oracle_top1": float(np.mean([int(row["ranker_oracle_embedding_rank"]) == 1 for row in selected])) if selected else None,
            "oracle_top3": float(np.mean([int(row["ranker_oracle_embedding_rank"]) <= 3 for row in selected])) if selected else None,
        }
    return result


def run_recipe(
    recipe: dict,
    cfg: dict,
    rows: list[dict[str, str]],
    predictions: dict[str, dict[str, str]],
    groups: dict[str, list[dict[str, str]]],
    dino_labels: dict[tuple[str, str], dict[str, str]],
    tracks: dict,
    backbone: FrozenDinoV2,
    device: torch.device,
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict]:
    name = str(recipe["name"])
    scales = [int(value) for value in recipe["scales"]]
    weights = np.asarray(recipe["scale_weights"], dtype=np.float32)
    if len(scales) != len(weights) or not np.isclose(weights.sum(), 1.0):
        raise ValueError(f"Recipe {name} scale weights must align and sum to one")
    row_index = {row["image_name"]: index for index, row in enumerate(rows)}
    query_xy = [
        (float(predictions[row["image_name"]]["pred_x"]), float(predictions[row["image_name"]]["pred_y"]))
        for row in rows
    ]
    cache_xy = [(float(row["target_tip_x"]), float(row["target_tip_y"])) for row in rows]
    candidates = np.stack(
        [
            np.asarray([row_index[item["candidate_image_name"]] for item in groups[row["image_name"]]], dtype=np.int32)
            for row in rows
        ]
    )
    scale_scores, pooled_parts, padding_parts = [], [], []
    token_cache_dir = project_path(cfg["token_cache_dir"])
    for size in scales:
        cache_key = "_".join(
            (
                str(recipe["padding_mode"]),
                str(recipe["canonicalization_mode"]),
                str(recipe["layer_recipe"]),
                str(size),
            )
        )
        query_tokens, query_pooled, query_padding = encode_rows(
            backbone, rows, query_xy, size, str(recipe["padding_mode"]),
            str(recipe["canonicalization_mode"]), str(recipe["layer_recipe"]), tracks,
            device, int(cfg["batch_size"]), float(cfg["center_sigma"]),
            f"phase4h {name} query scale={size}",
            token_cache_dir / f"{cache_key}_query",
        )
        cache_tokens, _, _ = encode_rows(
            backbone, rows, cache_xy, size, str(recipe["padding_mode"]),
            str(recipe["canonicalization_mode"]), str(recipe["layer_recipe"]), tracks,
            device, int(cfg["batch_size"]), float(cfg["center_sigma"]),
            f"phase4h {name} cache scale={size}",
            token_cache_dir / f"{cache_key}_cache",
        )
        scale_scores.append(
            score_scale(query_tokens, cache_tokens, candidates, str(recipe["matcher"]), cfg, device, int(cfg["score_batch_size"]))
        )
        pooled_parts.append(query_pooled)
        padding_parts.append(query_padding)
        del query_tokens, cache_tokens
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    similarities = np.stack(scale_scores, axis=0)
    combined_similarity = np.einsum("s,sqk->qk", weights, similarities)
    scores = -combined_similarity
    output_queries, output_candidates = [], []
    touch_cache: dict[str, np.ndarray] = {}
    target_matrix = np.asarray(
        [
            [float(item["candidate_tactile_embedding_distance"]) for item in groups[row["image_name"]]]
            for row in rows
        ],
        dtype=np.float32,
    )
    for query_index, row in enumerate(rows):
        order = np.argsort(scores[query_index], kind="stable")
        choice = int(order[0])
        selected = rows[int(candidates[query_index, choice])]
        query_touch = tactile_difference(row["touch_path"], touch_cache, int(cfg["tactile_size"]))
        selected_touch = tactile_difference(selected["touch_path"], touch_cache, int(cfg["tactile_size"]))
        metric = tactile_metrics(query_touch, selected_touch, float(cfg["tactile_mask_threshold"]))
        oracle = int(np.argmin(target_matrix[query_index]))
        prediction = predictions[row["image_name"]]
        output_queries.append(
            {
                "recipe_name": name,
                "query_record_id": row["record_id"],
                "query_image_name": row["image_name"],
                "query_probe": row["probe"],
                "oof_fold": prediction["oof_fold"],
                "pred_x": f"{query_xy[query_index][0]:.3f}",
                "pred_y": f"{query_xy[query_index][1]:.3f}",
                "dino_layer_recipe": recipe["layer_recipe"],
                "canonicalization_mode": recipe["canonicalization_mode"],
                "padding_mode": recipe["padding_mode"],
                "matcher": recipe["matcher"],
                "scales": "|".join(str(value) for value in scales),
                "query_padding_ratio": f"{float(np.mean([part[query_index] for part in padding_parts])):.6f}",
                "selected_cache_record_id": selected["record_id"],
                "selected_cache_image_name": selected["image_name"],
                "ranker_best_score": f"{scores[query_index, choice]:.6f}",
                "ranker_margin": f"{scores[query_index, int(order[1])] - scores[query_index, choice]:.6f}",
                "ranker_oracle_embedding_rank": str(int(ranks(scores[query_index])[oracle])),
                **{key: f"{metric[key]:.6f}" for key in ("tactile_diff_mae", "tactile_ssim", "tactile_mask_iou")},
            }
        )
        oracle_ranks = ranks(target_matrix[query_index])
        dino_ranks = ranks(scores[query_index])
        for rank, item in enumerate(order, start=1):
            item = int(item)
            source = groups[row["image_name"]][item]
            key = (row["image_name"], source["candidate_image_name"])
            labels = dino_labels.get(key, {})
            per_scale = [float(values[query_index, item]) for values in scale_scores]
            output_candidates.append(
                {
                    "recipe_name": name,
                    "query_record_id": row["record_id"],
                    "query_image_name": row["image_name"],
                    "query_probe": row["probe"],
                    "oof_fold": prediction["oof_fold"],
                    "candidate_rank": str(rank),
                    "candidate_score": f"{scores[query_index, item]:.6f}",
                    "detail_patch_score": f"{per_scale[0]:.6f}",
                    "context_patch_score": f"{per_scale[1]:.6f}" if len(per_scale) > 1 else "",
                    "wide_patch_score": f"{per_scale[2]:.6f}" if len(per_scale) > 2 else "",
                    "position_aware_match_score": f"{combined_similarity[query_index, item]:.6f}" if recipe["matcher"] == "position_soft" else "",
                    "candidate_record_id": source["candidate_record_id"],
                    "candidate_image_name": source["candidate_image_name"],
                    "candidate_tactile_embedding_distance": source["candidate_tactile_embedding_distance"],
                    "candidate_tactile_ssim": labels.get("candidate_tactile_ssim", ""),
                    "candidate_tactile_mask_iou": labels.get("candidate_tactile_mask_iou", ""),
                    "candidate_oracle_embedding_rank": str(int(oracle_ranks[item])),
                    "hard_negative_flag": str(int(dino_ranks[item] <= 8 and oracle_ranks[item] >= 17)),
                }
            )
    feature_path = project_path(cfg["feature_cache_pattern"].format(recipe=name))
    ensure_dir(feature_path.parent)
    np.savez_compressed(
        feature_path,
        image_names=np.asarray([row["image_name"] for row in rows]),
        query_features=np.concatenate(pooled_parts, axis=1).astype(np.float32),
        query_padding_ratio=np.stack(padding_parts, axis=1).mean(axis=1).astype(np.float32),
        recipe_json=np.asarray([json.dumps(recipe, sort_keys=True)]),
    )
    report = {
        "recipe": recipe,
        "feature_cache": str(feature_path),
        "summary": summarize(output_queries),
    }
    return output_queries, output_candidates, report


def build(config_path: str, section: str) -> dict:
    cfg = load_config(config_path)[section]
    set_seed(int(cfg["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    partition_path = project_path(cfg["final_partition_csv"])
    all_rows = read_csv_rows(project_path(cfg["samples_csv"]))
    assert_development_only(all_rows, partition_path)
    rows = [row for row in all_rows if row["dataset_split"] == "train"]
    predictions = prediction_map(
        read_csv_rows(project_path(cfg["oof_predictions_csv"])),
        rows,
        "train",
        "Phase4H DINO ablation OOF",
    )
    top_k = int(cfg["geometry_filter_k"])
    groups = candidate_groups(read_csv_rows(project_path(cfg["v1_candidate_csv"])), top_k)
    dino_rows = read_csv_rows(project_path(cfg["dino_candidate_csv"]))
    dino_groups = candidate_groups(dino_rows, top_k)
    assert_candidate_identity(groups, dino_groups)
    if set(groups) != {row["image_name"] for row in rows}:
        raise RuntimeError("Frozen Top-32 groups do not cover every Phase4H development-train query")
    source_by_name = {row["image_name"]: row for row in rows}
    for query, group in groups.items():
        if any(item["candidate_record_id"] == source_by_name[query]["record_id"] for item in group):
            raise RuntimeError(f"Same-record cache candidate found for {query}")
    dino_labels = {
        (row["query_image_name"], row["candidate_image_name"]): row
        for row in dino_rows
    }
    tracks = read_trajectory_tracks(project_path(cfg["motion_tracks_csv"]))
    print(f"phase4h: loading frozen {cfg['dino_model']}", flush=True)
    backbone = FrozenDinoV2(str(cfg["dino_model"]), int(cfg["dino_image_size"])).to(device)
    all_queries, all_candidates, reports = [], [], []
    for recipe in cfg["recipes"]:
        query_rows, candidate_rows, report = run_recipe(
            recipe, cfg, rows, predictions, groups, dino_labels, tracks, backbone, device,
        )
        all_queries.extend(query_rows)
        all_candidates.extend(candidate_rows)
        reports.append(report)
        write_csv_rows(project_path(cfg["query_output_csv"]), all_queries, QUERY_FIELDS)
        write_csv_rows(project_path(cfg["candidate_output_csv"]), all_candidates, CANDIDATE_FIELDS)
        write_json(
            project_path(cfg["metrics_json"]),
            {
                "mode": "phase4h_strict_oof_frozen_dino_ablation_v1",
                "device": str(device),
                "completed_recipes": reports,
                "candidate_set_fingerprint": candidate_set_fingerprint(groups),
                "integrity": {
                    "c2_contact_box": "unchanged",
                    "top32_candidates": "frozen V1 identity",
                    "same_record_candidates": 0,
                    "sealed_final_holdout_rows_read": 0,
                    "query_tactile_usage": "offline evaluation only",
                },
            },
        )
    summary = {
        "mode": "phase4h_strict_oof_frozen_dino_ablation_v1",
        "device": str(device),
        "queries_per_recipe": len(rows),
        "candidate_set_fingerprint": candidate_set_fingerprint(groups),
        "recipes": reports,
        "integrity": {
            "c2_contact_box": "unchanged",
            "top32_candidates": "frozen V1 identity",
            "same_record_candidates": 0,
            "sealed_final_holdout_rows_read": 0,
            "query_tactile_usage": "offline evaluation only",
        },
    }
    write_json(project_path(cfg["metrics_json"]), summary)
    print(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run strict-OOF Phase4H frozen-DINO ablations.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--section", default="phase4h_dino_ablation_oof_v1")
    args = parser.parse_args()
    build(args.config, args.section)


if __name__ == "__main__":
    main()
