from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .build_cache_retrieval import motion_geometry_feature, standardize
from .config import load_config, project_path
from .evaluate_oracle_tactile_retrieval import tactile_difference, tactile_embedding, tactile_metrics
from .train_soft_tactile_cache_ranker import SoftTactileRanker, build_groups, load_patches, predict, ranks
from .utils import ensure_dir, read_csv_rows, write_csv_rows, write_json


METRICS = [
    "tactile_diff_mae", "tactile_ssim", "tactile_mask_iou", "tactile_area_delta",
    "tactile_centroid_distance", "tactile_embedding_distance",
]
GRID_FIELDS = [
    "quantile", "advantage_threshold", "gate_coverage", "changed_queries", "gated_queries",
    "mean_tactile_diff_mae", "mean_tactile_ssim", "mean_tactile_mask_iou", "mean_tactile_embedding_distance",
    "far_mean_tactile_diff_mae", "far_mean_tactile_ssim", "far_mean_tactile_mask_iou", "eligible",
]
QUERY_FIELDS = [
    "query_record_id", "query_image_name", "query_probe", "current_cache_record_id", "ranker_cache_record_id",
    "gated_cache_record_id", "ranker_advantage", "ranker_margin", "gate_applied", "current_rank_of_tactile_best",
    "ranker_rank_of_tactile_best", "gated_rank_of_tactile_best",
    *[f"{prefix}_{metric}" for prefix in ("current", "ranker", "gated") for metric in METRICS],
]


def record_number(record_id: str) -> int:
    try:
        return int(record_id.rsplit("_", 1)[-1])
    except ValueError as exc:
        raise ValueError(f"Unexpected record ID: {record_id}") from exc


def summarize(rows: list[dict[str, str]], prefix: str) -> dict[str, float | int | None]:
    result: dict[str, float | int | None] = {"queries": len(rows)}
    for metric in METRICS:
        values = np.asarray([float(row[f"{prefix}_{metric}"]) for row in rows], dtype=np.float32)
        result[f"mean_{metric}"] = float(values.mean()) if len(values) else None
        result[f"median_{metric}"] = float(np.median(values)) if len(values) else None
    ranks = np.asarray([int(row[f"{prefix}_rank_of_tactile_best"]) for row in rows], dtype=np.float32)
    result["tactile_best_top1_rate"] = float(np.mean(ranks == 1)) if len(ranks) else None
    result["tactile_best_top5_rate"] = float(np.mean(ranks <= 5)) if len(ranks) else None
    result["median_tactile_best_rank"] = float(np.median(ranks)) if len(ranks) else None
    return result


def choice_metrics(
    query_diff: np.ndarray,
    candidate_index: int,
    candidates: np.ndarray,
    cache_rows: list[dict[str, str]],
    diff_cache: dict[str, np.ndarray],
    tactile_size: int,
    tactile_threshold: float,
) -> tuple[dict[str, float], dict[str, str]]:
    cache = cache_rows[int(candidates[candidate_index])]
    return tactile_metrics(query_diff, tactile_difference(cache["touch_path"], diff_cache, tactile_size), tactile_threshold), cache


def evaluate(config_path: str, section: str) -> dict:
    cfg = load_config(config_path)[section]
    rows = read_csv_rows(project_path(cfg["samples_csv"]))
    final_min_record = int(cfg.get("final_holdout_min_record", 950))
    forbidden = sorted({row["record_id"] for row in rows if record_number(row["record_id"]) >= final_min_record})
    if forbidden:
        raise RuntimeError(f"Refusing to access final-holdout records: {forbidden[:5]}")
    cache_rows = [row for row in rows if row["dataset_split"] == "train"]
    val_rows = [row for row in rows if row["dataset_split"] == "val"]
    overlap = {row["record_id"] for row in cache_rows} & {row["record_id"] for row in val_rows}
    if overlap:
        raise RuntimeError(f"Train and validation records overlap: {sorted(overlap)[:5]}")
    crop_size, tactile_size = int(cfg.get("cache_crop_size", 48)), int(cfg.get("tactile_size", 96))
    filter_k = min(int(cfg.get("geometry_filter_k", 32)), len(cache_rows))
    raw_cache_geometry = np.stack([motion_geometry_feature(row, float(row["target_tip_x"]), float(row["target_tip_y"])) for row in cache_rows])
    cache_geometry, geometry_mean, geometry_std = standardize(raw_cache_geometry, raw_cache_geometry)
    cache_patches, _, cache_hand_raw = load_patches(cache_rows, crop_size, geometry_mean, geometry_std)
    val_patches, val_geometry, val_hand_raw = load_patches(val_rows, crop_size, geometry_mean, geometry_std)
    cache_hand, hand_mean, hand_std = standardize(cache_hand_raw, cache_hand_raw)
    val_hand = (val_hand_raw - hand_mean) / hand_std
    cache_geometry = cache_geometry.astype(np.float32)
    diff_cache: dict[str, np.ndarray] = {}
    record_tactile: dict[str, np.ndarray] = {}
    for row in cache_rows:
        if row["record_id"] not in record_tactile:
            record_tactile[row["record_id"]] = tactile_difference(row["touch_path"], diff_cache, tactile_size)
    cache_tactile = np.stack([tactile_embedding(record_tactile[row["record_id"]]) for row in cache_rows]).astype(np.float32)
    val_tactile_images = np.stack([tactile_difference(row["touch_path"], diff_cache, tactile_size) for row in val_rows]).astype(np.float32)
    val_tactile = np.stack([tactile_embedding(image) for image in val_tactile_images]).astype(np.float32)
    diff_cache.clear()
    groups = build_groups(val_rows, val_geometry, val_hand, val_tactile, cache_rows, cache_geometry, cache_hand, cache_tactile, filter_k, False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = project_path(cfg["checkpoint"])
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = SoftTactileRanker(cache_geometry.shape[1], float(cfg.get("dropout", 0.1))).to(device)
    model.load_state_dict(checkpoint["model_state"])
    scores = predict(model, groups, val_patches, val_geometry, cache_patches, cache_geometry, device, int(cfg.get("batch_size", 16)))
    current_choices = np.argmin(groups.current_scores, axis=1)
    ranker_choices = np.argmin(scores, axis=1)
    advantages = scores[np.arange(len(scores)), current_choices] - scores[np.arange(len(scores)), ranker_choices]
    changed = ranker_choices != current_choices
    changed_advantages = advantages[changed]
    quantiles = [float(value) for value in cfg.get("gate_quantiles", [0.0, 0.25, 0.5, 0.75, 0.9])]
    thresholds = [float(np.quantile(changed_advantages, quantile)) if len(changed_advantages) else float("inf") for quantile in quantiles]
    tactile_threshold = float(cfg.get("tactile_mask_threshold", 0.04))

    # Tactile metrics are calculated lazily for selections that a gate can actually return.
    metric_cache: dict[str, np.ndarray] = {}
    selection_metrics: list[dict[int, tuple[dict[str, float], dict[str, str]]]] = [dict() for _ in val_rows]

    def selected_metrics(index: int, local_index: int) -> tuple[dict[str, float], dict[str, str]]:
        if local_index not in selection_metrics[index]:
            selection_metrics[index][local_index] = choice_metrics(
                val_tactile_images[index], local_index, groups.candidates[index], cache_rows,
                metric_cache, tactile_size, tactile_threshold,
            )
        return selection_metrics[index][local_index]

    def rows_for_threshold(threshold: float) -> tuple[list[dict[str, str]], np.ndarray]:
        use_ranker = changed & (advantages >= threshold)
        output: list[dict[str, str]] = []
        for index, query in enumerate(val_rows):
            current = int(current_choices[index])
            ranker = int(ranker_choices[index])
            gated = ranker if use_ranker[index] else current
            tactile_best = int(np.argmin(groups.targets[index]))
            selections = {"current": current, "ranker": ranker, "gated": gated}
            selected = {name: selected_metrics(index, choice) for name, choice in selections.items()}
            output.append({
                "query_record_id": query["record_id"], "query_image_name": query["image_name"], "query_probe": query["probe"],
                "current_cache_record_id": selected["current"][1]["record_id"], "ranker_cache_record_id": selected["ranker"][1]["record_id"],
                "gated_cache_record_id": selected["gated"][1]["record_id"], "ranker_advantage": f"{advantages[index]:.6f}",
                "ranker_margin": f"{np.partition(scores[index], 1)[1] - np.min(scores[index]):.6f}", "gate_applied": str(int(use_ranker[index])),
                "current_rank_of_tactile_best": str(int(ranks(groups.current_scores[index])[tactile_best])),
                "ranker_rank_of_tactile_best": str(int(ranks(scores[index])[tactile_best])),
                "gated_rank_of_tactile_best": str(int(ranks(scores[index] if use_ranker[index] else groups.current_scores[index])[tactile_best])),
                **{f"{name}_{metric}": f"{selected[name][0][metric]:.6f}" for name in selections for metric in METRICS},
            })
        return output, use_ranker

    baseline_rows, _ = rows_for_threshold(float("inf"))
    baseline = summarize(baseline_rows, "current")
    grid_rows: list[dict[str, str]] = []
    candidates: list[tuple[float, float, int, list[dict[str, str]], np.ndarray, bool]] = []
    min_coverage = float(cfg.get("minimum_gate_coverage", 0.05))
    for grid_index, (quantile, threshold) in enumerate(zip(quantiles, thresholds)):
        gated_rows, gate = rows_for_threshold(threshold)
        gated = summarize(gated_rows, "gated")
        far = summarize([row for row in gated_rows if int(row["query_probe"]) >= 75], "gated")
        baseline_far = summarize([row for row in baseline_rows if int(row["query_probe"]) >= 75], "current")
        coverage = float(gate.mean())
        eligible = (
            coverage >= min_coverage
            and gated["mean_tactile_diff_mae"] <= baseline["mean_tactile_diff_mae"]
            and gated["mean_tactile_ssim"] >= baseline["mean_tactile_ssim"]
            and gated["mean_tactile_mask_iou"] >= baseline["mean_tactile_mask_iou"]
            and far["mean_tactile_diff_mae"] <= baseline_far["mean_tactile_diff_mae"]
            and far["mean_tactile_mask_iou"] >= baseline_far["mean_tactile_mask_iou"]
        )
        grid_rows.append({
            "quantile": f"{quantile:.3f}", "advantage_threshold": f"{threshold:.6f}", "gate_coverage": f"{coverage:.6f}",
            "changed_queries": str(int(changed.sum())), "gated_queries": str(int(gate.sum())),
            "mean_tactile_diff_mae": f"{gated['mean_tactile_diff_mae']:.6f}", "mean_tactile_ssim": f"{gated['mean_tactile_ssim']:.6f}",
            "mean_tactile_mask_iou": f"{gated['mean_tactile_mask_iou']:.6f}", "mean_tactile_embedding_distance": f"{gated['mean_tactile_embedding_distance']:.6f}",
            "far_mean_tactile_diff_mae": f"{far['mean_tactile_diff_mae']:.6f}", "far_mean_tactile_ssim": f"{far['mean_tactile_ssim']:.6f}",
            "far_mean_tactile_mask_iou": f"{far['mean_tactile_mask_iou']:.6f}", "eligible": str(int(eligible)),
        })
        candidates.append((float(gated["mean_tactile_ssim"]), coverage, grid_index, gated_rows, gate, eligible))
    eligible = [item for item in candidates if item[5]]
    selected = max(eligible, key=lambda item: (item[0], item[1])) if eligible else None
    if selected is None:
        selected_rows, selected_gate, selected_grid = baseline_rows, np.zeros(len(baseline_rows), dtype=bool), None
    else:
        selected_rows, selected_gate = selected[3], selected[4]
        selected_grid = grid_rows[selected[2]]
    output_csv = project_path(cfg["output_csv"])
    grid_csv = project_path(cfg["grid_csv"])
    write_csv_rows(output_csv, selected_rows, QUERY_FIELDS)
    write_csv_rows(grid_csv, grid_rows, GRID_FIELDS)
    summary = {
        "mode": "validation_only_soft_tactile_cache_confidence_gate", "device": str(device), "cache_size": len(cache_rows),
        "validation_queries": len(val_rows), "geometry_filter_k": filter_k, "changed_queries": int(changed.sum()),
        "selection_policy": "maximize validation SSIM subject to non-worse overall MAE/SSIM/IoU, non-worse far MAE/IoU, and minimum gate coverage",
        "minimum_gate_coverage": min_coverage, "selected_gate": selected_grid, "baseline": baseline,
        "gated": summarize(selected_rows, "gated"), "far_gated": summarize([row for row in selected_rows if int(row["query_probe"]) >= 75], "gated"),
        "final_holdout_min_record": final_min_record, "checkpoint": str(checkpoint_path),
    }
    write_json(project_path(cfg["metrics_json"]), summary)
    print(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Select a validation-only confidence gate for the soft tactile cache ranker.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--section", default="soft_tactile_cache_gate_phase35_v3")
    args = parser.parse_args()
    evaluate(args.config, args.section)


if __name__ == "__main__":
    main()
