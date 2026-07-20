from __future__ import annotations

import argparse
import math
from collections import defaultdict

import numpy as np
import torch

from .build_cache_retrieval import visual_patch_feature
from .config import load_config, project_path
from .temporal_progress import DEFAULT_TTC_VALUES, TRAJECTORY_FEATURE_SIZE, TTCEstimator, motion_basis, read_trajectory_tracks, trajectory_features
from .train_contact_region import parse_probe, ttc_bucket_name
from .utils import read_csv_rows, write_csv_rows, write_json


FIELDS = [
    "dataset_split", "record_id", "image_name", "probe", "ttc_bucket", "estimated_ttc",
    "original_x", "original_y", "reranked_x", "reranked_y", "original_error_px",
    "reranked_error_px", "improved", "selected_rank", "heatmap_score", "candidate_ttc",
    "ttc_inconsistency", "lateral_deviation", "cache_similarity", "retrieved_record_id",
    "retrieved_image_name", "final_score", "cache_miss", "candidate_scores",
]


def parse_points(value: str) -> list[tuple[float, float, float]]:
    points = []
    for item in value.split(";"):
        if not item:
            continue
        x, y, score = item.split(",")
        points.append((float(x), float(y), float(score)))
    return points


def standardize(matrix: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (matrix - mean) / std


def summarize(rows: list[dict[str, str]], error_field: str) -> dict:
    errors = [float(row[error_field]) for row in rows]
    return {
        "samples": len(rows),
        "mean_error_px": float(np.mean(errors)) if errors else None,
        "median_error_px": float(np.median(errors)) if errors else None,
        "pck_16": float(np.mean([error <= 16 for error in errors])) if errors else None,
        "pck_32": float(np.mean([error <= 32 for error in errors])) if errors else None,
        "pck_48": float(np.mean([error <= 48 for error in errors])) if errors else None,
    }


def grouped(rows: list[dict[str, str]], error_field: str) -> dict:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["ttc_bucket"]].append(row)
    return {name: summarize(items, error_field) for name, items in groups.items()}


def rerank(config_path: str, section: str) -> dict:
    config = load_config(config_path)
    cfg = config[section]
    rows = read_csv_rows(project_path(cfg["samples_csv"]))
    rows_by_name = {row["image_name"]: row for row in rows}
    predictions = read_csv_rows(project_path(cfg["predictions_csv"]))
    tracks = read_trajectory_tracks(project_path(cfg["motion_tracks_csv"]))
    ttc_values = np.asarray(cfg.get("ttc_values", DEFAULT_TTC_VALUES), dtype=np.float32)
    history_frames = int(cfg.get("history_frames", 32))
    spatial_scale = float(cfg.get("spatial_scale_px", 48.0))
    speed_scale = float(cfg.get("speed_scale_px", 4.0))
    crop_size = int(cfg.get("cache_crop_size", 48))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(project_path(cfg["ttc_checkpoint"]), map_location=device, weights_only=False)
    estimator = TTCEstimator(TRAJECTORY_FEATURE_SIZE, int(cfg.get("hidden_size", 64)), len(ttc_values)).to(device)
    estimator.load_state_dict(checkpoint["model"])
    estimator.eval()

    cache_sources = [rows_by_name[pred["image_name"]] for pred in predictions if pred["dataset_split"] == "train"]
    cache_features = np.stack([
        visual_patch_feature(source["vision_path"], float(source["target_tip_x"]), float(source["target_tip_y"]), crop_size)
        for source in cache_sources
    ])
    cache_mean = cache_features.mean(axis=0)
    cache_std = cache_features.std(axis=0)
    cache_std[cache_std < 1e-6] = 1.0
    cache_z = standardize(cache_features, cache_mean, cache_std)
    feature_norm = math.sqrt(cache_z.shape[1])

    alpha = float(cfg.get("ttc_weight", 0.8))
    beta = float(cfg.get("lateral_weight", 0.5))
    gamma = float(cfg.get("cache_similarity_weight", 0.4))
    ttc_normalizer = float(cfg.get("ttc_normalizer", 100.0))
    lateral_normalizer = float(cfg.get("lateral_normalizer_px", 48.0))
    cache_miss_similarity = float(cfg.get("cache_miss_similarity", 0.15))
    cache_miss_ttc = float(cfg.get("cache_miss_ttc_frames", 60.0))
    buckets = {"near": {5, 10, 20}, "mid": {30, 50}, "far": {75, 100}}
    output_rows = []

    for pred in predictions:
        if pred["dataset_split"] not in {"val", "test"}:
            continue
        source = rows_by_name[pred["image_name"]]
        features = trajectory_features(source, tracks, history_frames, spatial_scale, speed_scale)
        trajectory = torch.from_numpy(features[None]).to(device)
        with torch.no_grad():
            probabilities = torch.softmax(estimator(trajectory)["ttc_logits"], dim=1)[0].cpu().numpy()
        estimated_ttc = float(np.sum(probabilities * ttc_values))
        direction, normalized_speed = motion_basis(features)
        speed_px_per_frame = normalized_speed * speed_scale
        tip = np.asarray([float(source["tip_x"]), float(source["tip_y"])], dtype=np.float32)
        target = np.asarray([float(source["target_tip_x"]), float(source["target_tip_y"])], dtype=np.float32)
        candidate_results = []
        for rank, (x, y, heatmap_score) in enumerate(parse_points(pred["topk_points"]), start=1):
            vector = np.asarray([x, y], dtype=np.float32) - tip
            if speed_px_per_frame > 1e-6 and float(np.linalg.norm(direction)) > 0.0:
                longitudinal = float(np.dot(vector, direction))
                lateral = abs(float(vector[0] * (-direction[1]) + vector[1] * direction[0]))
                candidate_ttc = max(longitudinal / speed_px_per_frame, 0.0)
            else:
                lateral = float(np.linalg.norm(vector))
                candidate_ttc = 0.0
            ttc_inconsistency = abs(candidate_ttc - estimated_ttc)
            visual = visual_patch_feature(source["vision_path"], x, y, crop_size)
            visual_z = standardize(visual[None], cache_mean, cache_std)[0]
            distances = np.linalg.norm(cache_z - visual_z[None], axis=1)
            cache_index = int(np.argmin(distances))
            cache_similarity = math.exp(-float(distances[cache_index]) / max(feature_norm, 1.0))
            final_score = heatmap_score - alpha * ttc_inconsistency / ttc_normalizer - beta * lateral / lateral_normalizer + gamma * cache_similarity
            candidate_results.append({
                "rank": rank, "x": x, "y": y, "heatmap_score": heatmap_score,
                "candidate_ttc": candidate_ttc, "ttc_inconsistency": ttc_inconsistency,
                "lateral": lateral, "cache_similarity": cache_similarity,
                "cache_index": cache_index, "final_score": final_score,
            })
        selected = max(candidate_results, key=lambda item: item["final_score"])
        retrieved = cache_sources[selected["cache_index"]]
        original = candidate_results[0]
        original_error = float(np.linalg.norm(np.asarray([original["x"], original["y"]]) - target))
        reranked_error = float(np.linalg.norm(np.asarray([selected["x"], selected["y"]]) - target))
        cache_miss = selected["cache_similarity"] < cache_miss_similarity or selected["ttc_inconsistency"] > cache_miss_ttc
        output_rows.append({
            "dataset_split": pred["dataset_split"], "record_id": source["record_id"], "image_name": source["image_name"],
            "probe": str(parse_probe(source) or ""), "ttc_bucket": ttc_bucket_name(parse_probe(source), buckets),
            "estimated_ttc": f"{estimated_ttc:.3f}", "original_x": f"{original['x']:.3f}", "original_y": f"{original['y']:.3f}",
            "reranked_x": f"{selected['x']:.3f}", "reranked_y": f"{selected['y']:.3f}",
            "original_error_px": f"{original_error:.3f}", "reranked_error_px": f"{reranked_error:.3f}",
            "improved": "1" if reranked_error < original_error else "0", "selected_rank": str(selected["rank"]),
            "heatmap_score": f"{selected['heatmap_score']:.6f}", "candidate_ttc": f"{selected['candidate_ttc']:.3f}",
            "ttc_inconsistency": f"{selected['ttc_inconsistency']:.3f}", "lateral_deviation": f"{selected['lateral']:.3f}",
            "cache_similarity": f"{selected['cache_similarity']:.6f}", "retrieved_record_id": retrieved["record_id"],
            "retrieved_image_name": retrieved["image_name"], "final_score": f"{selected['final_score']:.6f}",
            "cache_miss": "1" if cache_miss else "0",
            "candidate_scores": ";".join(f"{item['rank']}:{item['final_score']:.5f}" for item in candidate_results),
        })

    test_rows = [row for row in output_rows if row["dataset_split"] == "test"]
    before = summarize(test_rows, "original_error_px")
    after = summarize(test_rows, "reranked_error_px")
    before_buckets = grouped(test_rows, "original_error_px")
    after_buckets = grouped(test_rows, "reranked_error_px")
    oracle_far = float(cfg.get("oracle_far_median_error_px", 16.97))
    baseline_far = before_buckets.get("far", {}).get("median_error_px")
    reranked_far = after_buckets.get("far", {}).get("median_error_px")
    denominator = baseline_far - oracle_far if baseline_far is not None else 0.0
    recovery = (baseline_far - reranked_far) / denominator if denominator > 1e-6 else None
    summary = {
        "device": str(device), "queries": len(output_rows), "test_queries": len(test_rows),
        "weights": {"heatmap": 1.0, "ttc": alpha, "lateral": beta, "cache_similarity": gamma},
        "before": before, "after": after, "before_by_ttc_bucket": before_buckets, "after_by_ttc_bucket": after_buckets,
        "far_oracle_improvement_recovery": recovery,
        "selected_non_top1_rate": float(np.mean([row["selected_rank"] != "1" for row in test_rows])) if test_rows else None,
        "improved_rate": float(np.mean([row["improved"] == "1" for row in test_rows])) if test_rows else None,
        "cache_miss_rate": float(np.mean([row["cache_miss"] == "1" for row in test_rows])) if test_rows else None,
    }
    write_csv_rows(project_path(cfg["output_csv"]), output_rows, FIELDS)
    write_json(project_path(cfg["metrics_json"]), summary)
    print(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Physically rerank Top-K contact proposals with predicted TTC and cache similarity.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--section", default="proposal_reranking")
    args = parser.parse_args()
    rerank(args.config, args.section)


if __name__ == "__main__":
    main()
