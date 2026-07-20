from __future__ import annotations

import argparse
import math

import numpy as np
import torch

from .build_cache_retrieval import crop_contact_patch, motion_geometry_feature, standardize, visual_patch_feature_from_patch
from .config import load_config, project_path
from .train_soft_tactile_cache_ranker import CandidateGroups, SoftTactileRanker, predict, ranks
from .utils import read_csv_rows, write_csv_rows, write_json


FIELDS = [
    "query_record_id", "query_image_name", "query_probe", "oof_fold", "query_case", "candidate_rank",
    "candidate_x", "candidate_y", "heatmap_score", "heatmap_ratio", "candidate_box48_hit", "candidate_rel_tip_x",
    "candidate_rel_tip_y", "candidate_direction_projection", "candidate_lateral_offset", "cache_ranker_score",
    "cache_ranker_score_normalized", "cache_score_rank_within_top10", "soft_selects_candidate",
    "retrieved_cache_record_id", "retrieved_cache_image_name",
]


def record_number(record_id: str) -> int:
    try:
        return int(record_id.rsplit("_", 1)[-1])
    except ValueError as exc:
        raise ValueError(f"Unexpected record ID: {record_id}") from exc


def parse_points(value: str, topk: int) -> list[tuple[float, float, float]]:
    return [tuple(map(float, item.split(","))) for item in value.split(";")[:topk] if item]


def box48_hit(x: float, y: float, row: dict[str, str]) -> bool:
    return abs(x - float(row["target_tip_x"])) <= 24.0 and abs(y - float(row["target_tip_y"])) <= 24.0


def build(config_path: str, section: str) -> dict:
    cfg = load_config(config_path)[section]
    rows = read_csv_rows(project_path(cfg["samples_csv"]))
    predictions = read_csv_rows(project_path(cfg["oof_predictions_csv"]))
    final_min_record = int(cfg.get("final_holdout_min_record", 950))
    forbidden = sorted({row["record_id"] for row in rows if record_number(row["record_id"]) >= final_min_record})
    if forbidden:
        raise RuntimeError(f"Refusing to access final-holdout records: {forbidden[:5]}")
    cache_rows = [row for row in rows if row["dataset_split"] == "train"]
    rows_by_name = {row["image_name"]: row for row in cache_rows}
    oof_predictions = [row for row in predictions if row["dataset_split"] == "train" and row["image_name"] in rows_by_name]
    if len(oof_predictions) != len(cache_rows):
        raise RuntimeError(f"OOF predictions must cover all train queries, got {len(oof_predictions)} for {len(cache_rows)}.")
    topk = int(cfg.get("topk", 10))
    crop_size, filter_k = int(cfg.get("cache_crop_size", 48)), min(int(cfg.get("geometry_filter_k", 32)), len(cache_rows))
    raw_cache_geometry = np.stack([motion_geometry_feature(row, float(row["target_tip_x"]), float(row["target_tip_y"])) for row in cache_rows])
    cache_geometry, geometry_mean, geometry_std = standardize(raw_cache_geometry, raw_cache_geometry)
    cache_geometry = cache_geometry.astype(np.float32)
    cache_patches = np.stack([crop_contact_patch(row["vision_path"], float(row["target_tip_x"]), float(row["target_tip_y"]), crop_size) for row in cache_rows]).astype(np.float32)
    cache_hand_raw = np.stack([visual_patch_feature_from_patch(patch) for patch in cache_patches])
    cache_hand, hand_mean, hand_std = standardize(cache_hand_raw, cache_hand_raw)
    expanded_rows, patches, geometry, hand, meta = [], [], [], [], []
    for query_index, prediction in enumerate(oof_predictions):
        row = rows_by_name[prediction["image_name"]]
        points = parse_points(prediction["topk_points"], topk)
        if len(points) != topk:
            raise RuntimeError(f"Expected {topk} OOF candidates for {row['image_name']}, got {len(points)}.")
        top_score = max(points[0][2], 1e-8)
        for rank, (x, y, heat_score) in enumerate(points, start=1):
            patch = crop_contact_patch(row["vision_path"], x, y, crop_size)
            expanded_rows.append(row)
            patches.append(patch)
            geometry.append((motion_geometry_feature(row, x, y) - geometry_mean) / geometry_std)
            hand.append((visual_patch_feature_from_patch(patch) - hand_mean) / hand_std)
            tip_x, tip_y = float(row["tip_x"]), float(row["tip_y"])
            direction_x, direction_y = float(row["direction_x"]), float(row["direction_y"])
            dx, dy = x - tip_x, y - tip_y
            meta.append({
                "query_index": query_index, "rank": rank, "x": x, "y": y, "heat_score": heat_score, "heat_ratio": heat_score / top_score,
                "box48": box48_hit(x, y, row), "rel_tip_x": dx / float(row["image_width"]), "rel_tip_y": dy / float(row["image_height"]),
                "direction_projection": (dx * direction_x + dy * direction_y) / max(float(row["image_width"]), float(row["image_height"])),
                "lateral_offset": (-dx * direction_y + dy * direction_x) / max(float(row["image_width"]), float(row["image_height"])),
            })
    patches = np.stack(patches).astype(np.float32)
    geometry = np.stack(geometry).astype(np.float32)
    hand = np.stack(hand).astype(np.float32)
    candidates, current_scores = [], []
    for index, row in enumerate(expanded_rows):
        distances = np.linalg.norm(cache_geometry - geometry[index][None], axis=1)
        allowed = np.asarray([cache["record_id"] != row["record_id"] for cache in cache_rows])
        allowed_indices = np.flatnonzero(allowed)
        shortlist = allowed_indices[np.argpartition(distances[allowed_indices], filter_k - 1)[:filter_k]]
        shortlist = shortlist[np.argsort(distances[shortlist], kind="stable")]
        visual = np.linalg.norm(cache_hand[shortlist] - hand[index][None], axis=1)
        current = distances[shortlist] / math.sqrt(cache_geometry.shape[1]) + visual / math.sqrt(cache_hand.shape[1])
        candidates.append(shortlist.astype(np.int32))
        current_scores.append(current.astype(np.float32))
    groups = CandidateGroups(np.stack(candidates), np.zeros((len(candidates), filter_k), dtype=np.float32), np.stack(current_scores))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = project_path(cfg["checkpoint"])
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = SoftTactileRanker(cache_geometry.shape[1], float(cfg.get("dropout", 0.1))).to(device)
    model.load_state_dict(checkpoint["model_state"])
    scores = predict(model, groups, patches, geometry, cache_patches, cache_geometry, device, int(cfg.get("batch_size", 16)))
    cache_local = np.argmin(scores, axis=1)
    cache_scores = scores[np.arange(len(scores)), cache_local]
    per_query: list[list[int]] = [[] for _ in oof_predictions]
    for expanded_index, item in enumerate(meta):
        per_query[item["query_index"]].append(expanded_index)
    output_rows: list[dict[str, str]] = []
    query_cases, rank_hard_soft_hits, rank_hard_positive_ranks = [], [], []
    for indices in per_query:
        hits = [meta[index]["box48"] for index in indices]
        query_case = "easy" if hits[0] else ("rank_hard" if any(hits) else "proposal_miss")
        query_cases.append(query_case)
        raw_scores = cache_scores[indices]
        rank_values = ranks(raw_scores)
        score_scale = max(float(raw_scores.max() - raw_scores.min()), 1e-6)
        normalized = (raw_scores - raw_scores.min()) / score_scale
        selected_local = int(np.argmin(raw_scores))
        if query_case == "rank_hard":
            positive_local = [index for index, hit in enumerate(hits) if hit]
            rank_hard_soft_hits.append(bool(hits[selected_local]))
            rank_hard_positive_ranks.append(min(int(rank_values[index]) for index in positive_local))
        for local_index, expanded_index in enumerate(indices):
            item, row = meta[expanded_index], expanded_rows[expanded_index]
            cache = cache_rows[int(groups.candidates[expanded_index, cache_local[expanded_index]])]
            output_rows.append({
                "query_record_id": row["record_id"], "query_image_name": row["image_name"], "query_probe": row["probe"], "oof_fold": "unknown",
                "query_case": query_case, "candidate_rank": str(item["rank"]), "candidate_x": f"{item['x']:.3f}", "candidate_y": f"{item['y']:.3f}",
                "heatmap_score": f"{item['heat_score']:.6f}", "heatmap_ratio": f"{item['heat_ratio']:.6f}", "candidate_box48_hit": str(int(item["box48"])),
                "candidate_rel_tip_x": f"{item['rel_tip_x']:.6f}", "candidate_rel_tip_y": f"{item['rel_tip_y']:.6f}",
                "candidate_direction_projection": f"{item['direction_projection']:.6f}", "candidate_lateral_offset": f"{item['lateral_offset']:.6f}",
                "cache_ranker_score": f"{cache_scores[expanded_index]:.6f}", "cache_ranker_score_normalized": f"{normalized[local_index]:.6f}",
                "cache_score_rank_within_top10": str(int(rank_values[local_index])), "soft_selects_candidate": str(int(local_index == selected_local)),
                "retrieved_cache_record_id": cache["record_id"], "retrieved_cache_image_name": cache["image_name"],
            })
    summary = {
        "mode": "oof_topk_cache_calibration_supervision", "device": str(device), "queries": len(oof_predictions), "candidates": len(output_rows),
        "topk": topk, "geometry_filter_k": filter_k, "query_case_counts": {case: query_cases.count(case) for case in ("easy", "rank_hard", "proposal_miss")},
        "rank_hard_soft_selects_box48_rate": float(np.mean(rank_hard_soft_hits)) if rank_hard_soft_hits else None,
        "rank_hard_best_positive_cache_score_median_rank": float(np.median(rank_hard_positive_ranks)) if rank_hard_positive_ranks else None,
        "rank_hard_best_positive_cache_score_top5_rate": float(np.mean(np.asarray(rank_hard_positive_ranks) <= 5)) if rank_hard_positive_ranks else None,
        "final_holdout_min_record": final_min_record, "checkpoint": str(checkpoint_path),
        "note": "C2 candidates are strict record-level OOF. Frozen cache scores are calibration features; same-record cache entries are excluded.",
    }
    write_csv_rows(project_path(cfg["output_csv"]), output_rows, FIELDS)
    write_json(project_path(cfg["metrics_json"]), summary)
    print(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build OOF Top-K cache-score supervision for candidate calibration.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--section", default="oof_topk_cache_calibration_phase35_v3")
    args = parser.parse_args()
    build(args.config, args.section)


if __name__ == "__main__":
    main()
