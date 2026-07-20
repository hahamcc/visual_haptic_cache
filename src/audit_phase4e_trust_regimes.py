from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np

from .config import load_config, project_path
from .train_phase4b_predicted_box_cache_ranker import is_final_holdout
from .train_phase4e_cache_trust import candidate_features, finite
from .utils import read_csv_rows, write_csv_rows, write_json


OUTCOME_FIELDS = ("tactile_diff_mae", "tactile_ssim", "tactile_mask_iou")
AUDIT_FIELDS = [
    "query_record_id", "query_image_name", "regime", "probe", "c2_error_px", "c2_box48_hit",
    "ranker_best_score", "ranker_margin_normalized", "hand_margin", "top3_tactile_embedding_disagreement",
    "geometry_distance", "detail_visual_distance", "context_visual_distance", "c2_pred_score",
    "candidate_score_top5_mean", "candidate_score_top5_std", "candidate_score_all_std",
    "candidate_ranker_hand_disagreement", "candidate_ranker_hand_rank_gap", "ranker_oracle_embedding_rank",
    *OUTCOME_FIELDS,
]


def regime(probe: int) -> str:
    if probe <= 30:
        return "near"
    if probe == 50:
        return "mid"
    return "far"


def average_rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def spearman(left: list[float], right: list[float]) -> float | None:
    if len(left) < 3:
        return None
    x, y = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None
    return float(np.corrcoef(average_rank(x), average_rank(y))[0, 1])


def group_summary(rows: list[dict[str, str]], signal_names: list[str]) -> dict:
    result = {"queries": len(rows)}
    for outcome in OUTCOME_FIELDS:
        values = np.asarray([finite(row[outcome]) for row in rows], dtype=np.float32)
        result[f"mean_{outcome}"] = float(values.mean()) if len(values) else None
    result["c2_box48_hit_rate"] = float(np.mean([finite(row["c2_box48_hit"]) for row in rows])) if rows else None
    result["ranker_tactile_best_top1_rate"] = float(np.mean([int(row["ranker_oracle_embedding_rank"]) == 1 for row in rows])) if rows else None
    result["ranker_tactile_best_top3_rate"] = float(np.mean([int(row["ranker_oracle_embedding_rank"]) <= 3 for row in rows])) if rows else None
    correlations = {}
    for signal in signal_names:
        values = [finite(row[signal]) for row in rows]
        correlations[signal] = {
            "negative_mae": spearman(values, [-finite(row["tactile_diff_mae"]) for row in rows]),
            "ssim": spearman(values, [finite(row["tactile_ssim"]) for row in rows]),
            "iou": spearman(values, [finite(row["tactile_mask_iou"]) for row in rows]),
        }
    result["online_signal_spearman"] = correlations
    return result


def audit(config_path: str, section: str) -> dict:
    cfg = load_config(config_path)[section]
    samples = read_csv_rows(project_path(cfg["samples_csv"]))
    sample_by_name = {row["image_name"]: row for row in samples}
    queries = read_csv_rows(project_path(cfg["oof_query_csv"]))
    candidates = read_csv_rows(project_path(cfg["oof_candidate_csv"]))
    c2_predictions = read_csv_rows(project_path(cfg["oof_predictions_csv"]))
    c2_by_name = {row["image_name"]: row for row in c2_predictions if row.get("dataset_split") == "train"}
    candidate_by_name = candidate_features(candidates)
    if len(queries) != len({row["query_image_name"] for row in queries}):
        raise RuntimeError("Expected one strict OOF cache query row per image.")
    rows = []
    for query in queries:
        source = sample_by_name.get(query["query_image_name"])
        if source is None or source["dataset_split"] != "train" or is_final_holdout(source):
            raise RuntimeError("Audit may only read strict OOF development-train records, never final holdout.")
        c2 = c2_by_name.get(query["query_image_name"])
        aggregate = candidate_by_name.get(query["query_image_name"])
        if c2 is None or aggregate is None:
            raise RuntimeError(f"Missing OOF C2 or candidate signal for {query['query_image_name']}")
        rows.append({
            "query_record_id": query["query_record_id"], "query_image_name": query["query_image_name"],
            "regime": regime(int(query["query_probe"])), "probe": query["query_probe"], "c2_error_px": c2["error_px"], "c2_box48_hit": c2["box48_hit"],
            **{name: query[name] for name in ("ranker_best_score", "ranker_margin_normalized", "hand_margin", "top3_tactile_embedding_disagreement", "geometry_distance", "detail_visual_distance", "context_visual_distance", "c2_pred_score", "ranker_oracle_embedding_rank", *OUTCOME_FIELDS)},
            **{name: f"{aggregate[name]:.6f}" for name in ("candidate_score_top5_mean", "candidate_score_top5_std", "candidate_score_all_std", "candidate_ranker_hand_disagreement", "candidate_ranker_hand_rank_gap")},
        })
    write_csv_rows(project_path(cfg["output_csv"]), rows, AUDIT_FIELDS)
    signal_names = [
        "ranker_best_score", "ranker_margin_normalized", "hand_margin", "top3_tactile_embedding_disagreement",
        "geometry_distance", "detail_visual_distance", "context_visual_distance", "c2_pred_score",
        "candidate_score_top5_mean", "candidate_score_top5_std", "candidate_score_all_std",
        "candidate_ranker_hand_disagreement", "candidate_ranker_hand_rank_gap",
    ]
    grouped = defaultdict(list)
    grouped["all"] = rows
    for row in rows:
        grouped[row["regime"]].append(row)
    summary = {
        "mode": "phase4e_strict_oof_trust_regime_audit", "queries": len(rows),
        "regimes": {name: group_summary(items, signal_names) for name, items in grouped.items()},
        "integrity": {"cache_ranker": "strict Phase4E OOF", "c2": "strict record-level OOF", "sealed_final_holdout_rows_read": 0},
        "signal_orientation": {
            "ranker_best_score": "lower is preferred", "top3_tactile_embedding_disagreement": "lower means cached Top-3 tactile states agree",
            "geometry_distance": "lower means closer motion/contact geometry", "c2_pred_score": "higher means sharper C2 heatmap confidence",
        },
    }
    write_json(project_path(cfg["metrics_json"]), summary)
    print(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose why Phase4E cache trust differs by anticipation regime.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--section", default="phase4e_trust_regime_audit_v1")
    args = parser.parse_args()
    audit(args.config, args.section)


if __name__ == "__main__":
    main()
