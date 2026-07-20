from __future__ import annotations

import argparse
import itertools
import math
from collections import defaultdict

import numpy as np

from .build_cache_retrieval import visual_patch_feature
from .config import load_config, project_path
from .temporal_progress import motion_basis, read_trajectory_tracks, trajectory_features
from .utils import read_csv_rows, write_csv_rows, write_json


FIELDS = [
    "strategy", "dataset_split", "record_id", "image_name", "probe", "ttc_bucket",
    "ttc_confidence", "selected_source", "selected_rank", "pred_x", "pred_y",
    "target_x", "target_y", "error_px", "pck48", "box48_hit", "score",
]


def parse_topk(value: str) -> list[tuple[float, float, float]]:
    points = []
    for item in value.split(";"):
        if item:
            x, y, score = item.split(",")
            points.append((float(x), float(y), float(score)))
    return points


def bucket(probe: int) -> str:
    if probe <= 20:
        return "near"
    if probe <= 50:
        return "mid"
    return "far"


def confidence_from_ttc(row: dict[str, str], temperature: float = 1.0) -> tuple[float, np.ndarray]:
    probabilities = np.asarray([float(value) for value in row["probabilities"].split(";")], dtype=np.float32)
    probabilities = np.power(np.maximum(probabilities, 1e-12), 1.0 / max(temperature, 1e-6))
    probabilities /= probabilities.sum()
    entropy = float(-np.sum(probabilities * np.log(np.maximum(probabilities, 1e-12))) / np.log(len(probabilities)))
    return 1.0 - entropy, probabilities


def result_row(strategy: str, pred: dict, source: dict, confidence: float, selected_source: str, rank: int, x: float, y: float, score: float) -> dict[str, str]:
    target_x = float(source["target_tip_x"])
    target_y = float(source["target_tip_y"])
    error = math.hypot(x - target_x, y - target_y)
    return {
        "strategy": strategy, "dataset_split": pred["dataset_split"], "record_id": source["record_id"],
        "image_name": source["image_name"], "probe": source["probe"], "ttc_bucket": bucket(int(source["probe"])),
        "ttc_confidence": f"{confidence:.6f}", "selected_source": selected_source, "selected_rank": str(rank),
        "pred_x": f"{x:.3f}", "pred_y": f"{y:.3f}", "target_x": f"{target_x:.3f}", "target_y": f"{target_y:.3f}",
        "error_px": f"{error:.3f}", "pck48": "1" if error <= 48 else "0",
        "box48_hit": "1" if abs(x - target_x) <= 24 and abs(y - target_y) <= 24 else "0", "score": f"{score:.6f}",
    }


def summarize(rows: list[dict[str, str]]) -> dict:
    errors = np.asarray([float(row["error_px"]) for row in rows], dtype=np.float32)
    return {
        "samples": len(rows), "median_error_px": float(np.median(errors)) if len(errors) else None,
        "p75_error_px": float(np.quantile(errors, 0.75)) if len(errors) else None,
        "p90_error_px": float(np.quantile(errors, 0.90)) if len(errors) else None,
        "mean_error_px": float(np.mean(errors)) if len(errors) else None,
        "max_error_px": float(np.max(errors)) if len(errors) else None,
        "pck48": float(np.mean(errors <= 48)) if len(errors) else None,
        "failure_rate_gt48": float(np.mean(errors > 48)) if len(errors) else None,
        "box48_hit": float(np.mean([row["box48_hit"] == "1" for row in rows])) if rows else None,
    }


def full_summary(rows: list[dict[str, str]], baseline_by_name: dict[str, dict] | None = None) -> dict:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["ttc_bucket"]].append(row)
    output = {"overall": summarize(rows), "by_ttc_bucket": {name: summarize(items) for name, items in groups.items()}}
    if baseline_by_name is not None:
        wins = losses = ties = 0
        for row in rows:
            delta = float(row["error_px"]) - float(baseline_by_name[row["image_name"]]["error_px"])
            if delta < -1e-3:
                wins += 1
            elif delta > 1e-3:
                losses += 1
            else:
                ties += 1
        output["versus_A"] = {"wins": wins, "losses": losses, "ties": ties}
    return output


def constraint_score(summary: dict, baseline: dict, cfg: dict) -> tuple:
    overall = summary["overall"]
    near = summary["by_ttc_bucket"].get("near", {})
    mid = summary["by_ttc_bucket"].get("mid", {})
    far = summary["by_ttc_bucket"].get("far", {})
    violations = 0
    if overall["box48_hit"] + float(cfg.get("max_box48_regression", 0.0)) < baseline["overall"]["box48_hit"]:
        violations += 1
    tolerance = float(cfg.get("max_near_mid_pck48_regression", 0.02))
    for name, current in (("near", near), ("mid", mid)):
        if current.get("pck48", 0.0) + tolerance < baseline["by_ttc_bucket"][name]["pck48"]:
            violations += 1
    return (violations, far.get("failure_rate_gt48", 1.0), far.get("p75_error_px", float("inf")), far.get("p90_error_px", float("inf")), overall["median_error_px"])


def gate_rows(threshold: float, split: str, a_by_name: dict, c_by_name: dict, rows_by_name: dict, ttc_by_name: dict, temperature: float) -> list[dict[str, str]]:
    output = []
    for name, a in a_by_name.items():
        if a["dataset_split"] != split:
            continue
        c = c_by_name[name]
        confidence, _ = confidence_from_ttc(ttc_by_name[name], temperature)
        selected, label = (c, "C") if confidence >= threshold else (a, "A")
        output.append(result_row("E_gate", selected, rows_by_name[name], confidence, label, 1, float(selected["pred_x"]), float(selected["pred_y"]), float(selected["pred_score"])))
    return output


def build_cache(rows_by_name: dict, a_predictions: list[dict], crop_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sources = [rows_by_name[pred["image_name"]] for pred in a_predictions if pred["dataset_split"] == "train"]
    matrix = np.stack([visual_patch_feature(row["vision_path"], float(row["target_tip_x"]), float(row["target_tip_y"]), crop_size) for row in sources])
    mean, std = matrix.mean(axis=0), matrix.std(axis=0)
    std[std < 1e-6] = 1.0
    return (matrix - mean) / std, mean, std


def candidate_set(a: dict, c: dict) -> list[dict]:
    candidates = []
    for source_name, pred in (("A", a), ("C", c)):
        points = parse_topk(pred["topk_points"])
        top_score = max(points[0][2], 1e-6)
        for rank, (x, y, score) in enumerate(points, start=1):
            base_score = 0.7 * (1.0 - 0.12 * (rank - 1)) + 0.3 * min(score / top_score, 1.0)
            existing = next((item for item in candidates if math.hypot(item["x"] - x, item["y"] - y) <= 6.0), None)
            if existing is None:
                candidates.append({"x": x, "y": y, "base_score": base_score, "source": source_name, "rank": rank})
            elif base_score > existing["base_score"]:
                existing.update({"base_score": base_score, "source": source_name, "rank": rank})
    return candidates


def prepare_rerank_cases(split: str, a_by_name: dict, c_by_name: dict, rows_by_name: dict, ttc_by_name: dict, tracks: dict, cache_z: np.ndarray, cache_mean: np.ndarray, cache_std: np.ndarray, cfg: dict, temperature: float) -> list[dict]:
    cases = []
    ttc_values = np.asarray(cfg["ttc_values"], dtype=np.float32)
    sigma = float(cfg.get("ttc_kernel_sigma", 12.5))
    speed_scale = float(cfg.get("speed_scale_px", 4.0))
    crop_size = int(cfg.get("cache_crop_size", 48))
    for name, a in a_by_name.items():
        if a["dataset_split"] != split:
            continue
        c, source = c_by_name[name], rows_by_name[name]
        confidence, probabilities = confidence_from_ttc(ttc_by_name[name], temperature)
        features = trajectory_features(source, tracks, int(cfg.get("history_frames", 32)), float(cfg.get("spatial_scale_px", 48.0)), speed_scale)
        direction, normalized_speed = motion_basis(features)
        speed = normalized_speed * speed_scale
        tip = np.asarray([float(source["tip_x"]), float(source["tip_y"])], dtype=np.float32)
        candidates = candidate_set(a, c)
        for candidate in candidates:
            vector = np.asarray([candidate["x"], candidate["y"]], dtype=np.float32) - tip
            if speed > 1e-6 and np.linalg.norm(direction) > 0:
                longitudinal = float(np.dot(vector, direction))
                candidate_ttc = max(longitudinal / speed, 0.0)
                lateral = abs(float(vector[0] * -direction[1] + vector[1] * direction[0])) / 48.0
            else:
                candidate_ttc, lateral = 0.0, float(np.linalg.norm(vector)) / 48.0
            likelihood = float(np.sum(probabilities * np.exp(-0.5 * ((candidate_ttc - ttc_values) / sigma) ** 2)))
            candidate["ttc_penalty"] = -math.log(max(likelihood, 1e-8))
            candidate["lateral_penalty"] = lateral
            visual = visual_patch_feature(source["vision_path"], candidate["x"], candidate["y"], crop_size)
            visual_z = (visual - cache_mean) / cache_std
            candidate["visual_penalty"] = float(np.min(np.linalg.norm(cache_z - visual_z[None], axis=1))) / math.sqrt(cache_z.shape[1])
        cases.append({"a": a, "source": source, "confidence": confidence, "candidates": candidates})
    return cases


def rerank_rows(cases: list[dict], weights: tuple[float, float, float]) -> list[dict[str, str]]:
    ttc_weight, lateral_weight, visual_weight = weights
    rows = []
    for case in cases:
        for candidate in case["candidates"]:
            candidate["final_score"] = candidate["base_score"] - ttc_weight * candidate["ttc_penalty"] - lateral_weight * candidate["lateral_penalty"] - visual_weight * candidate["visual_penalty"]
        selected = max(case["candidates"], key=lambda item: item["final_score"])
        rows.append(result_row("F_rerank", case["a"], case["source"], case["confidence"], selected["source"], selected["rank"], selected["x"], selected["y"], selected["final_score"]))
    return rows


def evaluate(config_path: str, section: str) -> dict:
    cfg = load_config(config_path)[section]
    source_rows = read_csv_rows(project_path(cfg["samples_csv"]))
    rows_by_name = {row["image_name"]: row for row in source_rows}
    a_predictions = read_csv_rows(project_path(cfg["baseline_predictions_csv"]))
    c_predictions = read_csv_rows(project_path(cfg["predicted_ttc_predictions_csv"]))
    a_by_name = {row["image_name"]: row for row in a_predictions}
    c_by_name = {row["image_name"]: row for row in c_predictions}
    ttc_by_name = {row["image_name"]: row for row in read_csv_rows(project_path(cfg["ttc_predictions_csv"]))}
    tracks = read_trajectory_tracks(project_path(cfg["motion_tracks_csv"]))

    temperature_trials = []
    for temperature in [float(value) for value in cfg.get("calibration_temperatures", [1.0])]:
        losses = []
        for name, row in ttc_by_name.items():
            if row["dataset_split"] != "val":
                continue
            _, probabilities = confidence_from_ttc(row, temperature)
            losses.append(-math.log(max(float(probabilities[int(row["target_class"])]), 1e-12)))
        temperature_trials.append((float(np.mean(losses)), temperature))
    calibration_nll, best_temperature = min(temperature_trials)

    baseline_rows = {}
    baseline_summary = {}
    for split in ("val", "test"):
        baseline_rows[split] = [result_row("A", pred, rows_by_name[name], confidence_from_ttc(ttc_by_name[name], best_temperature)[0], "A", 1, float(pred["pred_x"]), float(pred["pred_y"]), float(pred["pred_score"])) for name, pred in a_by_name.items() if pred["dataset_split"] == split]
        baseline_summary[split] = full_summary(baseline_rows[split])

    gate_trials = []
    for threshold in [float(value) for value in cfg["gate_thresholds"]]:
        rows = gate_rows(threshold, "val", a_by_name, c_by_name, rows_by_name, ttc_by_name, best_temperature)
        summary = full_summary(rows)
        gate_trials.append((constraint_score(summary, baseline_summary["val"], cfg), threshold, summary))
    _, best_gate_threshold, best_gate_val = min(gate_trials, key=lambda item: item[0])

    cache_z, cache_mean, cache_std = build_cache(rows_by_name, a_predictions, int(cfg.get("cache_crop_size", 48)))
    val_cases = prepare_rerank_cases("val", a_by_name, c_by_name, rows_by_name, ttc_by_name, tracks, cache_z, cache_mean, cache_std, cfg, best_temperature)
    test_cases = prepare_rerank_cases("test", a_by_name, c_by_name, rows_by_name, ttc_by_name, tracks, cache_z, cache_mean, cache_std, cfg, best_temperature)
    rerank_trials = []
    grids = itertools.product(cfg["rerank_ttc_weights"], cfg["rerank_lateral_weights"], cfg["rerank_visual_weights"])
    for raw_weights in grids:
        weights = tuple(float(value) for value in raw_weights)
        rows = rerank_rows(val_cases, weights)
        summary = full_summary(rows)
        rerank_trials.append((constraint_score(summary, baseline_summary["val"], cfg), weights, summary))
    _, best_weights, best_rerank_val = min(rerank_trials, key=lambda item: item[0])

    gate_test = gate_rows(best_gate_threshold, "test", a_by_name, c_by_name, rows_by_name, ttc_by_name, best_temperature)
    rerank_test = rerank_rows(test_cases, best_weights)
    c_test = [result_row("C", pred, rows_by_name[name], confidence_from_ttc(ttc_by_name[name], best_temperature)[0], "C", 1, float(pred["pred_x"]), float(pred["pred_y"]), float(pred["pred_score"])) for name, pred in c_by_name.items() if pred["dataset_split"] == "test"]
    baseline_test_by_name = {row["image_name"]: row for row in baseline_rows["test"]}
    all_output = baseline_rows["test"] + c_test + gate_test + rerank_test
    write_csv_rows(project_path(cfg["output_csv"]), all_output, FIELDS)
    summary = {
        "selection_policy": "All E thresholds and F weights selected on validation only; test evaluated once.",
        "validation": {"A": baseline_summary["val"], "E": best_gate_val, "F": best_rerank_val},
        "selected": {"TTC_temperature": best_temperature, "validation_calibration_nll": calibration_nll, "E_confidence_threshold": best_gate_threshold, "F_weights": {"ttc": best_weights[0], "lateral": best_weights[1], "visual": best_weights[2]}},
        "test": {
            "A": full_summary(baseline_rows["test"], baseline_test_by_name),
            "C": full_summary(c_test, baseline_test_by_name),
            "E": full_summary(gate_test, baseline_test_by_name),
            "F": full_summary(rerank_test, baseline_test_by_name),
        },
        "output_csv": str(project_path(cfg["output_csv"])),
    }
    write_json(project_path(cfg["metrics_json"]), summary)
    print(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate confidence-gated C and validation-tuned TTC Top-K reranking.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--section", default="ttc_strategy_evaluation")
    args = parser.parse_args()
    evaluate(args.config, args.section)


if __name__ == "__main__":
    main()
