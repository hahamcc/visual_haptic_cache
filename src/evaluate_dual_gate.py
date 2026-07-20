from __future__ import annotations

import argparse
import math

import numpy as np

from .config import load_config, project_path
from .evaluate_ttc_strategies import confidence_from_ttc, full_summary, result_row
from .utils import read_csv_rows, write_csv_rows, write_json


FIELDS = [
    "strategy", "dataset_split", "record_id", "image_name", "probe", "ttc_bucket",
    "ttc_confidence", "selected_source", "selected_rank", "pred_x", "pred_y",
    "target_x", "target_y", "error_px", "pck48", "box48_hit", "score",
    "quality_ok", "fallback_reason", "real_point_count", "history_span_frames", "padding_ratio",
]


def quality_check(ttc: dict[str, str], cfg: dict) -> tuple[bool, str]:
    count = float(ttc.get("real_point_count", 0.0))
    span = float(ttc.get("history_span_frames", 0.0))
    padding = float(ttc.get("padding_ratio", 1.0))
    reasons = []
    if count < float(cfg["min_real_points"]):
        reasons.append("insufficient_points")
    if span < float(cfg["min_history_span"]):
        reasons.append("short_span")
    if padding > float(cfg["max_padding_ratio"]):
        reasons.append("excessive_padding")
    return not reasons, ";".join(reasons)


def build_rows(split: str, threshold: float, temperature: float, a_by_name: dict, c_by_name: dict, ttc_by_name: dict, source_by_name: dict, cfg: dict) -> list[dict[str, str]]:
    rows = []
    for name, a in a_by_name.items():
        if a["dataset_split"] != split:
            continue
        c, ttc, source = c_by_name[name], ttc_by_name[name], source_by_name[name]
        confidence, _ = confidence_from_ttc(ttc, temperature)
        quality_ok, quality_reason = quality_check(ttc, cfg)
        use_c = confidence >= threshold and quality_ok
        selected = c if use_c else a
        fallback_reason = "" if use_c else (quality_reason or "low_ttc_confidence")
        row = result_row("E2_dual_gate", selected, source, confidence, "C" if use_c else "A", 1, float(selected["pred_x"]), float(selected["pred_y"]), float(selected["pred_score"]))
        row.update({
            "quality_ok": "1" if quality_ok else "0", "fallback_reason": fallback_reason,
            "real_point_count": ttc.get("real_point_count", ""),
            "history_span_frames": ttc.get("history_span_frames", ""),
            "padding_ratio": ttc.get("padding_ratio", ""),
        })
        rows.append(row)
    return rows


def selection_summary(rows: list[dict[str, str]]) -> dict:
    return {
        "samples": len(rows),
        "c_selection_rate": float(np.mean([row["selected_source"] == "C" for row in rows])) if rows else None,
        "quality_fallback_rate": float(np.mean([row["quality_ok"] == "0" for row in rows])) if rows else None,
        "confidence_fallback_rate": float(np.mean([row["fallback_reason"] == "low_ttc_confidence" for row in rows])) if rows else None,
    }


def evaluate(config_path: str, section: str) -> dict:
    cfg = load_config(config_path)[section]
    source_by_name = {row["image_name"]: row for row in read_csv_rows(project_path(cfg["samples_csv"]))}
    a_by_name = {row["image_name"]: row for row in read_csv_rows(project_path(cfg["baseline_predictions_csv"]))}
    c_by_name = {row["image_name"]: row for row in read_csv_rows(project_path(cfg["predicted_ttc_predictions_csv"]))}
    ttc_by_name = {row["image_name"]: row for row in read_csv_rows(project_path(cfg["ttc_predictions_csv"]))}

    temperature_trials = []
    for temperature in [float(value) for value in cfg["calibration_temperatures"]]:
        losses = []
        for row in ttc_by_name.values():
            if row["dataset_split"] == "val":
                _, probabilities = confidence_from_ttc(row, temperature)
                losses.append(-math.log(max(float(probabilities[int(row["target_class"])]), 1e-12)))
        temperature_trials.append((float(np.mean(losses)), temperature))
    calibration_nll, temperature = min(temperature_trials)

    trials = []
    for threshold in [float(value) for value in cfg["confidence_thresholds"]]:
        rows = build_rows("val", threshold, temperature, a_by_name, c_by_name, ttc_by_name, source_by_name, cfg)
        summary = full_summary(rows)
        far = summary["by_ttc_bucket"]["far"]
        overall = summary["overall"]
        score = (
            int(overall["box48_hit"] < float(cfg.get("minimum_val_box48", 0.0))),
            far["failure_rate_gt48"], far["p75_error_px"], far["p90_error_px"], overall["median_error_px"],
        )
        trials.append((score, threshold, rows, summary))
    _, threshold, val_rows, val_summary = min(trials, key=lambda item: item[0])
    test_rows = build_rows("test", threshold, temperature, a_by_name, c_by_name, ttc_by_name, source_by_name, cfg)
    baseline_test = {name: row for name, row in a_by_name.items() if row["dataset_split"] == "test"}
    summary = {
        "selection_policy": "Temperature and confidence threshold selected on validation only; history quality is a fixed structural contract.",
        "selected": {"temperature": temperature, "calibration_nll": calibration_nll, "confidence_threshold": threshold},
        "quality_contract": {"min_real_points": cfg["min_real_points"], "min_history_span": cfg["min_history_span"], "max_padding_ratio": cfg["max_padding_ratio"]},
        "validation": {"metrics": val_summary, "selection": selection_summary(val_rows)},
        "test": {"metrics": full_summary(test_rows, baseline_test), "selection": selection_summary(test_rows)},
    }
    write_csv_rows(project_path(cfg["output_csv"]), val_rows + test_rows, FIELDS)
    write_json(project_path(cfg["metrics_json"]), summary)
    print(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate calibrated TTC confidence plus structural trajectory-quality gating.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--section", default="dual_gate_masked_16")
    args = parser.parse_args()
    evaluate(args.config, args.section)


if __name__ == "__main__":
    main()
