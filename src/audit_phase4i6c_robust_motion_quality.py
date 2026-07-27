"""Audit robust tip/base motion quality for strict Phase4I.6B records."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .audit_phase4i6b_temporal_far_candidate_pool import (
    classify_motion,
    motion_diagnostics,
)
from .config import load_config, project_path
from .utils import read_csv_rows, write_csv_rows, write_json


OUTPUT_FIELDS = [
    "robust_rank",
    "split",
    "record_id",
    "image_name",
    "raw_motion_profile",
    "robust_motion_profile",
    "track_quality_passed",
    "failure_reasons",
    "raw_tip_step_p50_px",
    "raw_tip_step_p90_px",
    "raw_tip_step_max_px",
    "raw_base_step_p90_px",
    "jump_ratio",
    "tip_base_length_mean_px",
    "tip_base_length_cv",
    "tip_base_velocity_coherence",
    "tip_base_velocity_residual_p90_px",
    "tip_base_velocity_residual_max_px",
    "minimum_tip_confidence",
    "minimum_base_confidence",
    "robust_mean_speed_px",
    "robust_speed_cv",
    "robust_normalized_speed_slope",
    "robust_mean_acceleration_px",
    "robust_direction_stability",
    "robust_cumulative_turn_radians",
    "robust_pause_ratio",
    "robust_cumulative_displacement_px",
]


def read_pose_tracks(
    path: str | Path,
) -> dict[tuple[str, str], list[dict[str, float]]]:
    output: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)
    for row in read_csv_rows(path):
        output[(row["split"], row["record_id"])].append(
            {
                "frame_id": int(row["frame_id"]),
                "tip_x": float(row["tip_x"]),
                "tip_y": float(row["tip_y"]),
                "base_x": float(row["base_x"]),
                "base_y": float(row["base_y"]),
                "tip_confidence": float(row.get("tip_confidence", 1.0)),
                "base_confidence": float(row.get("base_confidence", 1.0)),
            }
        )
    for points in output.values():
        points.sort(key=lambda item: int(item["frame_id"]))
    return output


def exact_pose_history(
    sample: dict[str, str],
    tracks: dict[tuple[str, str], list[dict[str, float]]],
    history_frames: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    current = int(sample["frame_id"])
    start = current - history_frames + 1
    by_frame = {
        int(point["frame_id"]): point
        for point in tracks.get((sample["split"], sample["record_id"]), [])
        if start <= int(point["frame_id"]) <= current
    }
    frames = list(range(start, current + 1))
    if any(frame not in by_frame for frame in frames):
        empty = np.empty((0, 2), dtype=np.float32)
        confidence = np.empty((0,), dtype=np.float32)
        return empty, empty, confidence, confidence
    tip = np.asarray(
        [[by_frame[frame]["tip_x"], by_frame[frame]["tip_y"]] for frame in frames],
        dtype=np.float32,
    )
    base = np.asarray(
        [[by_frame[frame]["base_x"], by_frame[frame]["base_y"]] for frame in frames],
        dtype=np.float32,
    )
    tip_confidence = np.asarray(
        [by_frame[frame]["tip_confidence"] for frame in frames],
        dtype=np.float32,
    )
    base_confidence = np.asarray(
        [by_frame[frame]["base_confidence"] for frame in frames],
        dtype=np.float32,
    )
    return tip, base, tip_confidence, base_confidence


def median_smooth(points: np.ndarray, radius: int) -> np.ndarray:
    if len(points) == 0 or radius <= 0:
        return points.copy()
    output = np.empty_like(points)
    for index in range(len(points)):
        left = max(0, index - radius)
        right = min(len(points), index + radius + 1)
        output[index] = np.median(points[left:right], axis=0)
    return output


def pose_quality_diagnostics(
    tip: np.ndarray,
    base: np.ndarray,
    tip_confidence: np.ndarray,
    base_confidence: np.ndarray,
    cfg: dict,
) -> dict[str, float | str | bool]:
    if len(tip) < 2 or len(base) != len(tip):
        return {
            "passed": False,
            "failure_reasons": "incomplete_pose_history",
        }
    tip_velocity = np.diff(tip, axis=0)
    base_velocity = np.diff(base, axis=0)
    tip_step = np.linalg.norm(tip_velocity, axis=1)
    base_step = np.linalg.norm(base_velocity, axis=1)
    jump = np.maximum(tip_step, base_step) > float(
        cfg["maximum_plausible_step_px"]
    )
    bone_length = np.linalg.norm(tip - base, axis=1)
    bone_mean = float(bone_length.mean())
    bone_cv = float(bone_length.std() / max(bone_mean, 1e-6))
    residual = np.linalg.norm(tip_velocity - base_velocity, axis=1)
    moving = (
        (tip_step > float(cfg["coherence_step_threshold_px"]))
        & (base_step > float(cfg["coherence_step_threshold_px"]))
    )
    if np.any(moving):
        cosine = np.sum(
            tip_velocity[moving] * base_velocity[moving],
            axis=1,
        ) / np.maximum(
            tip_step[moving] * base_step[moving],
            1e-6,
        )
        coherence = float(np.clip(cosine, -1.0, 1.0).mean())
    else:
        coherence = 1.0
    center = (tip + base) * 0.5
    robust_center = median_smooth(center, int(cfg["median_filter_radius"]))
    robust = motion_diagnostics(
        robust_center,
        float(cfg["pause_step_threshold_px"]),
    )
    failures = []
    jump_ratio = float(jump.mean())
    minimum_tip_confidence = float(tip_confidence.min())
    minimum_base_confidence = float(base_confidence.min())
    residual_p90 = float(np.quantile(residual, 0.9))
    residual_max = float(residual.max())
    if jump_ratio > float(cfg["maximum_jump_ratio"]):
        failures.append("jump_ratio_exceeded")
    if max(float(tip_step.max()), float(base_step.max())) > float(
        cfg["maximum_single_step_px"]
    ):
        failures.append("single_step_exceeded")
    if bone_cv > float(cfg["maximum_tip_base_length_cv"]):
        failures.append("tip_base_length_unstable")
    if coherence < float(cfg["minimum_velocity_coherence"]):
        failures.append("tip_base_velocity_incoherent")
    if residual_p90 > float(cfg["maximum_velocity_residual_p90_px"]):
        failures.append("tip_base_velocity_residual_exceeded")
    if residual_max > float(cfg["maximum_velocity_residual_max_px"]):
        failures.append("tip_base_velocity_residual_max_exceeded")
    if minimum_tip_confidence < float(cfg["minimum_track_confidence"]):
        failures.append("tip_confidence_below_threshold")
    if minimum_base_confidence < float(cfg["minimum_track_confidence"]):
        failures.append("base_confidence_below_threshold")
    return {
        "passed": not failures,
        "failure_reasons": "|".join(failures),
        "raw_tip_step_p50_px": float(np.quantile(tip_step, 0.5)),
        "raw_tip_step_p90_px": float(np.quantile(tip_step, 0.9)),
        "raw_tip_step_max_px": float(tip_step.max()),
        "raw_base_step_p90_px": float(np.quantile(base_step, 0.9)),
        "jump_ratio": jump_ratio,
        "tip_base_length_mean_px": bone_mean,
        "tip_base_length_cv": bone_cv,
        "tip_base_velocity_coherence": coherence,
        "tip_base_velocity_residual_p90_px": residual_p90,
        "tip_base_velocity_residual_max_px": residual_max,
        "minimum_tip_confidence": minimum_tip_confidence,
        "minimum_base_confidence": minimum_base_confidence,
        **{f"robust_{key}": value for key, value in robust.items()},
    }


def audit(config_path: str, section: str) -> dict:
    cfg = load_config(config_path)[section]
    record_rows = read_csv_rows(project_path(cfg["phase4i6b_record_csv"]))
    strict = {
        (row["split"], row["record_id"]): row
        for row in record_rows
        if row["strict_eligible"] == "1"
    }
    samples = [
        row
        for row in read_csv_rows(project_path(cfg["candidate_samples_csv"]))
        if (row["split"], row["record_id"]) in strict
        and int(row["probe"]) == int(cfg["reference_probe"])
    ]
    if len(samples) != len(strict):
        raise RuntimeError(
            "Phase4I.6C requires exactly one reference-probe sample per "
            f"strict record: {len(samples)} != {len(strict)}"
        )
    widths = {int(row["image_width"]) for row in samples}
    heights = {int(row["image_height"]) for row in samples}
    if len(widths) != 1 or len(heights) != 1:
        raise RuntimeError(
            "Phase4I.6C requires a single image resolution for its "
            f"quantization contract: widths={widths}, heights={heights}"
        )
    image_width = next(iter(widths))
    image_height = next(iter(heights))
    quantization_x = image_width / float(cfg["localizer_input_width"])
    quantization_y = image_height / float(cfg["localizer_input_height"])
    expected_quantization = float(cfg["expected_coordinate_quantization_px"])
    if (
        abs(quantization_x - expected_quantization) > 1e-6
        or abs(quantization_y - expected_quantization) > 1e-6
    ):
        raise RuntimeError(
            "Phase4I.6C localizer quantization changed: "
            f"x={quantization_x}, y={quantization_y}, "
            f"expected={expected_quantization}"
        )
    tracks = read_pose_tracks(project_path(cfg["candidate_motion_tracks_csv"]))
    output = []
    for sample in samples:
        tip, base, tip_confidence, base_confidence = exact_pose_history(
            sample,
            tracks,
            int(cfg["history_frames"]),
        )
        diagnostics = pose_quality_diagnostics(
            tip,
            base,
            tip_confidence,
            base_confidence,
            cfg,
        )
        robust_values = {
            key.replace("robust_", ""): float(value)
            for key, value in diagnostics.items()
            if key.startswith("robust_")
        }
        profile = classify_motion(robust_values, cfg)
        base_row = strict[(sample["split"], sample["record_id"])]
        row = {
            "robust_rank": "",
            "split": sample["split"],
            "record_id": sample["record_id"],
            "image_name": sample["image_name"],
            "raw_motion_profile": base_row["motion_profile"],
            "robust_motion_profile": profile,
            "track_quality_passed": str(int(bool(diagnostics["passed"]))),
            "failure_reasons": str(diagnostics["failure_reasons"]),
        }
        for field in OUTPUT_FIELDS:
            if field in row:
                continue
            value = diagnostics[field]
            row[field] = f"{float(value):.9f}"
        output.append(row)
    output.sort(
        key=lambda row: (
            -int(row["track_quality_passed"]),
            row["robust_motion_profile"],
            float(row["jump_ratio"]),
            row["record_id"],
        )
    )
    for index, row in enumerate(output, start=1):
        row["robust_rank"] = str(index)
    write_csv_rows(
        project_path(cfg["output_csv"]),
        output,
        OUTPUT_FIELDS,
    )
    passed = [row for row in output if row["track_quality_passed"] == "1"]
    profile_counts = Counter(row["robust_motion_profile"] for row in passed)
    target = int(cfg["target_new_records"])
    summary = {
        "mode": "phase4i6c_robust_motion_quality_audit_v1",
        "strict_input_records": len(output),
        "robust_track_eligible_records": len(passed),
        "target_new_records": target,
        "robust_pool_ready": len(passed) >= target,
        "raw_motion_profile_counts": dict(
            Counter(row["raw_motion_profile"] for row in output)
        ),
        "robust_motion_profile_counts": dict(profile_counts),
        "failure_counts": dict(
            Counter(
                reason
                for row in output
                for reason in row["failure_reasons"].split("|")
                if reason
            )
        ),
        "distribution": {
            metric: {
                "p50": float(
                    np.quantile(
                        [float(row[metric]) for row in output],
                        0.5,
                    )
                ),
                "p90": float(
                    np.quantile(
                        [float(row[metric]) for row in output],
                        0.9,
                    )
                ),
                "max": max(float(row[metric]) for row in output),
            }
            for metric in (
                "raw_tip_step_p90_px",
                "raw_tip_step_max_px",
                "jump_ratio",
                "tip_base_length_cv",
                "tip_base_velocity_residual_p90_px",
                "tip_base_velocity_residual_max_px",
            )
        },
        "thresholds": {
            key: cfg[key]
            for key in (
                "maximum_plausible_step_px",
                "maximum_single_step_px",
                "maximum_jump_ratio",
                "maximum_tip_base_length_cv",
                "minimum_velocity_coherence",
                "maximum_velocity_residual_p90_px",
                "maximum_velocity_residual_max_px",
                "minimum_track_confidence",
                "median_filter_radius",
            )
        },
        "localizer_quantization_contract": {
            "image_width": image_width,
            "image_height": image_height,
            "localizer_input_width": int(cfg["localizer_input_width"]),
            "localizer_input_height": int(cfg["localizer_input_height"]),
            "coordinate_quantization_x_px": quantization_x,
            "coordinate_quantization_y_px": quantization_y,
            "diagonal_one_cell_px": float(
                np.hypot(quantization_x, quantization_y)
            ),
            "opposed_two_keypoint_diagonal_px": float(
                2.0 * np.hypot(quantization_x, quantization_y)
            ),
        },
        "integrity": {
            "source": "strict Phase4I.6B development candidates only",
            "query_tactile_input": False,
            "model_retrained": False,
            "threshold_retuned": False,
            "sealed_final_holdout_rows_read": 0,
            "development_validation_outcomes_read": 0,
        },
        "output_csv": str(project_path(cfg["output_csv"])),
        "next_action": (
            "build visual-contact descriptors and select record pairs"
            if len(passed) >= target
            else "expand the raw candidate pool before final pair selection"
        ),
    }
    write_json(project_path(cfg["metrics_json"]), summary)
    print(summary)
    for row in output[: int(cfg["print_record_limit"])]:
        print(
            row["record_id"],
            "pass=",
            row["track_quality_passed"],
            "profile=",
            row["robust_motion_profile"],
            "jump=",
            row["jump_ratio"],
            "bone_cv=",
            row["tip_base_length_cv"],
            "residual_p90=",
            row["tip_base_velocity_residual_p90_px"],
            "failures=",
            row["failure_reasons"],
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit robust Phase4I.6 tip/base motion quality."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--section",
        default="phase4i6c_robust_motion_quality_audit_v1",
    )
    args = parser.parse_args()
    audit(args.config, args.section)


if __name__ == "__main__":
    main()
