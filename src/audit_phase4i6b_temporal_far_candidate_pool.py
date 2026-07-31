"""Audit strict trajectory eligibility and motion diversity in Phase4I.6A."""
from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .config import load_config, project_path
from .phase4h_dino_adaptation import final_holdout_keys
from .temporal_progress import (
    masked_trajectory_features,
    read_trajectory_tracks,
)
from .utils import read_csv_rows, write_csv_rows, write_json


QUERY_FIELDS = [
    "split",
    "record_id",
    "image_name",
    "probe",
    "frame_id",
    "contact_frame_detected",
    "observed_ttc",
    "sequence_ready",
    "real_point_count",
    "history_span",
    "padding_ratio",
    "max_frame_gap",
    "vision_path_exists",
    "touch_path_exists",
    "quality_passed",
    "failure_reasons",
]

RECORD_FIELDS = [
    "eligibility_rank",
    "split",
    "record_id",
    "queries",
    "probes",
    "strict_eligible",
    "failure_reasons",
    "motion_profile",
    "mean_speed_px",
    "speed_cv",
    "normalized_speed_slope",
    "mean_acceleration_px",
    "direction_stability",
    "cumulative_turn_radians",
    "pause_ratio",
    "cumulative_displacement_px",
    "probe100_remaining_distance_px",
    "probe75_remaining_distance_px",
]


def trajectory_points(
    sample: dict[str, str],
    tracks: dict[tuple[str, str], list[dict[str, float]]],
    history_frames: int,
) -> np.ndarray:
    current = int(sample["frame_id"])
    by_frame = {
        int(point["frame_id"]): point
        for point in tracks.get((sample["split"], sample["record_id"]), [])
        if current - history_frames + 1 <= int(point["frame_id"]) <= current
    }
    frames = list(range(current - history_frames + 1, current + 1))
    if any(frame not in by_frame for frame in frames):
        return np.empty((0, 2), dtype=np.float32)
    return np.asarray(
        [[by_frame[frame]["tip_x"], by_frame[frame]["tip_y"]] for frame in frames],
        dtype=np.float32,
    )


def motion_diagnostics(
    points: np.ndarray,
    pause_step_threshold_px: float,
) -> dict[str, float]:
    if len(points) < 2:
        return {
            "mean_speed_px": 0.0,
            "speed_cv": 0.0,
            "normalized_speed_slope": 0.0,
            "mean_acceleration_px": 0.0,
            "direction_stability": 0.0,
            "cumulative_turn_radians": 0.0,
            "pause_ratio": 1.0,
            "cumulative_displacement_px": 0.0,
        }
    velocity = np.diff(points, axis=0)
    speed = np.linalg.norm(velocity, axis=1)
    mean_speed = float(speed.mean())
    speed_cv = float(speed.std() / max(mean_speed, 1e-6))
    x = np.linspace(-1.0, 1.0, len(speed), dtype=np.float32)
    slope = float(np.polyfit(x, speed, 1)[0])
    normalized_slope = slope / max(mean_speed, 1e-6)
    acceleration = np.diff(velocity, axis=0)
    mean_acceleration = (
        float(np.linalg.norm(acceleration, axis=1).mean())
        if len(acceleration)
        else 0.0
    )
    moving = speed > pause_step_threshold_px
    unit = velocity[moving] / np.maximum(
        speed[moving, None],
        1e-6,
    )
    direction_stability = (
        float(np.linalg.norm(unit.mean(axis=0))) if len(unit) else 0.0
    )
    if len(unit) >= 2:
        cosine = np.clip(np.sum(unit[:-1] * unit[1:], axis=1), -1.0, 1.0)
        cumulative_turn = float(np.arccos(cosine).sum())
    else:
        cumulative_turn = 0.0
    return {
        "mean_speed_px": mean_speed,
        "speed_cv": speed_cv,
        "normalized_speed_slope": normalized_slope,
        "mean_acceleration_px": mean_acceleration,
        "direction_stability": direction_stability,
        "cumulative_turn_radians": cumulative_turn,
        "pause_ratio": float((~moving).mean()),
        "cumulative_displacement_px": float(speed.sum()),
    }


def classify_motion(diagnostics: dict[str, float], cfg: dict) -> str:
    if diagnostics["pause_ratio"] >= float(cfg["pause_ratio_threshold"]):
        return "pause_resume"
    turning = (
        diagnostics["direction_stability"]
        <= float(cfg["turning_stability_threshold"])
        or diagnostics["cumulative_turn_radians"]
        >= float(cfg["turning_radians_threshold"])
    )
    if turning:
        if diagnostics["speed_cv"] >= float(
            cfg["variable_speed_cv_threshold"]
        ):
            return "turning_variable_speed"
        return "turning_constant_speed"
    slope = diagnostics["normalized_speed_slope"]
    if slope >= float(cfg["speed_slope_threshold"]):
        return "straight_accelerating"
    if slope <= -float(cfg["speed_slope_threshold"]):
        return "straight_decelerating"
    return "straight_constant_velocity"


def query_quality(
    sample: dict[str, str],
    tracks: dict[tuple[str, str], list[dict[str, float]]],
    cfg: dict,
) -> dict[str, str]:
    probe = int(sample["probe"])
    observed_ttc = (
        int(sample["contact_frame_detected"]) - int(sample["frame_id"])
    )
    _, _, quality = masked_trajectory_features(
        sample,
        tracks,
        history_frames=int(cfg["history_frames"]),
        spatial_scale_px=float(cfg["spatial_scale_px"]),
        speed_scale_px=float(cfg["speed_scale_px"]),
    )
    vision_exists = Path(sample["vision_path"]).exists()
    touch_exists = Path(sample["touch_path"]).exists()
    failures = []
    if observed_ttc != probe:
        failures.append("contact_frame_minus_frame_id_mismatch")
    if sample.get("sequence_ready", "0") != "1":
        failures.append("sequence_not_ready")
    if quality["real_point_count"] < float(cfg["minimum_real_point_count"]):
        failures.append("insufficient_real_points")
    if quality["history_span_frames"] < float(cfg["minimum_history_span"]):
        failures.append("insufficient_history_span")
    if quality["padding_ratio"] > float(cfg["maximum_padding_ratio"]):
        failures.append("padding_ratio_exceeded")
    if quality["max_frame_gap"] > float(cfg["maximum_frame_gap"]):
        failures.append("frame_gap_exceeded")
    if bool(cfg["require_paths_exist"]) and not vision_exists:
        failures.append("vision_path_missing")
    if bool(cfg["require_paths_exist"]) and not touch_exists:
        failures.append("touch_path_missing")
    return {
        "split": sample["split"],
        "record_id": sample["record_id"],
        "image_name": sample["image_name"],
        "probe": str(probe),
        "frame_id": sample["frame_id"],
        "contact_frame_detected": sample["contact_frame_detected"],
        "observed_ttc": str(observed_ttc),
        "sequence_ready": sample.get("sequence_ready", "0"),
        "real_point_count": f"{quality['real_point_count']:.6f}",
        "history_span": f"{quality['history_span_frames']:.6f}",
        "padding_ratio": f"{quality['padding_ratio']:.6f}",
        "max_frame_gap": f"{quality['max_frame_gap']:.6f}",
        "vision_path_exists": str(int(vision_exists)),
        "touch_path_exists": str(int(touch_exists)),
        "quality_passed": str(int(not failures)),
        "failure_reasons": "|".join(failures),
    }


def remaining_distance(sample: dict[str, str]) -> float:
    return math.hypot(
        float(sample["target_tip_x"]) - float(sample["tip_x"]),
        float(sample["target_tip_y"]) - float(sample["tip_y"]),
    )


def build_record_rows(
    samples: list[dict[str, str]],
    query_rows: list[dict[str, str]],
    tracks: dict[tuple[str, str], list[dict[str, float]]],
    cfg: dict,
) -> list[dict[str, str]]:
    samples_by_record: dict[tuple[str, str], list[dict[str, str]]] = (
        defaultdict(list)
    )
    quality_by_name = {row["image_name"]: row for row in query_rows}
    for sample in samples:
        samples_by_record[(sample["split"], sample["record_id"])].append(
            sample
        )
    output = []
    required_probes = {int(value) for value in cfg["probe_focus"]}
    for (split, record_id), items in samples_by_record.items():
        observed = {int(item["probe"]) for item in items}
        failures = []
        if observed != required_probes:
            failures.append("required_probe_set_incomplete")
        if len(items) != len(required_probes):
            failures.append("duplicate_or_missing_probe_queries")
        if any(
            quality_by_name[item["image_name"]]["quality_passed"] != "1"
            for item in items
        ):
            failures.append("query_quality_failed")
        by_probe = {int(item["probe"]): item for item in items}
        reference = by_probe.get(max(required_probes), items[0])
        points = trajectory_points(
            reference,
            tracks,
            int(cfg["history_frames"]),
        )
        diagnostics = motion_diagnostics(
            points,
            float(cfg["pause_step_threshold_px"]),
        )
        profile = classify_motion(diagnostics, cfg)
        output.append(
            {
                "eligibility_rank": "",
                "split": split,
                "record_id": record_id,
                "queries": str(len(items)),
                "probes": "|".join(str(value) for value in sorted(observed)),
                "strict_eligible": str(int(not failures)),
                "failure_reasons": "|".join(failures),
                "motion_profile": profile,
                **{
                    key: f"{value:.9f}"
                    for key, value in diagnostics.items()
                },
                "probe100_remaining_distance_px": (
                    f"{remaining_distance(by_probe[100]):.9f}"
                    if 100 in by_probe
                    else ""
                ),
                "probe75_remaining_distance_px": (
                    f"{remaining_distance(by_probe[75]):.9f}"
                    if 75 in by_probe
                    else ""
                ),
            }
        )
    output.sort(
        key=lambda row: (
            -int(row["strict_eligible"]),
            row["motion_profile"],
            -float(row["cumulative_displacement_px"]),
            row["record_id"],
        )
    )
    for index, row in enumerate(output, start=1):
        row["eligibility_rank"] = str(index)
    return output


def audit(config_path: str, section: str) -> dict:
    cfg = load_config(config_path)[section]
    samples = read_csv_rows(project_path(cfg["candidate_samples_csv"]))
    tracks = read_trajectory_tracks(
        project_path(cfg["candidate_motion_tracks_csv"])
    )
    partitions = read_csv_rows(
        project_path(cfg["candidate_partition_csv"])
    )
    allowed = {
        (row["split"], row["record_id"])
        for row in partitions
        if row["partition"] == "development_candidate"
    }
    sample_keys = {(row["split"], row["record_id"]) for row in samples}
    unexpected = sorted(sample_keys - allowed)
    if unexpected:
        raise RuntimeError(
            f"Phase4I.6B found unreserved candidate records: {unexpected[:3]}"
        )
    development = read_csv_rows(
        project_path(cfg["development_samples_csv"])
    )
    development_keys = {
        (row["split"], row["record_id"]) for row in development
    }
    sealed = final_holdout_keys(project_path(cfg["final_partition_csv"]))
    overlap = sorted(sample_keys & (development_keys | sealed))
    if overlap:
        raise RuntimeError(
            f"Phase4I.6B candidate pool overlaps frozen records: {overlap[:3]}"
        )
    selected_samples = [
        row
        for row in samples
        if int(row["probe"]) in {int(value) for value in cfg["probe_focus"]}
    ]
    query_rows = [
        query_quality(sample, tracks, cfg) for sample in selected_samples
    ]
    record_rows = build_record_rows(
        selected_samples,
        query_rows,
        tracks,
        cfg,
    )
    write_csv_rows(
        project_path(cfg["query_output_csv"]),
        query_rows,
        QUERY_FIELDS,
    )
    write_csv_rows(
        project_path(cfg["record_output_csv"]),
        record_rows,
        RECORD_FIELDS,
    )
    eligible = [
        row for row in record_rows if row["strict_eligible"] == "1"
    ]
    profile_counts = Counter(row["motion_profile"] for row in eligible)
    target_records = int(cfg["target_new_records"])
    minimum_profile = int(cfg["minimum_records_per_motion_profile"])
    pool_ready = len(eligible) >= target_records
    balanced_ready = all(
        profile_counts.get(profile, 0) >= minimum_profile
        for profile in cfg["motion_profiles"]
    )
    summary = {
        "mode": "phase4i6b_temporal_far_candidate_pool_audit_v1",
        "candidate_samples": len(selected_samples),
        "candidate_records_with_samples": len(record_rows),
        "strict_eligible_records": len(eligible),
        "strict_eligible_queries": sum(
            int(row["queries"]) for row in eligible
        ),
        "target_new_records": target_records,
        "pool_ready": pool_ready,
        "balanced_motion_pool_ready": balanced_ready,
        "motion_profile_counts": dict(profile_counts),
        "record_failure_counts": dict(
            Counter(
                reason
                for row in record_rows
                for reason in row["failure_reasons"].split("|")
                if reason
            )
        ),
        "query_failure_counts": dict(
            Counter(
                reason
                for row in query_rows
                for reason in row["failure_reasons"].split("|")
                if reason
            )
        ),
        "thresholds": {
            "pause_step_threshold_px": float(
                cfg["pause_step_threshold_px"]
            ),
            "pause_ratio_threshold": float(cfg["pause_ratio_threshold"]),
            "turning_stability_threshold": float(
                cfg["turning_stability_threshold"]
            ),
            "turning_radians_threshold": float(
                cfg["turning_radians_threshold"]
            ),
            "variable_speed_cv_threshold": float(
                cfg["variable_speed_cv_threshold"]
            ),
            "speed_slope_threshold": float(
                cfg["speed_slope_threshold"]
            ),
        },
        "integrity": {
            "source": "frozen split3 Phase4I.6A development candidate pool",
            "model_retrained": False,
            "threshold_retuned": False,
            "query_tactile_input": False,
            "sealed_final_holdout_rows_read": 0,
            "development_rows_read_for_identity_exclusion": len(
                development
            ),
            "development_validation_outcomes_read": 0,
        },
        "outputs": {
            "query_csv": str(project_path(cfg["query_output_csv"])),
            "record_csv": str(project_path(cfg["record_output_csv"])),
        },
        "next_action": (
            "run Phase4I.6C robust tip/base trajectory audit"
            if pool_ready
            else "expand the raw development candidate pool before pairing"
        ),
    }
    write_json(project_path(cfg["metrics_json"]), summary)
    print(summary)
    for row in eligible[: int(cfg["print_record_limit"])]:
        print(
            row["record_id"],
            "profile=",
            row["motion_profile"],
            "speed=",
            row["mean_speed_px"],
            "cv=",
            row["speed_cv"],
            "slope=",
            row["normalized_speed_slope"],
            "stability=",
            row["direction_stability"],
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit Phase4I.6A strict far and motion eligibility."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--section",
        default="phase4i6b_temporal_far_candidate_pool_audit_v1",
    )
    args = parser.parse_args()
    audit(args.config, args.section)


if __name__ == "__main__":
    main()
