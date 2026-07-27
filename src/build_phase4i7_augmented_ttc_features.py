"""Build online-safe temporal DINO features for Phase4I.7 TTC training."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .build_phase4h_dino_ablation import image_batch
from .build_phase4i5_temporal_dino_features import (
    existing_cache_matches,
    history_index,
    validate_frame_offsets,
)
from .config import load_config, project_path
from .phase4f_dino_cross_attention import FrozenDinoV2
from .phase4h_dino_adaptation import (
    assert_development_only,
    contact_crop_reflect,
    final_holdout_keys,
    pooled_token_features,
)
from .train_phase4b_predicted_box_cache_ranker import set_seed
from .temporal_progress import (
    MASKED_TRAJECTORY_FEATURE_SIZE,
    masked_trajectory_features,
    read_trajectory_tracks,
)
from .utils import ensure_dir, read_csv_rows, write_json


def tip_geometry(
    sample: dict[str, str],
    track: dict[str, str],
    offset: int,
    oldest_offset: int,
) -> tuple[tuple[float, float], np.ndarray]:
    width = max(float(sample["image_width"]), 1.0)
    height = max(float(sample["image_height"]), 1.0)
    tip_x = float(track["tip_x"])
    tip_y = float(track["tip_y"])
    base_x = float(track["base_x"])
    base_y = float(track["base_y"])
    dx = tip_x - base_x
    dy = tip_y - base_y
    length = max(float(np.hypot(dx, dy)), 1e-6)
    diagonal = max(float(np.hypot(width, height)), 1.0)
    geometry = np.asarray(
        [
            tip_x / width,
            tip_y / height,
            base_x / width,
            base_y / height,
            dx / length,
            dy / length,
            length / diagonal,
            float(offset) / max(abs(oldest_offset), 1),
        ],
        dtype=np.float32,
    )
    return (tip_x, tip_y), geometry


def _sample_identity(rows: list[dict[str, str]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            (
                f"{row['split']}|{row['record_id']}|{row['image_name']}|"
                f"{row['vision_path']}|{row['frame_id']}|{row['probe']}|"
                f"{row['oof_fold']}\n"
            ).encode("utf-8")
        )
    return digest.hexdigest()


def cache_identity(
    rows: list[dict[str, str]],
    offsets: tuple[int, ...],
    cfg: dict,
    increment_fingerprint: str,
) -> dict:
    return {
        "mode": "phase4i7_augmented_ttc_feature_cache_v1",
        "queries": len(rows),
        "query_identity_sha256": _sample_identity(rows),
        "increment_fingerprint": increment_fingerprint,
        "frame_offsets": list(offsets),
        "crop_definition": "per-frame current sensor tip",
        "crop_size": int(cfg["crop_size"]),
        "padding_mode": "reflect",
        "dino_model": str(cfg["dino_model"]),
        "dino_image_size": int(cfg["dino_image_size"]),
        "dino_layer": int(cfg["dino_layer"]),
        "center_sigma": float(cfg["center_sigma"]),
        "trajectory_history_frames": int(cfg["trajectory_history_frames"]),
        "trajectory_spatial_scale_px": float(cfg["spatial_scale_px"]),
        "trajectory_speed_scale_px": float(cfg["speed_scale_px"]),
        "query_true_probe_feature_used": False,
        "query_target_tip_feature_used": False,
        "query_tactile_input": False,
        "future_visual_frames_used": False,
    }


def existing_phase4i7_cache_matches(prefix: Path, identity: dict) -> bool:
    motion_paths = [
        Path(f"{prefix}.motion.npy"),
        Path(f"{prefix}.motion_valid.npy"),
    ]
    if not all(path.is_file() for path in motion_paths):
        return False
    return existing_cache_matches(prefix, identity)


def _load_rows(cfg: dict) -> tuple[list[dict[str, str]], list[dict[str, str]], str]:
    final_partition = project_path(cfg["final_partition_csv"])
    original_all = read_csv_rows(project_path(cfg["original_samples_csv"]))
    assert_development_only(original_all, final_partition)
    original = [
        dict(row)
        for row in original_all
        if row["dataset_split"] == "train"
    ]
    fold_rows = read_csv_rows(project_path(cfg["original_oof_predictions_csv"]))
    fold_by_name = {
        row["image_name"]: row["oof_fold"]
        for row in fold_rows
        if row.get("dataset_split") == "train"
    }
    if len(fold_by_name) != len(original):
        raise RuntimeError(
            "Phase4I.7 original OOF fold manifest has the wrong size"
        )
    for row in original:
        if row["image_name"] not in fold_by_name:
            raise RuntimeError(
                f"Phase4I.7 original fold missing {row['image_name']}"
            )
        row["oof_fold"] = fold_by_name[row["image_name"]]
        row["phase4i7_source"] = "original"

    increment = [
        dict(row)
        for row in read_csv_rows(project_path(cfg["increment_samples_csv"]))
    ]
    sealed = final_holdout_keys(final_partition)
    if any((row["split"], row["record_id"]) in sealed for row in increment):
        raise RuntimeError("Phase4I.7 increment overlaps sealed final holdout")
    original_keys = {(row["split"], row["record_id"]) for row in original}
    increment_keys = {(row["split"], row["record_id"]) for row in increment}
    if original_keys & increment_keys:
        raise RuntimeError("Phase4I.7 original and increment records overlap")
    if any(row.get("oof_fold", "") not in {"0", "1", "2"} for row in increment):
        raise RuntimeError("Phase4I.7 increment is missing strict OOF folds")
    for row in increment:
        row["phase4i7_source"] = "increment"

    metrics_path = project_path(cfg["increment_metrics_json"])
    with metrics_path.open("r", encoding="utf-8") as handle:
        increment_metrics = json.load(handle)
    fingerprint = str(increment_metrics.get("fingerprint", ""))
    if not fingerprint:
        raise RuntimeError("Phase4I.7 increment fingerprint is missing")
    expected_queries = int(increment_metrics["selected_far_queries"])
    if len(increment) != expected_queries:
        raise RuntimeError(
            "Phase4I.7 increment query count differs from frozen metrics"
        )
    return original, increment, fingerprint


def build(config_path: str, section: str) -> dict:
    cfg = load_config(config_path)[section]
    set_seed(int(cfg["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    original, increment, increment_fingerprint = _load_rows(cfg)
    rows = original + increment
    offsets = validate_frame_offsets(cfg["frame_offsets"])
    original_track_rows = read_csv_rows(project_path(cfg["original_tracks_csv"]))
    increment_track_rows = read_csv_rows(project_path(cfg["increment_tracks_csv"]))
    combined_track_rows = original_track_rows + increment_track_rows
    tracks = history_index(combined_track_rows)
    trajectory_tracks = read_trajectory_tracks(
        project_path(cfg["original_tracks_csv"])
    )
    increment_trajectory_tracks = read_trajectory_tracks(
        project_path(cfg["increment_tracks_csv"])
    )
    trajectory_overlap = set(trajectory_tracks) & set(
        increment_trajectory_tracks
    )
    if trajectory_overlap:
        raise RuntimeError(
            f"Phase4I.7 trajectory pools overlap: "
            f"{sorted(trajectory_overlap)[:3]}"
        )
    trajectory_tracks.update(increment_trajectory_tracks)
    prefix = project_path(cfg["feature_cache_prefix"])
    identity = cache_identity(
        rows,
        offsets,
        cfg,
        increment_fingerprint,
    )
    if existing_phase4i7_cache_matches(prefix, identity):
        print(f"phase4i7: reusing temporal feature cache {prefix}", flush=True)
        return identity

    print(f"phase4i7: loading frozen {cfg['dino_model']}", flush=True)
    backbone = FrozenDinoV2(
        str(cfg["dino_model"]),
        int(cfg["dino_image_size"]),
    ).to(device)
    feature_dim = backbone.feature_dim * 3
    shape = (len(rows), len(offsets))
    ensure_dir(prefix.parent)
    features = np.lib.format.open_memmap(
        Path(f"{prefix}.features.npy"),
        mode="w+",
        dtype=np.float16,
        shape=(*shape, feature_dim),
    )
    geometry = np.lib.format.open_memmap(
        Path(f"{prefix}.geometry.npy"),
        mode="w+",
        dtype=np.float32,
        shape=(*shape, 8),
    )
    valid = np.lib.format.open_memmap(
        Path(f"{prefix}.valid.npy"),
        mode="w+",
        dtype=np.float32,
        shape=shape,
    )
    padding = np.lib.format.open_memmap(
        Path(f"{prefix}.padding.npy"),
        mode="w+",
        dtype=np.float32,
        shape=shape,
    )
    motion = np.lib.format.open_memmap(
        Path(f"{prefix}.motion.npy"),
        mode="w+",
        dtype=np.float32,
        shape=(
            len(rows),
            int(cfg["trajectory_history_frames"]),
            MASKED_TRAJECTORY_FEATURE_SIZE,
        ),
    )
    motion_valid = np.lib.format.open_memmap(
        Path(f"{prefix}.motion_valid.npy"),
        mode="w+",
        dtype=np.float32,
        shape=(len(rows), int(cfg["trajectory_history_frames"])),
    )
    features[:] = 0
    geometry[:] = 0
    valid[:] = 0
    padding[:] = 1
    motion[:] = 0
    motion_valid[:] = 0
    np.save(
        Path(f"{prefix}.names.npy"),
        np.asarray([row["image_name"] for row in rows]),
    )
    motion_quality = []
    for query_index, row in enumerate(rows):
        sequence, mask, quality = masked_trajectory_features(
            row,
            trajectory_tracks,
            history_frames=int(cfg["trajectory_history_frames"]),
            spatial_scale_px=float(cfg["spatial_scale_px"]),
            speed_scale_px=float(cfg["speed_scale_px"]),
        )
        if mask.sum() < int(cfg["minimum_motion_valid_frames"]):
            raise RuntimeError(
                f"Phase4I.7 motion history is too short for "
                f"{row['image_name']}: {int(mask.sum())}"
            )
        motion[query_index] = sequence
        motion_valid[query_index] = mask
        motion_quality.append(quality)

    jobs = []
    missing = []
    for query_index, row in enumerate(rows):
        current_frame = int(row["frame_id"])
        for time_index, offset in enumerate(offsets):
            frame = current_frame + offset
            track = tracks.get((row["split"], row["record_id"], frame))
            if track is None or not Path(track["vision_path"]).is_file():
                missing.append((row["image_name"], frame))
                continue
            center, frame_geometry = tip_geometry(
                row,
                track,
                offset,
                offsets[0],
            )
            jobs.append(
                (query_index, time_index, track, center, frame_geometry)
            )
    missing_fraction = len(missing) / max(len(rows) * len(offsets), 1)
    if missing_fraction > float(cfg["maximum_missing_frame_fraction"]):
        raise RuntimeError(
            "Phase4I.7 temporal history is incomplete: "
            f"{len(missing)} frames ({missing_fraction:.3%})"
        )

    batch_size = int(cfg["batch_size"])
    dino_layer = int(cfg["dino_layer"])
    next_report = 500
    with torch.no_grad():
        for start in range(0, len(jobs), batch_size):
            batch = jobs[start : start + batch_size]
            crops, ratios = [], []
            for _, _, track, center, _ in batch:
                crop, ratio = contact_crop_reflect(
                    track["vision_path"],
                    center[0],
                    center[1],
                    int(cfg["crop_size"]),
                )
                crops.append(crop)
                ratios.append(ratio)
            tokens = backbone.forward_layers(
                image_batch(crops).to(device),
                (dino_layer,),
            )[dino_layer]
            pooled = (
                pooled_token_features(tokens, float(cfg["center_sigma"]))
                .cpu()
                .numpy()
                .astype(np.float16)
            )
            for local, (
                query_index,
                time_index,
                _,
                _,
                frame_geometry,
            ) in enumerate(batch):
                features[query_index, time_index] = pooled[local]
                geometry[query_index, time_index] = frame_geometry
                valid[query_index, time_index] = 1
                padding[query_index, time_index] = ratios[local]
            completed = min(start + batch_size, len(jobs))
            if completed >= next_report or completed == len(jobs):
                print(
                    f"phase4i7 temporal DINO: {completed}/{len(jobs)} frames",
                    flush=True,
                )
                while next_report <= completed:
                    next_report += 500
    for array in (features, geometry, valid, padding, motion, motion_valid):
        array.flush()
    report = {
        **identity,
        "device": str(device),
        "original_queries": len(original),
        "increment_queries": len(increment),
        "feature_dim": feature_dim,
        "geometry_dim": 8,
        "motion_dim": MASKED_TRAJECTORY_FEATURE_SIZE,
        "motion_history_frames": int(cfg["trajectory_history_frames"]),
        "minimum_motion_valid_frames": int(
            min(float(item["real_point_count"]) for item in motion_quality)
        ),
        "mean_motion_valid_fraction": float(np.asarray(motion_valid).mean()),
        "valid_frames": len(jobs),
        "missing_frames": len(missing),
        "missing_frame_fraction": missing_fraction,
        "mean_padding_ratio": float(np.asarray(padding).mean()),
        "sealed_final_holdout_rows_read": 0,
        "development_validation_rows_read": 0,
    }
    write_json(Path(f"{prefix}.json"), report)
    write_json(project_path(cfg["feature_metrics_json"]), report)
    print(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Phase4I.7 augmented TTC temporal DINO features."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--section",
        default="phase4i7_augmented_ttc_oof_v1",
    )
    args = parser.parse_args()
    build(args.config, args.section)


if __name__ == "__main__":
    main()
