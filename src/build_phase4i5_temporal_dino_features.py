"""Build online-safe four-frame DINO features for contact-progress learning."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .build_phase4h_dino_ablation import image_batch
from .config import load_config, project_path
from .phase4f_dino_cross_attention import FrozenDinoV2
from .phase4h_dino_adaptation import (
    assert_development_only,
    contact_crop_reflect,
    pooled_token_features,
)
from .train_phase4b_predicted_box_cache_ranker import prediction_map, set_seed
from .utils import ensure_dir, read_csv_rows, write_json


def history_index(
    track_rows: list[dict[str, str]],
) -> dict[tuple[str, str, int], dict[str, str]]:
    output = {}
    for row in track_rows:
        key = (row["split"], row["record_id"], int(row["frame_id"]))
        if key in output:
            raise RuntimeError(f"Duplicate Phase4I.5 trajectory frame: {key}")
        output[key] = row
    return output


def validate_frame_offsets(offsets: list[int]) -> tuple[int, ...]:
    values = tuple(int(value) for value in offsets)
    if not values:
        raise ValueError("Phase4I.5 frame offsets cannot be empty")
    if any(value > 0 for value in values):
        raise ValueError("Phase4I.5 cannot read frames after the query time")
    if values != tuple(sorted(set(values))):
        raise ValueError("Phase4I.5 frame offsets must be unique and sorted")
    if values[-1] != 0:
        raise ValueError("Phase4I.5 frame offsets must end at the query frame")
    return values


def corridor_geometry(
    sample: dict[str, str],
    track: dict[str, str],
    predicted_xy: tuple[float, float],
    offset: int,
    oldest_offset: int,
) -> tuple[tuple[float, float], np.ndarray]:
    width = max(float(sample["image_width"]), 1.0)
    height = max(float(sample["image_height"]), 1.0)
    tip = np.asarray(
        [float(track["tip_x"]), float(track["tip_y"])],
        dtype=np.float32,
    )
    predicted = np.asarray(predicted_xy, dtype=np.float32)
    delta = predicted - tip
    center = 0.5 * (tip + predicted)
    diagonal = max(float(np.hypot(width, height)), 1.0)
    temporal_scale = max(abs(oldest_offset), 1)
    geometry = np.asarray(
        [
            tip[0] / width,
            tip[1] / height,
            predicted[0] / width,
            predicted[1] / height,
            delta[0] / width,
            delta[1] / height,
            float(np.linalg.norm(delta)) / diagonal,
            float(offset) / temporal_scale,
        ],
        dtype=np.float32,
    )
    return (float(center[0]), float(center[1])), geometry


def cache_identity(
    rows: list[dict[str, str]],
    predictions: dict[str, dict[str, str]],
    offsets: tuple[int, ...],
    cfg: dict,
) -> dict:
    digest = hashlib.sha256()
    for row in rows:
        prediction = predictions[row["image_name"]]
        digest.update(
            (
                f"{row['image_name']}|{row['vision_path']}|{row['frame_id']}|"
                f"{prediction['pred_x']}|{prediction['pred_y']}\n"
            ).encode("utf-8")
        )
    return {
        "mode": "phase4i5_temporal_dino_feature_cache_v1",
        "queries": len(rows),
        "query_identity_sha256": digest.hexdigest(),
        "frame_offsets": list(offsets),
        "crop_definition": "tip-to-C2-prediction corridor midpoint",
        "crop_size": int(cfg["crop_size"]),
        "padding_mode": "reflect",
        "dino_model": str(cfg["dino_model"]),
        "dino_image_size": int(cfg["dino_image_size"]),
        "dino_layer": int(cfg["dino_layer"]),
        "center_sigma": float(cfg["center_sigma"]),
        "query_true_probe_feature_used": False,
        "query_tactile_input": False,
        "future_visual_frames_used": False,
    }


def existing_cache_matches(prefix: Path, identity: dict) -> bool:
    paths = [
        Path(f"{prefix}.features.npy"),
        Path(f"{prefix}.geometry.npy"),
        Path(f"{prefix}.valid.npy"),
        Path(f"{prefix}.padding.npy"),
        Path(f"{prefix}.names.npy"),
        Path(f"{prefix}.json"),
    ]
    if not all(path.is_file() for path in paths):
        return False
    with paths[-1].open("r", encoding="utf-8") as handle:
        existing = json.load(handle)
    if any(existing.get(key) != value for key, value in identity.items()):
        raise RuntimeError("Existing Phase4I.5 feature-cache identity mismatch")
    return True


def build(config_path: str, section: str) -> dict:
    cfg = load_config(config_path)[section]
    set_seed(int(cfg["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_rows = read_csv_rows(project_path(cfg["samples_csv"]))
    assert_development_only(
        all_rows,
        project_path(cfg["final_partition_csv"]),
    )
    rows = [row for row in all_rows if row["dataset_split"] == "train"]
    predictions = prediction_map(
        read_csv_rows(project_path(cfg["oof_predictions_csv"])),
        rows,
        "train",
        "Phase4I.5 temporal DINO OOF",
    )
    if any(
        not prediction.get("oof_fold", "")
        for prediction in predictions.values()
    ):
        raise RuntimeError("Phase4I.5 contact predictions must be strict OOF")
    offsets = validate_frame_offsets(cfg["frame_offsets"])
    tracks = history_index(read_csv_rows(project_path(cfg["motion_tracks_csv"])))
    prefix = project_path(cfg["feature_cache_prefix"])
    identity = cache_identity(rows, predictions, offsets, cfg)
    if existing_cache_matches(prefix, identity):
        print(f"phase4i5: reusing temporal feature cache {prefix}", flush=True)
        return identity

    print(f"phase4i5: loading frozen {cfg['dino_model']}", flush=True)
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
    features[:] = 0
    geometry[:] = 0
    valid[:] = 0
    padding[:] = 1
    np.save(
        Path(f"{prefix}.names.npy"),
        np.asarray([row["image_name"] for row in rows]),
    )

    jobs = []
    missing = []
    for query_index, row in enumerate(rows):
        prediction = predictions[row["image_name"]]
        predicted_xy = (
            float(prediction["pred_x"]),
            float(prediction["pred_y"]),
        )
        current_frame = int(row["frame_id"])
        for time_index, offset in enumerate(offsets):
            frame = current_frame + offset
            track = tracks.get((row["split"], row["record_id"], frame))
            if track is None or not Path(track["vision_path"]).is_file():
                missing.append((row["image_name"], frame))
                continue
            center, frame_geometry = corridor_geometry(
                row,
                track,
                predicted_xy,
                offset,
                offsets[0],
            )
            jobs.append(
                (
                    query_index,
                    time_index,
                    track,
                    center,
                    frame_geometry,
                )
            )
    maximum_missing_fraction = float(cfg["maximum_missing_frame_fraction"])
    missing_fraction = len(missing) / max(len(rows) * len(offsets), 1)
    if missing_fraction > maximum_missing_fraction:
        raise RuntimeError(
            "Phase4I.5 temporal history is incomplete: "
            f"{len(missing)} missing frames ({missing_fraction:.3%})"
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
                    f"phase4i5 temporal DINO: {completed}/{len(jobs)} frames",
                    flush=True,
                )
                while next_report <= completed:
                    next_report += 500
    for array in (features, geometry, valid, padding):
        array.flush()
    report = {
        **identity,
        "device": str(device),
        "feature_dim": feature_dim,
        "geometry_dim": 8,
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
        description="Build Phase4I.5 four-frame frozen-DINO features."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--section",
        default="phase4i5_temporal_progress_oof_v1",
    )
    args = parser.parse_args()
    build(args.config, args.section)


if __name__ == "__main__":
    main()
