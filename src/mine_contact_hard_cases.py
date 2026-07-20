from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import load_config, project_path
from .evaluate_proposal_recall import load_model_and_dataset
from .temporal_progress import masked_trajectory_features, read_trajectory_tracks
from .train_contact_region import collate_batch, forward_contact_model, parse_probe, topk_points
from .utils import write_csv_rows, write_json


FIELDS = [
    "dataset_split", "record_partition", "record_id", "image_name", "probe", "motion_type",
    "top1_error_px", "top10_min_error_px", "top1_box48", "top10_box48", "case_type",
    "max_turn_degrees", "speed_change_ratio", "direction_stability", "top10_points",
]


def movement_type(trajectory: np.ndarray, mask: np.ndarray) -> tuple[str, float, float, float]:
    points = trajectory[np.flatnonzero(mask > 0.5)]
    if len(points) < 3:
        return "low_motion", 0.0, 0.0, 0.0
    velocities = points[1:, 8:10]
    speeds = np.linalg.norm(velocities, axis=1)
    active = velocities[speeds > 1e-6]
    if len(active) < 2:
        return "low_motion", 0.0, 0.0, float(points[-1, 14])
    unit = active / np.linalg.norm(active, axis=1, keepdims=True)
    dots = np.clip(np.sum(unit[1:] * unit[:-1], axis=1), -1.0, 1.0)
    max_turn = float(np.degrees(np.arccos(dots)).max()) if len(dots) else 0.0
    window = max(1, len(speeds) // 3)
    start_speed = float(np.mean(speeds[:window]))
    end_speed = float(np.mean(speeds[-window:]))
    speed_change = (end_speed - start_speed) / max(start_speed, 1e-6)
    stability = float(points[-1, 14])
    if max_turn >= 30.0:
        return "turning", max_turn, speed_change, stability
    if speed_change >= 0.25:
        return "accelerating", max_turn, speed_change, stability
    if speed_change <= -0.25:
        return "decelerating", max_turn, speed_change, stability
    return "straight_steady", max_turn, speed_change, stability


def mine(config_path: str, section: str) -> dict:
    config = load_config(config_path)
    cfg = config[section]
    contact_section = str(cfg["contact_model_section"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, dataset = load_model_and_dataset(config, contact_section, device, {"train", "val"})
    loader = DataLoader(dataset, batch_size=int(cfg.get("batch_size", 16)), shuffle=False, num_workers=0, collate_fn=collate_batch)
    model_cfg = config[contact_section]
    tracks = read_trajectory_tracks(project_path(model_cfg["motion_tracks_csv"]))
    history = int(model_cfg.get("trajectory_history_frames", 16))
    spatial_scale = float(model_cfg.get("trajectory_spatial_scale_px", 48.0))
    speed_scale = float(model_cfg.get("trajectory_speed_scale_px", 4.0))
    topk = int(cfg.get("topk", 10))
    radius = int(cfg.get("suppression_radius", 6))
    output_rows = []

    with torch.no_grad():
        for batch in loader:
            output = forward_contact_model(model, batch, device)
            for index, row in enumerate(batch["rows"]):
                target_x, target_y, width, height = batch["coords"][index].numpy()
                raw_points = topk_points(output["heatmap"][index, 0].cpu(), topk, radius)
                points = [(x / int(model_cfg["input_width"]) * width, y / int(model_cfg["input_height"]) * height, score) for x, y, score in raw_points]
                errors = [float(math.hypot(x - target_x, y - target_y)) for x, y, _ in points]
                boxes = [abs(x - target_x) <= 24.0 and abs(y - target_y) <= 24.0 for x, y, _ in points]
                if boxes[0]:
                    case_type = "easy"
                elif any(boxes):
                    case_type = "rank_hard"
                else:
                    case_type = "proposal_miss"
                trajectory, mask, _ = masked_trajectory_features(row, tracks, history, spatial_scale, speed_scale)
                motion_type, max_turn, speed_change, stability = movement_type(trajectory, mask)
                output_rows.append({
                    "dataset_split": row["dataset_split"], "record_partition": row.get("record_partition", row["dataset_split"]),
                    "record_id": row["record_id"], "image_name": row["image_name"], "probe": str(parse_probe(row) or ""),
                    "motion_type": motion_type, "top1_error_px": f"{errors[0]:.3f}", "top10_min_error_px": f"{min(errors):.3f}",
                    "top1_box48": "1" if boxes[0] else "0", "top10_box48": "1" if any(boxes) else "0", "case_type": case_type,
                    "max_turn_degrees": f"{max_turn:.3f}", "speed_change_ratio": f"{speed_change:.6f}", "direction_stability": f"{stability:.6f}",
                    "top10_points": ";".join(f"{x:.3f},{y:.3f},{score:.6f}" for x, y, score in points),
                })
    by_split: dict[str, dict] = {}
    for split in ("train", "val"):
        rows = [row for row in output_rows if row["dataset_split"] == split]
        by_split[split] = {
            "samples": len(rows), "case_counts": dict(Counter(row["case_type"] for row in rows)),
            "motion_type_counts": dict(Counter(row["motion_type"] for row in rows)),
            "case_by_motion": {
                motion: dict(Counter(row["case_type"] for row in rows if row["motion_type"] == motion))
                for motion in sorted({row["motion_type"] for row in rows})
            },
            "case_by_probe": {
                probe: dict(Counter(row["case_type"] for row in rows if row["probe"] == probe))
                for probe in sorted({row["probe"] for row in rows}, key=int)
            },
        }
    summary = {
        "device": str(device), "contact_model_section": contact_section, "contact_model_frozen": True,
        "policy": "Only train/validation are mined. The fixed final holdout is deliberately excluded.",
        "topk": topk, "splits": by_split,
    }
    write_csv_rows(project_path(cfg["output_csv"]), output_rows, FIELDS)
    write_json(project_path(cfg["metrics_json"]), summary)
    print(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine easy, rank-hard, and proposal-miss samples without viewing final holdout.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--section", default="phase35_hard_case_mining")
    args = parser.parse_args()
    mine(args.config, args.section)


if __name__ == "__main__":
    main()
