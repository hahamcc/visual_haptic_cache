from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import load_config, project_path
from .temporal_progress import DEFAULT_TTC_VALUES, read_trajectory_tracks
from .train_contact_region import (
    ContactRegionDataset,
    TemporalConditionedUNet,
    TinyUNet,
    collate_batch,
    forward_contact_model,
    parse_probe,
    read_motion_tracks,
    topk_points,
)
from .utils import read_csv_rows, write_csv_rows, write_json


FIELDS = [
    "model", "dataset_split", "record_id", "image_name", "probe", "top1_error",
    "recall_1", "recall_5", "recall_10", "recall_20", "box_recall_1",
    "box_recall_5", "box_recall_10", "box_recall_20", "top20_min_error",
]


def load_model_and_dataset(
    config: dict,
    section: str,
    device: torch.device,
    allowed_splits: set[str] | None = None,
) -> tuple[torch.nn.Module, ContactRegionDataset]:
    cfg = config[section]
    allowed_splits = allowed_splits or {"val", "test"}
    rows = [row for row in read_csv_rows(project_path(cfg["samples_csv"])) if row["dataset_split"] in allowed_splits]
    temporal_fusion = str(cfg.get("temporal_fusion", "none"))
    use_trajectory = temporal_fusion in {"predicted_ttc", "trajectory"}
    use_ttc_channel = bool(cfg.get("use_ttc_channel", False))
    use_motion_channels = bool(cfg.get("use_motion_channels", False))
    input_channels = 7 + int(use_ttc_channel) + (4 if use_motion_channels else 0)
    history = int(cfg.get("trajectory_history_frames", 32))
    trajectory_format = str(cfg.get("trajectory_format", "legacy"))
    trajectory_tracks = read_trajectory_tracks(project_path(cfg["motion_tracks_csv"])) if use_trajectory else {}
    dataset = ContactRegionDataset(
        rows, int(cfg["input_width"]), int(cfg["input_height"]), float(cfg["geometry_sigma"]),
        use_ttc_channel=use_ttc_channel, ttc_normalizer=float(cfg.get("ttc_normalizer", 100.0)),
        use_motion_channels=use_motion_channels,
        motion_tracks=read_motion_tracks(project_path(cfg["motion_tracks_csv"])) if use_motion_channels else {},
        motion_window_frames=int(cfg.get("motion_window_frames", 15)),
        use_trajectory_branch=use_trajectory, trajectory_tracks=trajectory_tracks,
        trajectory_history_frames=history,
        trajectory_spatial_scale_px=float(cfg.get("trajectory_spatial_scale_px", 48.0)),
        trajectory_speed_scale_px=float(cfg.get("trajectory_speed_scale_px", 4.0)),
        displacement_scale_px=float(cfg.get("displacement_scale_px", 48.0)),
        ttc_values=[int(value) for value in cfg.get("ttc_values", DEFAULT_TTC_VALUES)],
        trajectory_format=trajectory_format,
    )
    if use_trajectory:
        model = TemporalConditionedUNet(
            in_channels=input_channels, trajectory_hidden_size=int(cfg.get("trajectory_hidden_size", 64)),
            num_ttc_classes=len(cfg.get("ttc_values", DEFAULT_TTC_VALUES)), fusion_mode=temporal_fusion,
            trajectory_format=trajectory_format,
        )
    else:
        model = TinyUNet(in_channels=input_channels)
    checkpoint = torch.load(project_path(cfg["checkpoint_dir"]) / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    return model, dataset


def summarize(rows: list[dict[str, str]]) -> dict:
    result = {"samples": len(rows)}
    for k in (1, 5, 10, 20):
        result[f"recall_{k}"] = float(np.mean([row[f"recall_{k}"] == "1" for row in rows])) if rows else None
        result[f"box_recall_{k}"] = float(np.mean([row[f"box_recall_{k}"] == "1" for row in rows])) if rows else None
    result["median_top1_error_px"] = float(np.median([float(row["top1_error"]) for row in rows])) if rows else None
    return result


def evaluate(config_path: str, section: str) -> dict:
    config = load_config(config_path)
    cfg = config[section]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_rows = []
    summary = {"device": str(device), "models": {}}
    for label, model_section in cfg["model_sections"].items():
        model, dataset = load_model_and_dataset(config, model_section, device)
        loader = DataLoader(dataset, batch_size=int(cfg.get("batch_size", 16)), shuffle=False, num_workers=0, collate_fn=collate_batch)
        model_rows = []
        with torch.no_grad():
            for batch in loader:
                output = forward_contact_model(model, batch, device)
                for index, row in enumerate(batch["rows"]):
                    target_x, target_y, width, height = batch["coords"][index].numpy()
                    points = topk_points(output["heatmap"][index, 0].cpu(), 20, int(cfg.get("suppression_radius", 6)))
                    points = [(x / int(config[model_section]["input_width"]) * width, y / int(config[model_section]["input_height"]) * height, score) for x, y, score in points]
                    errors = [float(np.hypot(x - target_x, y - target_y)) for x, y, _ in points]
                    boxes = [abs(x - target_x) <= 24 and abs(y - target_y) <= 24 for x, y, _ in points]
                    result = {
                        "model": label, "dataset_split": row["dataset_split"], "record_id": row["record_id"],
                        "image_name": row["image_name"], "probe": str(parse_probe(row) or ""),
                        "top1_error": f"{errors[0]:.3f}", "top20_min_error": f"{min(errors):.3f}",
                    }
                    for k in (1, 5, 10, 20):
                        result[f"recall_{k}"] = "1" if min(errors[:k]) <= 48 else "0"
                        result[f"box_recall_{k}"] = "1" if any(boxes[:k]) else "0"
                    output_rows.append(result)
                    model_rows.append(result)
        splits = {}
        for split_name in ("val", "test"):
            split_rows = [row for row in model_rows if row["dataset_split"] == split_name]
            by_probe = defaultdict(list)
            for row in split_rows:
                by_probe[row["probe"]].append(row)
            splits[split_name] = {"overall": summarize(split_rows), "by_probe": {probe: summarize(items) for probe, items in sorted(by_probe.items(), key=lambda item: int(item[0]))}}
        summary["models"][label] = splits
    write_csv_rows(project_path(cfg["output_csv"]), output_rows, FIELDS)
    write_json(project_path(cfg["metrics_json"]), summary)
    print(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate strict and Euclidean proposal recall at K=1/5/10/20.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--section", default="proposal_recall")
    args = parser.parse_args()
    evaluate(args.config, args.section)


if __name__ == "__main__":
    main()
