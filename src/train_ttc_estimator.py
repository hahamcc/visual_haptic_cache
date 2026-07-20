from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .config import load_config, project_path
from .temporal_progress import (
    DEFAULT_TTC_VALUES,
    MASKED_TRAJECTORY_FEATURE_SIZE,
    MaskedTTCEstimator,
    TRAJECTORY_FEATURE_SIZE,
    TTCEstimator,
    displacement_target,
    masked_trajectory_features,
    read_trajectory_tracks,
    trajectory_features,
)
from .train_contact_region import parse_probe, set_seed, ttc_bucket_name
from .utils import ensure_dir, read_csv_rows, write_csv_rows, write_json


PREDICTION_FIELDS = [
    "dataset_split", "record_id", "image_name", "target_ttc", "predicted_ttc",
    "target_class", "predicted_class", "exact_hit", "adjacent_hit", "absolute_error",
    "target_bucket", "predicted_bucket", "probabilities", "target_dx", "target_dy",
    "predicted_dx", "predicted_dy", "displacement_error_px",
    "real_point_count", "history_span_frames", "padding_ratio", "repeated_point_ratio",
    "max_frame_gap", "cumulative_displacement",
]


class TrajectoryTTCDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], tracks: dict, cfg: dict) -> None:
        self.rows = rows
        self.tracks = tracks
        self.ttc_values = [int(value) for value in cfg.get("ttc_values", DEFAULT_TTC_VALUES)]
        self.class_by_value = {value: index for index, value in enumerate(self.ttc_values)}
        self.history_frames = int(cfg.get("history_frames", 32))
        self.spatial_scale_px = float(cfg.get("spatial_scale_px", 48.0))
        self.speed_scale_px = float(cfg.get("speed_scale_px", 4.0))
        self.displacement_scale_px = float(cfg.get("displacement_scale_px", 48.0))
        self.trajectory_format = str(cfg.get("trajectory_format", "legacy"))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        probe = parse_probe(row)
        if probe not in self.class_by_value:
            raise ValueError(f"Unsupported TTC value {probe} in {row['image_name']}")
        if self.trajectory_format == "masked":
            trajectory, valid_mask, quality = masked_trajectory_features(
                row, self.tracks, self.history_frames, self.spatial_scale_px, self.speed_scale_px
            )
        else:
            trajectory = trajectory_features(
                row, self.tracks, self.history_frames, self.spatial_scale_px, self.speed_scale_px
            )
            valid_mask = np.ones(self.history_frames, dtype=np.float32)
            quality = {
                "real_point_count": float(self.history_frames), "history_span_frames": float(self.history_frames - 1),
                "padding_ratio": 0.0, "repeated_point_ratio": 0.0, "max_frame_gap": 1.0,
                "cumulative_displacement": 0.0,
            }
        return {
            "trajectory": torch.from_numpy(trajectory),
            "valid_mask": torch.from_numpy(valid_mask),
            "ttc_class": torch.tensor(self.class_by_value[probe], dtype=torch.long),
            "displacement": torch.from_numpy(displacement_target(row, self.displacement_scale_px)),
            "quality": quality,
            "row": row,
        }


def collate_batch(batch: list[dict]) -> dict:
    return {
        "trajectory": torch.stack([item["trajectory"] for item in batch]),
        "valid_mask": torch.stack([item["valid_mask"] for item in batch]),
        "ttc_class": torch.stack([item["ttc_class"] for item in batch]),
        "displacement": torch.stack([item["displacement"] for item in batch]),
        "rows": [item["row"] for item in batch],
        "quality": [item["quality"] for item in batch],
    }


def forward_ttc_model(model: nn.Module, batch: dict, device: torch.device) -> dict[str, torch.Tensor]:
    trajectory = batch["trajectory"].to(device)
    if isinstance(model, MaskedTTCEstimator):
        return model(trajectory, batch["valid_mask"].to(device))
    return model(trajectory)


def coarse_bucket(value: float) -> str:
    if value <= 20:
        return "near"
    if value <= 50:
        return "mid"
    return "far"


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, cfg: dict, split: str) -> tuple[dict, list[dict]]:
    model.eval()
    ttc_values = np.asarray(cfg.get("ttc_values", DEFAULT_TTC_VALUES), dtype=np.float32)
    displacement_scale = float(cfg.get("displacement_scale_px", 48.0))
    class_loss_fn = nn.CrossEntropyLoss()
    displacement_loss_fn = nn.SmoothL1Loss()
    losses = []
    predictions = []
    confusion = np.zeros((3, 3), dtype=np.int64)
    bucket_names = ["near", "mid", "far"]
    with torch.no_grad():
        for batch in loader:
            labels = batch["ttc_class"].to(device)
            target_displacement = batch["displacement"].to(device)
            output = forward_ttc_model(model, batch, device)
            loss = class_loss_fn(output["ttc_logits"], labels) + float(cfg.get("displacement_loss_weight", 0.25)) * displacement_loss_fn(output["displacement"], target_displacement)
            losses.append(float(loss.item()))
            probabilities = torch.softmax(output["ttc_logits"], dim=1).cpu().numpy()
            predicted_displacement = output["displacement"].cpu().numpy()
            for index, row in enumerate(batch["rows"]):
                target_class = int(labels[index].item())
                predicted_class = int(np.argmax(probabilities[index]))
                target_ttc = float(ttc_values[target_class])
                predicted_ttc = float(np.sum(probabilities[index] * ttc_values))
                target_bucket = coarse_bucket(target_ttc)
                predicted_bucket = coarse_bucket(predicted_ttc)
                confusion[bucket_names.index(target_bucket), bucket_names.index(predicted_bucket)] += 1
                target_delta = target_displacement[index].cpu().numpy() * displacement_scale
                predicted_delta = predicted_displacement[index] * displacement_scale
                predictions.append({
                    "dataset_split": split,
                    "record_id": row["record_id"],
                    "image_name": row["image_name"],
                    "target_ttc": f"{target_ttc:.3f}",
                    "predicted_ttc": f"{predicted_ttc:.3f}",
                    "target_class": str(target_class),
                    "predicted_class": str(predicted_class),
                    "exact_hit": "1" if predicted_class == target_class else "0",
                    "adjacent_hit": "1" if abs(predicted_class - target_class) <= 1 else "0",
                    "absolute_error": f"{abs(predicted_ttc - target_ttc):.3f}",
                    "target_bucket": target_bucket,
                    "predicted_bucket": predicted_bucket,
                    "probabilities": ";".join(f"{value:.6f}" for value in probabilities[index]),
                    "target_dx": f"{target_delta[0]:.3f}",
                    "target_dy": f"{target_delta[1]:.3f}",
                    "predicted_dx": f"{predicted_delta[0]:.3f}",
                    "predicted_dy": f"{predicted_delta[1]:.3f}",
                    "displacement_error_px": f"{float(np.linalg.norm(predicted_delta - target_delta)):.3f}",
                    **{key: f"{float(value):.6f}" for key, value in batch["quality"][index].items()},
                })
    exact = [row["exact_hit"] == "1" for row in predictions]
    adjacent = [row["adjacent_hit"] == "1" for row in predictions]
    errors = [float(row["absolute_error"]) for row in predictions]
    displacement_errors = [float(row["displacement_error_px"]) for row in predictions]
    return {
        "split": split,
        "samples": len(predictions),
        "loss": float(np.mean(losses)) if losses else None,
        "bucket_accuracy": float(np.mean(exact)) if exact else None,
        "adjacent_bucket_accuracy": float(np.mean(adjacent)) if adjacent else None,
        "ttc_mae_frames": float(np.mean(errors)) if errors else None,
        "median_displacement_error_px": float(np.median(displacement_errors)) if displacement_errors else None,
        "near_mid_far_confusion": {name: {pred: int(confusion[i, j]) for j, pred in enumerate(bucket_names)} for i, name in enumerate(bucket_names)},
    }, predictions


def train_ttc(config_path: str, section: str, epochs_override: int | None, eval_only: bool) -> dict:
    cfg = load_config(config_path)[section]
    rows = read_csv_rows(project_path(cfg["samples_csv"]))
    tracks = read_trajectory_tracks(project_path(cfg["motion_tracks_csv"]))
    rows_by_split: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        rows_by_split[row["dataset_split"]].append(row)
    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    datasets = {name: TrajectoryTTCDataset(items, tracks, cfg) for name, items in rows_by_split.items()}
    loaders = {
        name: DataLoader(dataset, batch_size=int(cfg.get("batch_size", 32)), shuffle=name == "train", num_workers=0, collate_fn=collate_batch)
        for name, dataset in datasets.items()
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if str(cfg.get("trajectory_format", "legacy")) == "masked":
        model = MaskedTTCEstimator(hidden_size=int(cfg.get("hidden_size", 64)), num_classes=len(cfg.get("ttc_values", DEFAULT_TTC_VALUES))).to(device)
    else:
        model = TTCEstimator(input_size=TRAJECTORY_FEATURE_SIZE, hidden_size=int(cfg.get("hidden_size", 64)), num_classes=len(cfg.get("ttc_values", DEFAULT_TTC_VALUES))).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("learning_rate", 0.001)), weight_decay=float(cfg.get("weight_decay", 1e-5)))
    class_loss_fn = nn.CrossEntropyLoss()
    displacement_loss_fn = nn.SmoothL1Loss()
    checkpoint_dir = ensure_dir(project_path(cfg["checkpoint_dir"]))
    best_path = checkpoint_dir / "best.pt"
    last_path = checkpoint_dir / "last.pt"
    epochs = int(epochs_override or cfg.get("epochs", 100))
    best_val = float("inf")
    selection_metric = str(cfg.get("selection_metric", "loss"))
    history = []
    start = time.time()
    if eval_only and not best_path.exists():
        raise FileNotFoundError(best_path)
    if not eval_only:
        for epoch in range(1, epochs + 1):
            model.train()
            train_losses = []
            for batch in loaders["train"]:
                output = forward_ttc_model(model, batch, device)
                loss = class_loss_fn(output["ttc_logits"], batch["ttc_class"].to(device))
                loss = loss + float(cfg.get("displacement_loss_weight", 0.25)) * displacement_loss_fn(output["displacement"], batch["displacement"].to(device))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                train_losses.append(float(loss.item()))
            val_summary, _ = evaluate(model, loaders["val"], device, cfg, "val")
            history.append({"epoch": epoch, "train_loss": float(np.mean(train_losses)), "val_loss": val_summary["loss"], "val_ttc_mae_frames": val_summary["ttc_mae_frames"]})
            state = {"model": model.state_dict(), "config": cfg, "epoch": epoch, "history": history}
            torch.save(state, last_path)
            selection_value = float(val_summary[selection_metric])
            if selection_value < best_val:
                best_val = selection_value
                torch.save(state, best_path)
            if epoch == 1 or epoch == epochs or epoch % 10 == 0:
                print(f"epoch={epoch:03d} train_loss={np.mean(train_losses):.5f} val_loss={val_summary['loss']:.5f} val_mae={val_summary['ttc_mae_frames']:.2f}")
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    summaries = {}
    all_predictions = []
    for split in ("train", "val", "test"):
        summaries[split], predictions = evaluate(model, loaders[split], device, cfg, split)
        all_predictions.extend(predictions)
    write_csv_rows(project_path(cfg["predictions_csv"]), all_predictions, PREDICTION_FIELDS)
    summary = {
        "device": str(device), "config_section": section, "history_frames": int(cfg.get("history_frames", 32)),
        "trajectory_format": str(cfg.get("trajectory_format", "legacy")),
        "selection_metric": selection_metric,
        "input_policy": "only tip/base samples at or before current frame; no contact frame or probe input",
        "ttc_values": cfg.get("ttc_values", DEFAULT_TTC_VALUES), "elapsed_seconds": round(time.time() - start, 2),
        "checkpoint_epoch": checkpoint.get("epoch"), **summaries, "best_checkpoint": str(best_path),
    }
    write_json(project_path(cfg["metrics_json"]), summary)
    print(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an online-safe trajectory TTC classifier.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--section", default="ttc_estimator")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()
    train_ttc(args.config, args.section, args.epochs, args.eval_only)


if __name__ == "__main__":
    main()
