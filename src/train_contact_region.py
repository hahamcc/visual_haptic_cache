from __future__ import annotations

import argparse
import re
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .config import load_config, project_path
from .temporal_progress import (
    DEFAULT_TTC_VALUES,
    MASKED_TRAJECTORY_FEATURE_SIZE,
    MaskedTrajectoryEncoder,
    TRAJECTORY_FEATURE_SIZE,
    TrajectoryEncoder,
    displacement_target,
    masked_trajectory_features,
    read_trajectory_tracks,
    trajectory_features,
)
from .utils import ensure_dir, read_csv_rows, write_csv_rows, write_json


PREDICTION_FIELDS = [
    "dataset_split",
    "split",
    "record_id",
    "image_name",
    "probe",
    "ttc_bucket",
    "vision_path",
    "touch_path",
    "is_contact_outlier",
    "target_x",
    "target_y",
    "pred_x",
    "pred_y",
    "pred_score",
    "predicted_ttc",
    "ttc_absolute_error",
    "predicted_delta_x",
    "predicted_delta_y",
    "displacement_error_px",
    "error_px",
    "motion_dx",
    "motion_dy",
    "parallel_error",
    "perpendicular_error",
    "abs_parallel_error",
    "abs_perpendicular_error",
    "abs_error_x",
    "abs_error_y",
    "pck_16",
    "pck_32",
    "pck_48",
    "bbox_hit",
    "box48_hit",
    "top5_hit_48",
    "top5_bbox_hit",
    "top5_box48_hit",
    "topk_points",
]

RETRIEVAL_FIELDS = [
    "dataset_split",
    "query_split",
    "query_record_id",
    "query_image_name",
    "query_vision_path",
    "query_pred_x",
    "query_pred_y",
    "query_target_x",
    "query_target_y",
    "retrieved_split",
    "retrieved_record_id",
    "retrieved_image_name",
    "retrieved_vision_path",
    "retrieved_touch_path",
    "retrieved_target_x",
    "retrieved_target_y",
    "distance",
    "same_record",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_heatmap(width: int, height: int, x: float, y: float, sigma: float) -> np.ndarray:
    xs = np.arange(width, dtype=np.float32)
    ys = np.arange(height, dtype=np.float32)[:, None]
    heatmap = np.exp(-((xs - x) ** 2 + (ys - y) ** 2) / (2.0 * sigma * sigma))
    return heatmap.astype(np.float32)


def read_manifest_touch_paths(path: Path) -> dict[tuple[str, str, int], str]:
    if not path.exists():
        return {}
    return {
        (row["split"], row["record_id"], int(row["frame_id"])): row["touch_path"]
        for row in read_csv_rows(path)
    }


def attach_touch_paths(rows: list[dict[str, str]], manifest_csv: Path) -> None:
    touch_by_key = read_manifest_touch_paths(manifest_csv)
    for row in rows:
        if row.get("touch_path"):
            continue
        split = row["split"]
        record_id = row["record_id"]
        contact_frame_text = row.get("contact_frame_from_name") or row.get("contact_frame_detected") or row.get("contact_frame")
        if not contact_frame_text:
            row["touch_path"] = ""
            continue
        contact_frame = int(contact_frame_text)
        current_frame = int(row["frame_id"])
        row["touch_path"] = touch_by_key.get(
            (split, record_id, contact_frame),
            touch_by_key.get((split, record_id, current_frame), ""),
        )


def read_motion_tracks(path: Path) -> dict[tuple[str, str], list[tuple[int, float, float]]]:
    if not path.exists():
        return {}
    tracks: dict[tuple[str, str], list[tuple[int, float, float]]] = defaultdict(list)
    for row in read_csv_rows(path):
        tracks[(row["split"], row["record_id"])].append(
            (int(row["frame_id"]), float(row["tip_x"]), float(row["tip_y"]))
        )
    for key in tracks:
        tracks[key].sort(key=lambda item: item[0])
    return tracks


def motion_direction_from_tracks(
    row: dict[str, str],
    tracks: dict[tuple[str, str], list[tuple[int, float, float]]],
    window_frames: int,
) -> tuple[float | None, float | None]:
    key = (row["split"], row["record_id"])
    frame_id = int(row["frame_id"])
    points = [
        (frame, tip_x, tip_y)
        for frame, tip_x, tip_y in tracks.get(key, [])
        if frame_id - window_frames <= frame <= frame_id
    ]
    if len(points) < 2:
        return None, None

    frames = np.asarray([item[0] for item in points], dtype=np.float32)
    xs = np.asarray([item[1] for item in points], dtype=np.float32)
    ys = np.asarray([item[2] for item in points], dtype=np.float32)
    centered = frames - frames.mean()
    denom = float(np.sum(centered * centered))
    if denom <= 1e-6:
        dx = float(xs[-1] - xs[0])
        dy = float(ys[-1] - ys[0])
    else:
        dx = float(np.sum(centered * (xs - xs.mean())) / denom)
        dy = float(np.sum(centered * (ys - ys.mean())) / denom)
    norm = float(np.hypot(dx, dy))
    if norm <= 1e-6:
        return None, None
    return dx / norm, dy / norm


def motion_feature_maps_from_tracks(
    row: dict[str, str],
    tracks: dict[tuple[str, str], list[tuple[int, float, float]]],
    window_frames: int,
) -> tuple[float, float, float, float]:
    key = (row["split"], row["record_id"])
    frame_id = int(row["frame_id"])
    width = max(float(row.get("image_width", 1.0)), 1.0)
    height = max(float(row.get("image_height", 1.0)), 1.0)
    denom = max(width, height)
    points = [
        (frame, tip_x, tip_y)
        for frame, tip_x, tip_y in tracks.get(key, [])
        if frame_id - window_frames <= frame <= frame_id
    ]
    if len(points) < 2:
        return 0.0, 0.0, 0.0, 0.0

    frame0, x0, y0 = points[0]
    frame1, x1, y1 = points[-1]
    dt = max(float(frame1 - frame0), 1.0)
    velocity_x = (x1 - x0) / dt
    velocity_y = (y1 - y0) / dt
    speed = float(np.hypot(velocity_x, velocity_y))

    unit_steps = []
    for prev, cur in zip(points[:-1], points[1:]):
        step_dt = max(float(cur[0] - prev[0]), 1.0)
        dx = (cur[1] - prev[1]) / step_dt
        dy = (cur[2] - prev[2]) / step_dt
        norm = float(np.hypot(dx, dy))
        if norm > 1e-6:
            unit_steps.append((dx / norm, dy / norm))
    if len(unit_steps) < 2:
        stability = 0.0
    else:
        cosines = [
            a[0] * b[0] + a[1] * b[1]
            for a, b in zip(unit_steps[:-1], unit_steps[1:])
        ]
        stability = float((np.mean(cosines) + 1.0) / 2.0)

    return velocity_x / width, velocity_y / height, speed / denom, stability


def parse_probe(row: dict[str, str]) -> int | None:
    if row.get("probe"):
        return int(float(row["probe"]))
    match = re.search(r"probe(\d+)", row.get("image_name", ""))
    if match:
        return int(match.group(1))
    return None


def normalize_ttc_buckets(raw_buckets: dict | None) -> dict[str, set[int]]:
    if not raw_buckets:
        raw_buckets = {"near": [5, 10, 20, 25], "mid": [30, 50], "far": [75, 100]}
    return {name: {int(value) for value in values} for name, values in raw_buckets.items()}


def ttc_bucket_name(probe: int | None, buckets: dict[str, set[int]]) -> str:
    if probe is None:
        return "unknown"
    for name, values in buckets.items():
        if probe in values:
            return name
    return "other"


def summarize_prediction_rows(rows: list[dict[str, str]]) -> dict:
    errors = [float(row["error_px"]) for row in rows]
    pck16 = [row["pck_16"] == "1" for row in rows]
    pck32 = [row["pck_32"] == "1" for row in rows]
    pck48 = [row["pck_48"] == "1" for row in rows]
    box48_hits = [row["box48_hit"] == "1" for row in rows]
    top5_hits = [row["top5_hit_48"] == "1" for row in rows]
    top5_box48_hits = [row["top5_box48_hit"] == "1" for row in rows]
    parallel_errors = [
        float(row["parallel_error"])
        for row in rows
        if row.get("parallel_error", "") != ""
    ]
    perpendicular_errors = [
        float(row["perpendicular_error"])
        for row in rows
        if row.get("perpendicular_error", "") != ""
    ]
    summary = {
        "samples": len(rows),
        "mean_error_px": float(np.mean(errors)) if errors else None,
        "median_error_px": float(np.median(errors)) if errors else None,
        "pck_16": float(np.mean(pck16)) if pck16 else None,
        "pck_32": float(np.mean(pck32)) if pck32 else None,
        "pck_48": float(np.mean(pck48)) if pck48 else None,
        "box48_hit": float(np.mean(box48_hits)) if box48_hits else None,
        "top5_hit_48": float(np.mean(top5_hits)) if top5_hits else None,
        "top5_box48_hit": float(np.mean(top5_box48_hits)) if top5_box48_hits else None,
    }
    if parallel_errors:
        summary.update(
            {
                "mean_parallel_error_px": float(np.mean(parallel_errors)),
                "median_parallel_error_px": float(np.median(parallel_errors)),
                "median_abs_parallel_error_px": float(np.median(np.abs(parallel_errors))),
                "parallel_negative_rate": float(np.mean([value < 0.0 for value in parallel_errors])),
            }
        )
    if perpendicular_errors:
        summary.update(
            {
                "mean_perpendicular_error_px": float(np.mean(perpendicular_errors)),
                "median_perpendicular_error_px": float(np.median(perpendicular_errors)),
                "median_abs_perpendicular_error_px": float(np.median(np.abs(perpendicular_errors))),
            }
        )
    return summary


def grouped_prediction_summary(rows: list[dict[str, str]], field: str) -> dict:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row.get(field, "unknown")].append(row)
    def sort_key(item: tuple[str, list[dict[str, str]]]) -> tuple[int, str]:
        key = item[0]
        return (int(key), key) if key.isdigit() else (9999, key)
    return {
        key: summarize_prediction_rows(items)
        for key, items in sorted(groups.items(), key=sort_key)
    }


class ContactRegionDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, str]],
        input_width: int,
        input_height: int,
        geometry_sigma: float,
        use_ttc_channel: bool = False,
        ttc_normalizer: float = 100.0,
        use_motion_channels: bool = False,
        motion_tracks: dict[tuple[str, str], list[tuple[int, float, float]]] | None = None,
        motion_window_frames: int = 15,
        use_trajectory_branch: bool = False,
        trajectory_tracks: dict | None = None,
        trajectory_history_frames: int = 32,
        trajectory_spatial_scale_px: float = 48.0,
        trajectory_speed_scale_px: float = 4.0,
        displacement_scale_px: float = 48.0,
        ttc_values: list[int] | None = None,
        trajectory_format: str = "legacy",
    ) -> None:
        self.rows = rows
        self.input_width = input_width
        self.input_height = input_height
        self.geometry_sigma = geometry_sigma
        self.use_ttc_channel = use_ttc_channel
        self.ttc_normalizer = max(float(ttc_normalizer), 1.0)
        self.use_motion_channels = use_motion_channels
        self.motion_tracks = motion_tracks or {}
        self.motion_window_frames = motion_window_frames
        self.use_trajectory_branch = use_trajectory_branch
        self.trajectory_tracks = trajectory_tracks or {}
        self.trajectory_history_frames = trajectory_history_frames
        self.trajectory_spatial_scale_px = trajectory_spatial_scale_px
        self.trajectory_speed_scale_px = trajectory_speed_scale_px
        self.displacement_scale_px = displacement_scale_px
        self.ttc_values = ttc_values or DEFAULT_TTC_VALUES
        self.ttc_class_by_value = {value: index for index, value in enumerate(self.ttc_values)}
        self.trajectory_format = trajectory_format

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        image_path = row["vision_path"] or row["image_path"]
        image = Image.open(image_path).convert("RGB")
        orig_w, orig_h = image.size
        image = image.resize((self.input_width, self.input_height), Image.BILINEAR)
        image_arr = np.asarray(image, dtype=np.float32) / 255.0
        image_arr = np.transpose(image_arr, (2, 0, 1))

        tip_x = float(row["tip_x"]) / orig_w * self.input_width
        tip_y = float(row["tip_y"]) / orig_h * self.input_height
        base_x = float(row["base_x"]) / orig_w * self.input_width
        base_y = float(row["base_y"]) / orig_h * self.input_height
        tip_map = make_heatmap(self.input_width, self.input_height, tip_x, tip_y, self.geometry_sigma)
        base_map = make_heatmap(self.input_width, self.input_height, base_x, base_y, self.geometry_sigma)
        direction_x = np.full((self.input_height, self.input_width), float(row["direction_x"]), dtype=np.float32)
        direction_y = np.full((self.input_height, self.input_width), float(row["direction_y"]), dtype=np.float32)
        geometry_maps = [tip_map, base_map, direction_x, direction_y]
        if self.use_ttc_channel:
            probe = parse_probe(row) or 0
            ttc_map = np.full(
                (self.input_height, self.input_width),
                float(probe) / self.ttc_normalizer,
                dtype=np.float32,
            )
            geometry_maps.append(ttc_map)
        if self.use_motion_channels:
            velocity_x, velocity_y, speed, stability = motion_feature_maps_from_tracks(
                row,
                self.motion_tracks,
                self.motion_window_frames,
            )
            geometry_maps.extend(
                [
                    np.full((self.input_height, self.input_width), velocity_x, dtype=np.float32),
                    np.full((self.input_height, self.input_width), velocity_y, dtype=np.float32),
                    np.full((self.input_height, self.input_width), speed, dtype=np.float32),
                    np.full((self.input_height, self.input_width), stability, dtype=np.float32),
                ]
            )
        features = np.concatenate([image_arr, np.stack(geometry_maps, axis=0)], axis=0)

        target = np.load(row["heatmap_path"]).astype(np.float32)[None, :, :]
        coords = np.asarray(
            [
                float(row["target_tip_x"]),
                float(row["target_tip_y"]),
                float(row["image_width"]),
                float(row["image_height"]),
            ],
            dtype=np.float32,
        )
        if self.use_trajectory_branch:
            if self.trajectory_format == "masked":
                trajectory, trajectory_mask, trajectory_quality = masked_trajectory_features(
                    row, self.trajectory_tracks, self.trajectory_history_frames,
                    self.trajectory_spatial_scale_px, self.trajectory_speed_scale_px,
                )
            else:
                trajectory = trajectory_features(
                    row, self.trajectory_tracks, self.trajectory_history_frames,
                    self.trajectory_spatial_scale_px, self.trajectory_speed_scale_px,
                )
                trajectory_mask = np.ones(self.trajectory_history_frames, dtype=np.float32)
                trajectory_quality = {}
            probe = parse_probe(row)
            if probe not in self.ttc_class_by_value:
                raise ValueError(f"Unsupported TTC value {probe} in {row['image_name']}")
            ttc_class = self.ttc_class_by_value[probe]
            displacement = displacement_target(row, self.displacement_scale_px)
        else:
            feature_size = MASKED_TRAJECTORY_FEATURE_SIZE if self.trajectory_format == "masked" else TRAJECTORY_FEATURE_SIZE
            trajectory = np.zeros((self.trajectory_history_frames, feature_size), dtype=np.float32)
            trajectory_mask = np.zeros(self.trajectory_history_frames, dtype=np.float32)
            trajectory_quality = {}
            ttc_class = -1
            displacement = np.zeros(2, dtype=np.float32)
        return {
            "input": torch.from_numpy(features),
            "target": torch.from_numpy(target),
            "coords": torch.from_numpy(coords),
            "trajectory": torch.from_numpy(trajectory),
            "trajectory_mask": torch.from_numpy(trajectory_mask),
            "trajectory_quality": trajectory_quality,
            "ttc_class": torch.tensor(ttc_class, dtype=torch.long),
            "displacement": torch.from_numpy(displacement),
            "row": row,
        }


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TinyUNet(nn.Module):
    def __init__(self, in_channels: int = 7, out_channels: int = 1, features: int = 16) -> None:
        super().__init__()
        self.enc1 = ConvBlock(in_channels, features)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(features, features * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(features * 2, features * 4)
        self.up2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(features * 4, features * 2)
        self.up1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(features * 2, features)
        self.head = nn.Conv2d(features, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))
        d2 = self.up2(b)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return torch.sigmoid(self.head(d1))


class TemporalConditionedUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 7,
        features: int = 16,
        trajectory_hidden_size: int = 64,
        num_ttc_classes: int = 7,
        fusion_mode: str = "predicted_ttc",
        trajectory_format: str = "legacy",
    ) -> None:
        super().__init__()
        if fusion_mode not in {"predicted_ttc", "trajectory"}:
            raise ValueError(f"Unsupported temporal fusion mode: {fusion_mode}")
        self.fusion_mode = fusion_mode
        self.trajectory_format = trajectory_format
        self.enc1 = ConvBlock(in_channels, features)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(features, features * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(features * 2, features * 4)
        if trajectory_format == "masked":
            self.trajectory_encoder = MaskedTrajectoryEncoder(MASKED_TRAJECTORY_FEATURE_SIZE, trajectory_hidden_size)
        else:
            self.trajectory_encoder = TrajectoryEncoder(TRAJECTORY_FEATURE_SIZE, trajectory_hidden_size)
        self.ttc_head = nn.Linear(trajectory_hidden_size, num_ttc_classes)
        condition_size = num_ttc_classes + 1
        if fusion_mode == "trajectory":
            condition_size += trajectory_hidden_size
        self.film = nn.Linear(condition_size, features * 8)
        self.displacement_head = nn.Sequential(
            nn.Linear(features * 4 + condition_size, trajectory_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(trajectory_hidden_size, 2),
        )
        self.up2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(features * 4, features * 2)
        self.up1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(features * 2, features)
        self.head = nn.Conv2d(features, 1, kernel_size=1)

    def forward(self, image: torch.Tensor, trajectory: torch.Tensor, trajectory_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        e1 = self.enc1(image)
        e2 = self.enc2(self.pool1(e1))
        bottleneck = self.bottleneck(self.pool2(e2))
        if self.trajectory_format == "masked":
            if trajectory_mask is None:
                raise ValueError("Masked trajectory format requires trajectory_mask")
            trajectory_feature = self.trajectory_encoder(trajectory, trajectory_mask)
        else:
            trajectory_feature = self.trajectory_encoder(trajectory)
        ttc_logits = self.ttc_head(trajectory_feature)
        ttc_probabilities = torch.softmax(ttc_logits, dim=1)
        class_axis = torch.linspace(0.0, 1.0, ttc_probabilities.shape[1], device=image.device)
        expected_class = torch.sum(ttc_probabilities * class_axis[None, :], dim=1, keepdim=True)
        condition_parts = [ttc_probabilities, expected_class]
        if self.fusion_mode == "trajectory":
            condition_parts.append(trajectory_feature)
        condition = torch.cat(condition_parts, dim=1)
        gamma, beta = self.film(condition).chunk(2, dim=1)
        fused = bottleneck * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]
        pooled = torch.mean(fused, dim=(2, 3))
        displacement = self.displacement_head(torch.cat([pooled, condition], dim=1))
        d2 = self.up2(fused)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return {
            "heatmap": torch.sigmoid(self.head(d1)),
            "ttc_logits": ttc_logits,
            "ttc_probabilities": ttc_probabilities,
            "displacement": displacement,
        }

    def load_ttc_checkpoint(self, path: Path) -> None:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state = checkpoint["model"]
        encoder_state = {key.removeprefix("encoder."): value for key, value in state.items() if key.startswith("encoder.")}
        head_state = {key.removeprefix("ttc_head."): value for key, value in state.items() if key.startswith("ttc_head.")}
        self.trajectory_encoder.load_state_dict(encoder_state)
        self.ttc_head.load_state_dict(head_state)

    def freeze_ttc(self) -> None:
        for module in (self.trajectory_encoder, self.ttc_head):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad = False


def collate_batch(batch: list[dict]) -> dict:
    return {
        "input": torch.stack([item["input"] for item in batch]),
        "target": torch.stack([item["target"] for item in batch]),
        "coords": torch.stack([item["coords"] for item in batch]),
        "trajectory": torch.stack([item["trajectory"] for item in batch]),
        "trajectory_mask": torch.stack([item["trajectory_mask"] for item in batch]),
        "trajectory_quality": [item["trajectory_quality"] for item in batch],
        "ttc_class": torch.stack([item["ttc_class"] for item in batch]),
        "displacement": torch.stack([item["displacement"] for item in batch]),
        "rows": [item["row"] for item in batch],
    }


def forward_contact_model(model: nn.Module, batch: dict, device: torch.device) -> dict[str, torch.Tensor]:
    inputs = batch["input"].to(device)
    if isinstance(model, TemporalConditionedUNet):
        return model(inputs, batch["trajectory"].to(device), batch["trajectory_mask"].to(device))
    return {"heatmap": model(inputs)}


def topk_points(
    heatmap: torch.Tensor,
    k: int,
    suppression_radius: int,
) -> list[tuple[float, float, float]]:
    work = heatmap.clone()
    height, width = work.shape
    points = []
    for _ in range(k):
        flat_idx = int(torch.argmax(work).item())
        score = float(work.reshape(-1)[flat_idx].item())
        y = flat_idx // width
        x = flat_idx % width
        points.append((float(x), float(y), score))
        x0 = max(0, x - suppression_radius)
        x1 = min(width, x + suppression_radius + 1)
        y0 = max(0, y - suppression_radius)
        y1 = min(height, y + suppression_radius + 1)
        work[y0:y1, x0:x1] = -1.0
    return points


def draw_box(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    box_size: int,
    color: str,
    width: int = 3,
) -> None:
    half = box_size / 2.0
    draw.rectangle((x - half, y - half, x + half, y + half), outline=color, width=width)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    input_width: int,
    input_height: int,
    split_name: str,
    topk: int,
    suppression_radius: int,
    bbox_half_size: float,
    ttc_buckets: dict[str, set[int]],
    motion_tracks: dict[tuple[str, str], list[tuple[int, float, float]]],
    motion_window_frames: int,
    ttc_values: list[int] | None = None,
    displacement_scale_px: float = 48.0,
) -> tuple[dict, list[dict]]:
    model.eval()
    criterion = nn.MSELoss()
    losses: list[float] = []
    predictions: list[dict] = []
    errors: list[float] = []
    pck16: list[bool] = []
    pck32: list[bool] = []
    pck48: list[bool] = []
    bbox_hits: list[bool] = []
    top5_hits: list[bool] = []
    top5_bbox_hits: list[bool] = []
    box48_hits: list[bool] = []
    top5_box48_hits: list[bool] = []
    outlier_errors: list[float] = []
    normal_errors: list[float] = []
    ttc_errors: list[float] = []
    displacement_errors: list[float] = []
    ttc_axis = np.asarray(ttc_values or DEFAULT_TTC_VALUES, dtype=np.float32)

    with torch.no_grad():
        for batch in loader:
            targets = batch["target"].to(device)
            output = forward_contact_model(model, batch, device)
            preds = output["heatmap"]
            losses.append(float(criterion(preds, targets).item()))
            for idx, row in enumerate(batch["rows"]):
                target_x, target_y, orig_w, orig_h = batch["coords"][idx].cpu().numpy()
                points = topk_points(preds[idx, 0].cpu(), topk, suppression_radius)
                scaled_points = [
                    (x / input_width * orig_w, y / input_height * orig_h, score)
                    for x, y, score in points
                ]
                pred_x, pred_y, pred_score = scaled_points[0]
                if "ttc_probabilities" in output:
                    probabilities = output["ttc_probabilities"][idx].cpu().numpy()
                    predicted_ttc = float(np.sum(probabilities * ttc_axis))
                    target_ttc = float(parse_probe(row) or 0)
                    ttc_error = abs(predicted_ttc - target_ttc)
                    predicted_delta = output["displacement"][idx].cpu().numpy() * displacement_scale_px
                    target_delta = batch["displacement"][idx].cpu().numpy() * displacement_scale_px
                    displacement_error = float(np.linalg.norm(predicted_delta - target_delta))
                    ttc_errors.append(ttc_error)
                    displacement_errors.append(displacement_error)
                else:
                    predicted_ttc = None
                    ttc_error = None
                    predicted_delta = None
                    displacement_error = None
                abs_x = abs(pred_x - float(target_x))
                abs_y = abs(pred_y - float(target_y))
                error = float(np.hypot(abs_x, abs_y))
                motion_dx, motion_dy = motion_direction_from_tracks(row, motion_tracks, motion_window_frames)
                if motion_dx is None or motion_dy is None:
                    parallel_error = None
                    perpendicular_error = None
                else:
                    error_x = pred_x - float(target_x)
                    error_y = pred_y - float(target_y)
                    parallel_error = error_x * motion_dx + error_y * motion_dy
                    perpendicular_error = error_x * (-motion_dy) + error_y * motion_dx
                is_outlier = row["record_id"] == "rec_00007"
                top5_hit = any(
                    float(np.hypot(x - float(target_x), y - float(target_y))) <= 48.0
                    for x, y, _ in scaled_points
                )
                top5_bbox_hit = any(
                    abs(x - float(target_x)) <= bbox_half_size and abs(y - float(target_y)) <= bbox_half_size
                    for x, y, _ in scaled_points
                )
                box48_half = 24.0
                top5_box48_hit = any(
                    abs(x - float(target_x)) <= box48_half and abs(y - float(target_y)) <= box48_half
                    for x, y, _ in scaled_points
                )
                errors.append(error)
                pck16.append(error <= 16.0)
                pck32.append(error <= 32.0)
                pck48.append(error <= 48.0)
                bbox_hit = abs_x <= bbox_half_size and abs_y <= bbox_half_size
                box48_hit = abs_x <= box48_half and abs_y <= box48_half
                bbox_hits.append(bbox_hit)
                box48_hits.append(box48_hit)
                top5_hits.append(top5_hit)
                top5_bbox_hits.append(top5_bbox_hit)
                top5_box48_hits.append(top5_box48_hit)
                if is_outlier:
                    outlier_errors.append(error)
                else:
                    normal_errors.append(error)
                predictions.append(
                    {
                        "dataset_split": split_name,
                        "split": row["split"],
                        "record_id": row["record_id"],
                        "image_name": row["image_name"],
                        "probe": str(parse_probe(row) or ""),
                        "ttc_bucket": ttc_bucket_name(parse_probe(row), ttc_buckets),
                        "vision_path": row["vision_path"],
                        "touch_path": row.get("touch_path", ""),
                        "is_contact_outlier": "1" if is_outlier else "0",
                        "target_x": f"{float(target_x):.3f}",
                        "target_y": f"{float(target_y):.3f}",
                        "pred_x": f"{pred_x:.3f}",
                        "pred_y": f"{pred_y:.3f}",
                        "pred_score": f"{pred_score:.6f}",
                        "predicted_ttc": f"{predicted_ttc:.3f}" if predicted_ttc is not None else "",
                        "ttc_absolute_error": f"{ttc_error:.3f}" if ttc_error is not None else "",
                        "predicted_delta_x": f"{predicted_delta[0]:.3f}" if predicted_delta is not None else "",
                        "predicted_delta_y": f"{predicted_delta[1]:.3f}" if predicted_delta is not None else "",
                        "displacement_error_px": f"{displacement_error:.3f}" if displacement_error is not None else "",
                        "error_px": f"{error:.3f}",
                        "motion_dx": f"{motion_dx:.6f}" if motion_dx is not None else "",
                        "motion_dy": f"{motion_dy:.6f}" if motion_dy is not None else "",
                        "parallel_error": f"{parallel_error:.3f}" if parallel_error is not None else "",
                        "perpendicular_error": f"{perpendicular_error:.3f}" if perpendicular_error is not None else "",
                        "abs_parallel_error": f"{abs(parallel_error):.3f}" if parallel_error is not None else "",
                        "abs_perpendicular_error": f"{abs(perpendicular_error):.3f}" if perpendicular_error is not None else "",
                        "abs_error_x": f"{abs_x:.3f}",
                        "abs_error_y": f"{abs_y:.3f}",
                        "pck_16": "1" if error <= 16.0 else "0",
                        "pck_32": "1" if error <= 32.0 else "0",
                        "pck_48": "1" if error <= 48.0 else "0",
                        "bbox_hit": "1" if bbox_hit else "0",
                        "box48_hit": "1" if box48_hit else "0",
                        "top5_hit_48": "1" if top5_hit else "0",
                        "top5_bbox_hit": "1" if top5_bbox_hit else "0",
                        "top5_box48_hit": "1" if top5_box48_hit else "0",
                        "topk_points": ";".join(f"{x:.3f},{y:.3f},{score:.6f}" for x, y, score in scaled_points),
                    }
                )

    summary = {
        "split": split_name,
        "samples": len(predictions),
        "loss": float(np.mean(losses)) if losses else None,
        "mean_error_px": float(np.mean(errors)) if errors else None,
        "median_error_px": float(np.median(errors)) if errors else None,
        "pck_16": float(np.mean(pck16)) if pck16 else None,
        "pck_32": float(np.mean(pck32)) if pck32 else None,
        "pck_48": float(np.mean(pck48)) if pck48 else None,
        "bbox_hit": float(np.mean(bbox_hits)) if bbox_hits else None,
        "box48_hit": float(np.mean(box48_hits)) if box48_hits else None,
        "top5_hit_48": float(np.mean(top5_hits)) if top5_hits else None,
        "top5_bbox_hit": float(np.mean(top5_bbox_hits)) if top5_bbox_hits else None,
        "top5_box48_hit": float(np.mean(top5_box48_hits)) if top5_box48_hits else None,
        "normal_median_error_px": float(np.median(normal_errors)) if normal_errors else None,
        "outlier_median_error_px": float(np.median(outlier_errors)) if outlier_errors else None,
        "ttc_mae_frames": float(np.mean(ttc_errors)) if ttc_errors else None,
        "median_displacement_error_px": float(np.median(displacement_errors)) if displacement_errors else None,
    }
    summary["by_probe"] = grouped_prediction_summary(predictions, "probe")
    summary["by_ttc_bucket"] = grouped_prediction_summary(predictions, "ttc_bucket")
    return summary, predictions


def heatmap_preview(heatmap_path: str | Path, size: tuple[int, int]) -> Image.Image:
    arr = np.load(heatmap_path).astype(np.float32)
    arr = np.clip(arr / max(float(arr.max()), 1e-8) * 255.0, 0, 255).astype(np.uint8)
    return ImageOps.colorize(Image.fromarray(arr, mode="L").resize(size), black="black", white="red")


def draw_prediction_overlay(row: dict, output_path: Path, box_size: int, title: str | None = None) -> None:
    image = Image.open(row["vision_path"]).convert("RGB")
    draw = ImageDraw.Draw(image)
    target = (float(row["target_x"]), float(row["target_y"]))
    pred = (float(row["pred_x"]), float(row["pred_y"]))
    draw_box(draw, target[0], target[1], box_size, "lime", 4)
    draw_box(draw, pred[0], pred[1], box_size, "magenta", 4)
    draw.line((target[0], target[1], pred[0], pred[1]), fill="white", width=2)
    for point in row["topk_points"].split(";")[1:]:
        x_text, y_text, _ = point.split(",")
        x = float(x_text)
        y = float(y_text)
        draw_box(draw, x, y, box_size, "yellow", 2)
    if row["is_contact_outlier"] == "1":
        draw.rectangle((4, 4, 170, 28), fill="black")
        draw.text((10, 9), "contact outlier rec_00007", fill="orange")
    if title:
        draw.rectangle((4, image.height - 28, image.width - 4, image.height - 4), fill="black")
        draw.text((10, image.height - 23), title, fill="white")
    heatmap = heatmap_preview(row["heatmap_path"], image.size) if "heatmap_path" in row else None
    if heatmap is None:
        canvas = image
    else:
        canvas = Image.new("RGB", (image.width * 2, image.height), "black")
        canvas.paste(image, (0, 0))
        canvas.paste(heatmap, (image.width, 0))
    ensure_dir(output_path.parent)
    canvas.save(output_path)


def save_debug_predictions(
    predictions: list[dict],
    rows_by_name: dict[str, dict],
    output_dir: Path,
    limit: int,
    box_size: int,
) -> None:
    ensure_dir(output_dir)
    for idx, pred in enumerate(predictions[:limit]):
        row = {**pred, **rows_by_name[pred["image_name"]]}
        output_path = output_dir / f"{idx:03d}_{Path(pred['image_name']).stem}_proposal.jpg"
        title = f"err={pred['error_px']} box48={pred['box48_hit']} top5box48={pred['top5_box48_hit']}"
        draw_prediction_overlay(row, output_path, box_size, title)


def crop_mean_rgb(image_path: str, x: float, y: float, crop_size: int) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    half = crop_size // 2
    left = max(0, int(round(x)) - half)
    top = max(0, int(round(y)) - half)
    right = min(image.width, int(round(x)) + half)
    bottom = min(image.height, int(round(y)) + half)
    crop = image.crop((left, top, right, bottom))
    arr = np.asarray(crop, dtype=np.float32) / 255.0
    if arr.size == 0:
        return np.zeros(3, dtype=np.float32)
    return arr.reshape(-1, 3).mean(axis=0).astype(np.float32)


def cache_feature(row: dict, x: float, y: float, crop_size: int) -> np.ndarray:
    width = float(row["image_width"])
    height = float(row["image_height"])
    numeric = np.asarray(
        [
            x / width,
            y / height,
            float(row["tip_x"]) / width,
            float(row["tip_y"]) / height,
            float(row["base_x"]) / width,
            float(row["base_y"]) / height,
            float(row["direction_x"]),
            float(row["direction_y"]),
            float(row["probe"]) / 100.0,
        ],
        dtype=np.float32,
    )
    crop = crop_mean_rgb(row["vision_path"], x, y, crop_size)
    return np.concatenate([numeric, crop], axis=0)


def save_retrieval_debug(row: dict, output_path: Path, box_size: int) -> None:
    query = Image.open(row["query_vision_path"]).convert("RGB")
    retrieved = Image.open(row["retrieved_vision_path"]).convert("RGB")
    touch = Image.open(row["retrieved_touch_path"]).convert("RGB") if row["retrieved_touch_path"] else Image.new("RGB", query.size, "black")
    touch = touch.resize(query.size)
    for image, x_key, y_key, color in (
        (query, "query_pred_x", "query_pred_y", "magenta"),
        (retrieved, "retrieved_target_x", "retrieved_target_y", "lime"),
    ):
        draw = ImageDraw.Draw(image)
        x = float(row[x_key])
        y = float(row[y_key])
        draw_box(draw, x, y, box_size, color, 4)
    canvas = Image.new("RGB", (query.width * 3, query.height), "black")
    canvas.paste(query, (0, 0))
    canvas.paste(retrieved, (query.width, 0))
    canvas.paste(touch, (query.width * 2, 0))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, canvas.width, 28), fill="black")
    draw.text(
        (8, 8),
        f"query | retrieved | touch  dist={float(row['distance']):.4f} same_record={row['same_record']}",
        fill="white",
    )
    ensure_dir(output_path.parent)
    canvas.save(output_path)


def build_retrieval_outputs(
    predictions: list[dict],
    rows_by_name: dict[str, dict],
    crop_size: int,
    output_csv: Path,
    output_json: Path,
    debug_dir: Path,
    debug_samples: int,
    box_size: int,
) -> dict:
    train_predictions = [pred for pred in predictions if pred["dataset_split"] == "train"]
    query_predictions = [pred for pred in predictions if pred["dataset_split"] in {"val", "test"}]
    cache_vectors = []
    cache_rows = []
    for pred in train_predictions:
        source = rows_by_name[pred["image_name"]]
        x = float(source["target_tip_x"])
        y = float(source["target_tip_y"])
        cache_vectors.append(cache_feature(source, x, y, crop_size))
        cache_rows.append({**source, **pred})
    if not cache_vectors:
        write_csv_rows(output_csv, [], RETRIEVAL_FIELDS)
        summary = {"cache_size": 0, "queries": 0}
        write_json(output_json, summary)
        return summary

    cache_matrix = np.stack(cache_vectors, axis=0)
    retrieval_rows = []
    distances = []
    same_record_hits = []
    for pred in query_predictions:
        source = rows_by_name[pred["image_name"]]
        query_vec = cache_feature(source, float(pred["pred_x"]), float(pred["pred_y"]), crop_size)
        dists = np.linalg.norm(cache_matrix - query_vec[None, :], axis=1)
        best_idx = int(np.argmin(dists))
        best = cache_rows[best_idx]
        distance = float(dists[best_idx])
        same_record = pred["record_id"] == best["record_id"]
        distances.append(distance)
        same_record_hits.append(same_record)
        retrieval_rows.append(
            {
                "dataset_split": pred["dataset_split"],
                "query_split": pred["split"],
                "query_record_id": pred["record_id"],
                "query_image_name": pred["image_name"],
                "query_vision_path": pred["vision_path"],
                "query_pred_x": pred["pred_x"],
                "query_pred_y": pred["pred_y"],
                "query_target_x": pred["target_x"],
                "query_target_y": pred["target_y"],
                "retrieved_split": best["split"],
                "retrieved_record_id": best["record_id"],
                "retrieved_image_name": best["image_name"],
                "retrieved_vision_path": best["vision_path"],
                "retrieved_touch_path": best.get("touch_path", ""),
                "retrieved_target_x": best["target_x"],
                "retrieved_target_y": best["target_y"],
                "distance": f"{distance:.6f}",
                "same_record": "1" if same_record else "0",
            }
        )

    write_csv_rows(output_csv, retrieval_rows, RETRIEVAL_FIELDS)
    for idx, row in enumerate(retrieval_rows[:debug_samples]):
        output_path = debug_dir / f"{idx:03d}_{Path(row['query_image_name']).stem}_retrieval.jpg"
        save_retrieval_debug(row, output_path, box_size)
    summary = {
        "cache_size": len(cache_rows),
        "queries": len(retrieval_rows),
        "mean_distance": float(np.mean(distances)) if distances else None,
        "median_distance": float(np.median(distances)) if distances else None,
        "same_record_rate": float(np.mean(same_record_hits)) if same_record_hits else None,
        "output_csv": str(output_csv),
        "debug_dir": str(debug_dir),
    }
    write_json(output_json, summary)
    return summary


def train_contact_region(
    config_path: str,
    epochs_override: int | None = None,
    eval_only: bool = False,
    section: str = "contact_region",
) -> dict:
    cfg = load_config(config_path)
    if section not in cfg:
        raise KeyError(f"Missing config section: {section}")
    region_cfg = cfg[section]
    rows = read_csv_rows(project_path(region_cfg["samples_csv"]))
    attach_touch_paths(rows, project_path(cfg["manifest"]["output_csv"]))
    rows_by_name = {row["image_name"]: row for row in rows}
    rows_by_split: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        rows_by_split[row["dataset_split"]].append(row)

    seed = int(region_cfg.get("seed", 42))
    set_seed(seed)
    input_width = int(region_cfg["input_width"])
    input_height = int(region_cfg["input_height"])
    batch_size = int(region_cfg["batch_size"])
    epochs = int(epochs_override or region_cfg["epochs"])
    topk = int(region_cfg["topk"])
    suppression_radius = int(region_cfg["topk_suppression_radius"])
    bbox_half_size = float(region_cfg["bbox_half_size"])
    contact_box_size = int(region_cfg.get("contact_box_size", 48))
    ttc_buckets = normalize_ttc_buckets(region_cfg.get("ttc_buckets"))
    motion_window_frames = int(region_cfg.get("motion_window_frames", 15))
    motion_tracks = read_motion_tracks(project_path(region_cfg["motion_tracks_csv"])) if region_cfg.get("motion_tracks_csv") else {}
    use_ttc_channel = bool(region_cfg.get("use_ttc_channel", False))
    ttc_normalizer = float(region_cfg.get("ttc_normalizer", 100.0))
    use_motion_channels = bool(region_cfg.get("use_motion_channels", False))
    temporal_fusion = str(region_cfg.get("temporal_fusion", "none"))
    use_trajectory_branch = temporal_fusion in {"predicted_ttc", "trajectory"}
    trajectory_history_frames = int(region_cfg.get("trajectory_history_frames", 32))
    trajectory_format = str(region_cfg.get("trajectory_format", "legacy"))
    trajectory_spatial_scale_px = float(region_cfg.get("trajectory_spatial_scale_px", 48.0))
    trajectory_speed_scale_px = float(region_cfg.get("trajectory_speed_scale_px", 4.0))
    displacement_scale_px = float(region_cfg.get("displacement_scale_px", 48.0))
    ttc_values = [int(value) for value in region_cfg.get("ttc_values", DEFAULT_TTC_VALUES)]
    trajectory_tracks = read_trajectory_tracks(project_path(region_cfg["motion_tracks_csv"])) if use_trajectory_branch else {}
    input_channels = 7 + (1 if use_ttc_channel else 0) + (4 if use_motion_channels else 0)

    dataset_kwargs = {
        "input_width": input_width,
        "input_height": input_height,
        "geometry_sigma": float(region_cfg["geometry_sigma"]),
        "use_ttc_channel": use_ttc_channel,
        "ttc_normalizer": ttc_normalizer,
        "use_motion_channels": use_motion_channels,
        "motion_tracks": motion_tracks,
        "motion_window_frames": motion_window_frames,
        "use_trajectory_branch": use_trajectory_branch,
        "trajectory_tracks": trajectory_tracks,
        "trajectory_history_frames": trajectory_history_frames,
        "trajectory_spatial_scale_px": trajectory_spatial_scale_px,
        "trajectory_speed_scale_px": trajectory_speed_scale_px,
        "displacement_scale_px": displacement_scale_px,
        "ttc_values": ttc_values,
        "trajectory_format": trajectory_format,
    }
    train_dataset = ContactRegionDataset(rows_by_split["train"], **dataset_kwargs)
    val_dataset = ContactRegionDataset(rows_by_split["val"], **dataset_kwargs)
    test_dataset = ContactRegionDataset(rows_by_split["test"], **dataset_kwargs)
    generator = torch.Generator().manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = max(int(region_cfg.get("num_workers", 0)), 0)
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": bool(region_cfg.get("pin_memory", device.type == "cuda")),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(region_cfg.get("persistent_workers", True))
        loader_kwargs["prefetch_factor"] = max(int(region_cfg.get("prefetch_factor", 2)), 1)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_batch,
        generator=generator,
        **loader_kwargs,
    )
    eval_train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch, **loader_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch, **loader_kwargs)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch, **loader_kwargs)
    if use_trajectory_branch:
        model = TemporalConditionedUNet(
            in_channels=input_channels,
            trajectory_hidden_size=int(region_cfg.get("trajectory_hidden_size", 64)),
            num_ttc_classes=len(ttc_values),
            fusion_mode=temporal_fusion,
            trajectory_format=trajectory_format,
        )
        ttc_checkpoint = project_path(region_cfg.get("ttc_checkpoint", "checkpoints/ttc_estimator/best.pt"))
        if not ttc_checkpoint.exists():
            raise FileNotFoundError(f"Train the TTC estimator first; missing: {ttc_checkpoint}")
        model.load_ttc_checkpoint(ttc_checkpoint)
        if bool(region_cfg.get("freeze_ttc_estimator", temporal_fusion == "predicted_ttc")):
            model.freeze_ttc()
        model = model.to(device)
    else:
        model = TinyUNet(in_channels=input_channels).to(device)
    criterion = nn.MSELoss()
    ttc_criterion = nn.CrossEntropyLoss()
    displacement_criterion = nn.SmoothL1Loss()
    ttc_loss_weight = float(region_cfg.get("ttc_loss_weight", 0.0))
    displacement_loss_weight = float(region_cfg.get("displacement_loss_weight", 0.0))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(region_cfg["learning_rate"]),
        weight_decay=float(region_cfg.get("weight_decay", 0.0)),
    )
    checkpoint_dir = ensure_dir(project_path(region_cfg["checkpoint_dir"]))
    best_path = checkpoint_dir / "best.pt"
    last_path = checkpoint_dir / "last.pt"
    best_val = float("inf")
    best_epoch = 0
    stale_epochs = 0
    early_stopping_patience = max(int(region_cfg.get("early_stopping_patience", 0)), 0)
    early_stopping_min_delta = max(float(region_cfg.get("early_stopping_min_delta", 0.0)), 0.0)
    amp_enabled = device.type == "cuda" and bool(region_cfg.get("amp", False))
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    history = []
    start = time.time()

    if eval_only and not best_path.exists():
        raise FileNotFoundError(f"Missing checkpoint for --eval-only: {best_path}")

    if not eval_only:
        for epoch in range(1, epochs + 1):
            model.train()
            train_losses = []
            for batch in train_loader:
                targets = batch["target"].to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                    output = forward_contact_model(model, batch, device)
                    loss = criterion(output["heatmap"], targets)
                    if "ttc_logits" in output and ttc_loss_weight > 0.0:
                        loss = loss + ttc_loss_weight * ttc_criterion(output["ttc_logits"], batch["ttc_class"].to(device))
                    if "displacement" in output and displacement_loss_weight > 0.0:
                        loss = loss + displacement_loss_weight * displacement_criterion(output["displacement"], batch["displacement"].to(device))
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_losses.append(float(loss.item()))
            val_summary, _ = evaluate(
                model,
                val_loader,
                device,
                input_width,
                input_height,
                "val",
                topk,
                suppression_radius,
                bbox_half_size,
                ttc_buckets,
                motion_tracks,
                motion_window_frames,
                ttc_values,
                displacement_scale_px,
            )
            train_loss = float(np.mean(train_losses)) if train_losses else 0.0
            val_loss = float(val_summary["loss"]) if val_summary["loss"] is not None else 0.0
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_median_error_px": val_summary["median_error_px"],
                    "val_pck_48": val_summary["pck_48"],
                    "val_top5_hit_48": val_summary["top5_hit_48"],
                }
            )
            state = {
                "model": model.state_dict(),
                "config": region_cfg,
                "epoch": epoch,
                "val_loss": val_loss,
                "val_summary": val_summary,
            }
            torch.save(state, last_path)
            if val_loss < best_val - early_stopping_min_delta:
                best_val = val_loss
                best_epoch = epoch
                stale_epochs = 0
                torch.save(state, best_path)
            else:
                stale_epochs += 1
            if epoch == 1 or epoch == epochs or epoch % 10 == 0:
                print(
                    f"epoch={epoch:03d} train_loss={train_loss:.6f} "
                    f"val_loss={val_loss:.6f} val_median_px={val_summary['median_error_px']:.3f} "
                    f"val_pck48={val_summary['pck_48']:.3f}"
                )
            if early_stopping_patience > 0 and stale_epochs >= early_stopping_patience:
                print(f"early_stop epoch={epoch:03d} best_epoch={best_epoch:03d} stale_epochs={stale_epochs}")
                break

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    if eval_only:
        history = checkpoint.get("history", [])
    train_summary, train_predictions = evaluate(
        model,
        eval_train_loader,
        device,
        input_width,
        input_height,
        "train",
        topk,
        suppression_radius,
        bbox_half_size,
        ttc_buckets,
        motion_tracks,
        motion_window_frames,
        ttc_values,
        displacement_scale_px,
    )
    val_summary, val_predictions = evaluate(
        model,
        val_loader,
        device,
        input_width,
        input_height,
        "val",
        topk,
        suppression_radius,
        bbox_half_size,
        ttc_buckets,
        motion_tracks,
        motion_window_frames,
        ttc_values,
        displacement_scale_px,
    )
    test_summary, test_predictions = evaluate(
        model,
        test_loader,
        device,
        input_width,
        input_height,
        "test",
        topk,
        suppression_radius,
        bbox_half_size,
        ttc_buckets,
        motion_tracks,
        motion_window_frames,
        ttc_values,
        displacement_scale_px,
    )
    all_predictions = train_predictions + val_predictions + test_predictions

    predictions_path = project_path(region_cfg["predictions_csv"])
    metrics_path = project_path(region_cfg["metrics_json"])
    retrieval_csv = project_path(region_cfg["retrieval_csv"])
    retrieval_json = project_path(region_cfg["retrieval_json"])
    debug_dir = project_path(region_cfg["debug_dir"])
    retrieval_debug_dir = project_path(region_cfg["retrieval_debug_dir"])
    write_csv_rows(predictions_path, all_predictions, PREDICTION_FIELDS)
    save_debug_predictions(
        val_predictions + test_predictions,
        rows_by_name,
        debug_dir,
        int(region_cfg["debug_samples"]),
        contact_box_size,
    )
    retrieval_summary = build_retrieval_outputs(
        all_predictions,
        rows_by_name,
        int(region_cfg["cache_crop_size"]),
        retrieval_csv,
        retrieval_json,
        retrieval_debug_dir,
        int(region_cfg["debug_samples"]),
        contact_box_size,
    )

    summary = {
        "device": str(device),
        "config_section": section,
        "seed": seed,
        "input_channels": input_channels,
        "input_width": input_width,
        "input_height": input_height,
        "contact_box_size": contact_box_size,
        "use_ttc_channel": use_ttc_channel,
        "ttc_normalizer": ttc_normalizer,
        "use_motion_channels": use_motion_channels,
        "temporal_fusion": temporal_fusion,
        "trajectory_history_frames": trajectory_history_frames if use_trajectory_branch else None,
        "trajectory_format": trajectory_format if use_trajectory_branch else None,
        "ttc_loss_weight": ttc_loss_weight,
        "displacement_loss_weight": displacement_loss_weight,
        "ttc_buckets": {name: sorted(values) for name, values in ttc_buckets.items()},
        "motion_direction": {
            "source": region_cfg.get("motion_tracks_csv", ""),
            "window_frames": motion_window_frames,
            "method": "least_squares_tip_velocity",
        },
        "epochs": epochs,
        "best_epoch": best_epoch,
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_min_delta": early_stopping_min_delta,
        "data_loader": {"num_workers": num_workers, "pin_memory": loader_kwargs["pin_memory"]},
        "amp_enabled": amp_enabled,
        "eval_only": eval_only,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "elapsed_seconds": round(time.time() - start, 2),
        "split_counts": {name: len(items) for name, items in rows_by_split.items()},
        "contact_outlier_record": "rec_00007",
        "train": train_summary,
        "val": val_summary,
        "test": test_summary,
        "retrieval": retrieval_summary,
        "history": history,
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "predictions_csv": str(predictions_path),
        "debug_dir": str(debug_dir),
    }
    write_json(metrics_path, summary)
    print(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Phase 2 future contact-region baseline.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--section", default="contact_region")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()
    train_contact_region(args.config, args.epochs, args.eval_only, args.section)


if __name__ == "__main__":
    main()
