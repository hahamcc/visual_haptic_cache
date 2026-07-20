from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from .utils import read_csv_rows


DEFAULT_TTC_VALUES = [5, 10, 20, 30, 50, 75, 100]
TRAJECTORY_FEATURE_SIZE = 15
MASKED_TRAJECTORY_FEATURE_SIZE = 17
VELOCITY_SLICE = slice(8, 10)


def read_trajectory_tracks(path: str | Path) -> dict[tuple[str, str], list[dict[str, float]]]:
    tracks: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)
    if not Path(path).exists():
        return tracks
    for row in read_csv_rows(path):
        tracks[(row["split"], row["record_id"])].append(
            {
                "frame_id": int(row["frame_id"]),
                "tip_x": float(row["tip_x"]),
                "tip_y": float(row["tip_y"]),
                "base_x": float(row["base_x"]),
                "base_y": float(row["base_y"]),
            }
        )
    for points in tracks.values():
        points.sort(key=lambda item: item["frame_id"])
    return tracks


def _interpolate_history(
    points: list[dict[str, float]], current_frame: int, history_frames: int
) -> np.ndarray:
    past = [point for point in points if point["frame_id"] <= current_frame]
    if not past:
        return np.zeros((history_frames, 4), dtype=np.float32)

    source_frames = np.asarray([point["frame_id"] for point in past], dtype=np.float32)
    target_frames = np.arange(current_frame - history_frames + 1, current_frame + 1, dtype=np.float32)
    channels = []
    for name in ("tip_x", "tip_y", "base_x", "base_y"):
        values = np.asarray([point[name] for point in past], dtype=np.float32)
        channels.append(np.interp(target_frames, source_frames, values, left=values[0], right=values[-1]))
    return np.stack(channels, axis=1).astype(np.float32)


def trajectory_features(
    row: dict[str, str],
    tracks: dict[tuple[str, str], list[dict[str, float]]],
    history_frames: int = 32,
    spatial_scale_px: float = 48.0,
    speed_scale_px: float = 4.0,
) -> np.ndarray:
    """Build online-safe features using only trajectory samples at or before the query frame."""
    key = (row["split"], row["record_id"])
    current_frame = int(row["frame_id"])
    history = _interpolate_history(tracks.get(key, []), current_frame, history_frames)
    tip = history[:, :2]
    base = history[:, 2:]
    velocity = np.diff(tip, axis=0, prepend=tip[:1])
    acceleration = np.diff(velocity, axis=0, prepend=velocity[:1])
    step_distance = np.linalg.norm(velocity, axis=1)
    cumulative = np.cumsum(step_distance)
    cumulative -= cumulative[0]

    width = max(float(row.get("image_width", 768.0)), 1.0)
    height = max(float(row.get("image_height", 512.0)), 1.0)
    coordinate_scale = np.asarray([width, height], dtype=np.float32)
    tip_absolute = tip / coordinate_scale
    base_absolute = base / coordinate_scale

    tip_relative = (tip - tip[-1:]) / max(spatial_scale_px, 1.0)
    base_relative = (base - tip) / max(spatial_scale_px, 1.0)
    velocity = velocity / max(speed_scale_px, 1.0)
    acceleration = acceleration / max(speed_scale_px, 1.0)
    cumulative = cumulative[:, None] / max(spatial_scale_px, 1.0)
    speed = np.linalg.norm(velocity, axis=1, keepdims=True)
    unit_velocity = velocity / np.maximum(speed, 1e-6)
    direction_stability = np.zeros((history_frames, 1), dtype=np.float32)
    for index in range(1, history_frames):
        valid = speed[: index + 1, 0] > 1e-6
        if np.any(valid):
            direction_stability[index, 0] = float(np.linalg.norm(unit_velocity[: index + 1][valid].mean(axis=0)))
    return np.concatenate(
        [tip_absolute, base_absolute, tip_relative, base_relative, velocity, acceleration, cumulative, speed, direction_stability],
        axis=1,
    ).astype(np.float32)


def displacement_target(row: dict[str, str], scale_px: float = 48.0) -> np.ndarray:
    return np.asarray(
        [
            (float(row["target_tip_x"]) - float(row["tip_x"])) / scale_px,
            (float(row["target_tip_y"]) - float(row["tip_y"])) / scale_px,
        ],
        dtype=np.float32,
    )


def masked_trajectory_features(
    row: dict[str, str],
    tracks: dict[tuple[str, str], list[dict[str, float]]],
    history_frames: int = 32,
    spatial_scale_px: float = 48.0,
    speed_scale_px: float = 4.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Return exact-frame features and a mask; missing frames are never interpolated."""
    current_frame = int(row["frame_id"])
    target_frames = list(range(current_frame - history_frames + 1, current_frame + 1))
    point_by_frame = {
        int(point["frame_id"]): point
        for point in tracks.get((row["split"], row["record_id"]), [])
        if target_frames[0] <= int(point["frame_id"]) <= current_frame
    }
    mask = np.asarray([1.0 if frame in point_by_frame else 0.0 for frame in target_frames], dtype=np.float32)
    features = np.zeros((history_frames, MASKED_TRAJECTORY_FEATURE_SIZE), dtype=np.float32)
    features[:, 15] = np.linspace(-1.0, 0.0, history_frames, dtype=np.float32)
    features[:, 16] = mask
    valid_indices = np.flatnonzero(mask > 0.5)
    if not len(valid_indices):
        return features, mask, {
            "real_point_count": 0.0, "history_span_frames": 0.0, "padding_ratio": 1.0,
            "repeated_point_ratio": 0.0, "max_frame_gap": 0.0, "cumulative_displacement": 0.0,
        }

    width = max(float(row.get("image_width", 768.0)), 1.0)
    height = max(float(row.get("image_height", 512.0)), 1.0)
    coordinate_scale = np.asarray([width, height], dtype=np.float32)
    current_point = point_by_frame[target_frames[valid_indices[-1]]]
    current_tip = np.asarray([current_point["tip_x"], current_point["tip_y"]], dtype=np.float32)
    previous_velocity = None
    cumulative = 0.0
    unit_velocities: list[np.ndarray] = []
    repeated = 0
    gaps = []
    previous_index = None
    previous_tip = None
    for index in valid_indices:
        point = point_by_frame[target_frames[index]]
        tip = np.asarray([point["tip_x"], point["tip_y"]], dtype=np.float32)
        base = np.asarray([point["base_x"], point["base_y"]], dtype=np.float32)
        features[index, 0:2] = tip / coordinate_scale
        features[index, 2:4] = base / coordinate_scale
        features[index, 4:6] = (tip - current_tip) / max(spatial_scale_px, 1.0)
        features[index, 6:8] = (base - tip) / max(spatial_scale_px, 1.0)
        if previous_index is not None and previous_tip is not None:
            dt = max(index - previous_index, 1)
            raw_velocity = (tip - previous_tip) / dt
            velocity = raw_velocity / max(speed_scale_px, 1.0)
            features[index, 8:10] = velocity
            if previous_velocity is not None:
                features[index, 10:12] = velocity - previous_velocity
            step = float(np.linalg.norm(tip - previous_tip))
            cumulative += step
            repeated += int(step <= 1e-6)
            gaps.append(dt)
            norm = float(np.linalg.norm(velocity))
            if norm > 1e-6:
                unit_velocities.append(velocity / norm)
            previous_velocity = velocity
        features[index, 12] = cumulative / max(spatial_scale_px, 1.0)
        features[index, 13] = float(np.linalg.norm(features[index, 8:10]))
        features[index, 14] = float(np.linalg.norm(np.mean(unit_velocities, axis=0))) if unit_velocities else 0.0
        previous_index = int(index)
        previous_tip = tip

    quality = {
        "real_point_count": float(len(valid_indices)),
        "history_span_frames": float(valid_indices[-1] - valid_indices[0]) if len(valid_indices) > 1 else 0.0,
        "padding_ratio": float(1.0 - len(valid_indices) / history_frames),
        "repeated_point_ratio": float(repeated / max(len(valid_indices) - 1, 1)),
        "max_frame_gap": float(max(gaps)) if gaps else 0.0,
        "cumulative_displacement": cumulative,
    }
    return features, mask, quality


def motion_basis(features: np.ndarray) -> tuple[np.ndarray, float]:
    velocity = features[:, VELOCITY_SLICE]
    recent = velocity[-8:].mean(axis=0)
    speed = float(np.linalg.norm(recent))
    if speed <= 1e-6:
        return np.asarray([0.0, 0.0], dtype=np.float32), 0.0
    return (recent / speed).astype(np.float32), speed


class TrajectoryEncoder(nn.Module):
    def __init__(self, input_size: int = TRAJECTORY_FEATURE_SIZE, hidden_size: int = 64) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.projection = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
        )

    def forward(self, sequence: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        if lengths is None:
            _, hidden = self.gru(sequence)
        else:
            packed = pack_padded_sequence(sequence, lengths.cpu(), batch_first=True, enforce_sorted=False)
            _, hidden = self.gru(packed)
        return self.projection(hidden[-1])


class TTCEstimator(nn.Module):
    def __init__(self, input_size: int = TRAJECTORY_FEATURE_SIZE, hidden_size: int = 64, num_classes: int = 7) -> None:
        super().__init__()
        self.encoder = TrajectoryEncoder(input_size, hidden_size)
        self.ttc_head = nn.Linear(hidden_size, num_classes)
        self.displacement_head = nn.Linear(hidden_size, 2)

    def forward(self, sequence: torch.Tensor) -> dict[str, torch.Tensor]:
        feature = self.encoder(sequence)
        return {
            "feature": feature,
            "ttc_logits": self.ttc_head(feature),
            "displacement": self.displacement_head(feature),
        }


class MaskedTrajectoryEncoder(nn.Module):
    def __init__(self, input_size: int = MASKED_TRAJECTORY_FEATURE_SIZE, hidden_size: int = 64) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell = nn.GRUCell(input_size, hidden_size)
        self.projection = nn.Sequential(
            nn.LayerNorm(hidden_size), nn.Linear(hidden_size, hidden_size), nn.ReLU(inplace=True)
        )

    def forward(self, sequence: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        hidden = torch.zeros(sequence.shape[0], self.hidden_size, device=sequence.device, dtype=sequence.dtype)
        for index in range(sequence.shape[1]):
            updated = self.cell(sequence[:, index], hidden)
            mask = valid_mask[:, index:index + 1].to(sequence.dtype)
            hidden = mask * updated + (1.0 - mask) * hidden
        return self.projection(hidden)


class MaskedTTCEstimator(nn.Module):
    def __init__(self, hidden_size: int = 64, num_classes: int = 7) -> None:
        super().__init__()
        self.encoder = MaskedTrajectoryEncoder(MASKED_TRAJECTORY_FEATURE_SIZE, hidden_size)
        self.ttc_head = nn.Linear(hidden_size, num_classes)
        self.displacement_head = nn.Linear(hidden_size, 2)

    def forward(self, sequence: torch.Tensor, valid_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        feature = self.encoder(sequence, valid_mask)
        return {
            "feature": feature,
            "ttc_logits": self.ttc_head(feature),
            "displacement": self.displacement_head(feature),
        }
