"""Shared, online-safe components for Phase4H DINO-to-tactile cache ranking."""
from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

from .temporal_progress import DEFAULT_TTC_VALUES, masked_trajectory_features
from .train_phase4b_predicted_box_cache_ranker import is_final_holdout
from .utils import read_csv_rows


TACTILE_LATENT_DIM = 77
PHASE4H_QUERY_FORBIDDEN_FIELDS = frozenset(
    {
        "touch",
        "touch_path",
        "target_ttc",
        "target_class",
        "contact_frame",
        "contact_frame_detected",
        "contact_frame_from_name",
    }
)


def _overlap_ratio(width: int, height: int, x: float, y: float, size: int) -> float:
    left, top = int(round(x - size / 2.0)), int(round(y - size / 2.0))
    right, bottom = left + size, top + size
    overlap_w = max(0, min(width, right) - max(0, left))
    overlap_h = max(0, min(height, bottom) - max(0, top))
    return float(overlap_w * overlap_h / max(size * size, 1))


def _reflect_crop(array: np.ndarray, x: float, y: float, size: int) -> np.ndarray:
    """Crop a square with reflection padding and no synthetic black border."""
    left, top = int(round(x - size / 2.0)), int(round(y - size / 2.0))
    right, bottom = left + size, top + size
    pad_left, pad_top = max(0, -left), max(0, -top)
    pad_right, pad_bottom = max(0, right - array.shape[1]), max(0, bottom - array.shape[0])
    mode = "reflect" if min(array.shape[:2]) > 1 else "edge"
    if pad_left or pad_top or pad_right or pad_bottom:
        array = np.pad(
            array,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode=mode,
        )
    left, top = left + pad_left, top + pad_top
    crop = array[top : top + size, left : left + size]
    if crop.shape[:2] != (size, size):
        raise RuntimeError(f"Reflection crop produced {crop.shape[:2]}, expected {(size, size)}")
    return crop


def contact_crop_reflect(
    image_path: str | Path,
    x: float,
    y: float,
    size: int,
    rotation_degrees: float = 0.0,
) -> tuple[np.ndarray, float]:
    """Return a contact-centered RGB crop and the original-field padding ratio.

    Rotation is performed on a sqrt(2)-larger reflected crop before taking the
    final center crop, so rotated corners never introduce a black fill value.
    """
    with Image.open(image_path) as handle:
        image = handle.convert("RGB")
        array = np.asarray(image, dtype=np.uint8)
        padding_ratio = 1.0 - _overlap_ratio(image.width, image.height, x, y, size)
    if abs(rotation_degrees) < 1e-6:
        crop = _reflect_crop(array, x, y, size)
    else:
        outer = int(math.ceil(size * math.sqrt(2.0))) + 4
        if outer % 2 != size % 2:
            outer += 1
        crop = _reflect_crop(array, x, y, outer)
        rotated = Image.fromarray(crop).rotate(
            float(rotation_degrees),
            resample=Image.Resampling.BICUBIC,
            expand=False,
        )
        offset = (outer - size) // 2
        crop = np.asarray(rotated.crop((offset, offset, offset + size, offset + size)), dtype=np.uint8)
    return crop.astype(np.float32) / 255.0, float(padding_ratio)


def _sensor_vector(row: dict[str, str]) -> np.ndarray:
    return np.asarray(
        [float(row["tip_x"]) - float(row["base_x"]), float(row["tip_y"]) - float(row["base_y"])],
        dtype=np.float32,
    )


def recent_motion_vector(
    row: dict[str, str],
    tracks: dict[tuple[str, str], list[dict[str, float]]],
    history_frames: int = 16,
) -> tuple[np.ndarray, bool]:
    current = int(row["frame_id"])
    points = [
        point
        for point in tracks.get((row["split"], row["record_id"]), [])
        if current - history_frames + 1 <= int(point["frame_id"]) <= current
    ]
    points.sort(key=lambda item: int(item["frame_id"]))
    valid = (
        len(points) >= history_frames
        and int(points[-1]["frame_id"]) - int(points[0]["frame_id"]) >= history_frames - 1
    )
    if valid:
        vector = np.asarray(
            [points[-1]["tip_x"] - points[0]["tip_x"], points[-1]["tip_y"] - points[0]["tip_y"]],
            dtype=np.float32,
        )
        if float(np.linalg.norm(vector)) >= 1.0:
            return vector, True
    return _sensor_vector(row), False


def canonical_rotation_degrees(
    row: dict[str, str],
    mode: str,
    tracks: dict[tuple[str, str], list[dict[str, float]]] | None = None,
    history_frames: int = 16,
) -> tuple[float, str]:
    """Rotate the approach vector to image-down, using only pre-query state."""
    if mode == "raw":
        return 0.0, "raw"
    if mode == "sensor_axis":
        vector, used = _sensor_vector(row), "sensor_axis"
    elif mode == "motion_axis":
        vector, valid = recent_motion_vector(row, tracks or {}, history_frames)
        used = "motion_axis" if valid else "sensor_axis_fallback"
    else:
        raise ValueError(f"Unsupported canonicalization mode: {mode}")
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-6:
        return 0.0, f"{used}_zero_vector"
    screen_angle = math.degrees(math.atan2(float(vector[1]), float(vector[0])))
    # PIL's positive angle is counter-clockwise, while image y grows down.
    return float(screen_angle - 90.0), used


def tactile_latent(diff: np.ndarray, threshold: float = 0.04, grid: int = 8) -> np.ndarray:
    """Build the fixed 77-D tactile cache index described by the Phase4H protocol."""
    if diff.ndim != 3 or diff.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 tactile difference, got {diff.shape}")
    gray = (0.299 * diff[..., 0] + 0.587 * diff[..., 1] + 0.114 * diff[..., 2]).astype(np.float32)
    height, width = gray.shape
    cell_h, cell_w = height // grid, width // grid
    if cell_h == 0 or cell_w == 0:
        raise ValueError(f"Tactile image {gray.shape} is too small for {grid}x{grid} pooling")
    pooled = gray[: cell_h * grid, : cell_w * grid].reshape(grid, cell_h, grid, cell_w).mean(axis=(1, 3))
    rgb_mean = diff.reshape(-1, 3).mean(axis=0)
    rgb_std = diff.reshape(-1, 3).std(axis=0)
    mask = gray >= float(threshold)
    area = float(mask.mean())
    ys, xs = np.where(mask)
    if len(xs):
        x = xs.astype(np.float32) / max(width - 1, 1)
        y = ys.astype(np.float32) / max(height - 1, 1)
        cx, cy = float(x.mean()), float(y.mean())
        moments = np.asarray(
            [np.mean((x - cx) ** 2), np.mean((y - cy) ** 2), np.mean((x - cx) * (y - cy))],
            dtype=np.float32,
        )
    else:
        cx, cy = 0.5, 0.5
        moments = np.zeros(3, dtype=np.float32)
    energy = float(gray.mean())
    result = np.concatenate(
        [
            pooled.reshape(-1),
            rgb_mean,
            rgb_std,
            np.asarray([area, cx, cy], dtype=np.float32),
            moments,
            np.asarray([energy], dtype=np.float32),
        ]
    ).astype(np.float32)
    if result.shape != (TACTILE_LATENT_DIM,):
        raise RuntimeError(f"Expected {TACTILE_LATENT_DIM}-D tactile latent, got {result.shape}")
    return result


def combine_layer_tokens(layer_tokens: dict[int, torch.Tensor], recipe: str) -> torch.Tensor:
    if recipe.startswith("layer"):
        layer = int(recipe.removeprefix("layer"))
        return layer_tokens[layer]
    if recipe == "mean_8_10_12":
        normalized = [F.normalize(layer_tokens[layer], dim=-1) for layer in (8, 10, 12)]
        return torch.stack(normalized, dim=0).mean(dim=0)
    raise ValueError(f"Unsupported DINO layer recipe: {recipe}")


def gaussian_token_weights(tokens: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    side = int(round(math.sqrt(tokens)))
    if side * side != tokens:
        raise ValueError(f"Expected a square token grid, got {tokens} tokens")
    axis = torch.linspace(-1.0, 1.0, side, device=device, dtype=dtype)
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    weights = torch.exp(-(x.square() + y.square()) / (2.0 * sigma * sigma)).reshape(-1)
    return weights / weights.sum().clamp_min(torch.finfo(dtype).eps)


def pooled_token_features(tokens: torch.Tensor, sigma: float = 0.35) -> torch.Tensor:
    """Contact-aware pooling: global mean, center-weighted mean and weighted std."""
    weights = gaussian_token_weights(tokens.shape[1], sigma, tokens.device, tokens.dtype)
    center = torch.einsum("t,btd->bd", weights, tokens)
    variance = torch.einsum("t,btd->bd", weights, (tokens - center[:, None]).square())
    return torch.cat((tokens.mean(dim=1), center, torch.sqrt(variance.clamp_min(1e-8))), dim=1)


def _token_positions(tokens: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    side = int(round(math.sqrt(tokens)))
    if side * side != tokens:
        raise ValueError(f"Expected square token grid, got {tokens}")
    axis = torch.linspace(-1.0, 1.0, side, device=device, dtype=dtype)
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack((x.reshape(-1), y.reshape(-1)), dim=1)


def position_aware_soft_similarity(
    query: torch.Tensor,
    cache: torch.Tensor,
    radius: int = 2,
    temperature: float = 0.07,
    position_penalty: float = 0.25,
    center_sigma: float = 0.35,
) -> torch.Tensor:
    """Symmetric local soft correspondence with spatial and center priors.

    Args:
        query: [batch, tokens, channels].
        cache: [batch, candidates, tokens, channels].
    Returns:
        Similarity [batch, candidates], where larger is better.
    """
    if query.ndim != 3 or cache.ndim != 4 or query.shape[0] != cache.shape[0]:
        raise ValueError(f"Unexpected query/cache shapes: {query.shape}, {cache.shape}")
    if query.shape[1] != cache.shape[2] or query.shape[2] != cache.shape[3]:
        raise ValueError(f"Token grids/channels differ: {query.shape}, {cache.shape}")
    positions = _token_positions(query.shape[1], query.device, query.dtype)
    side = int(round(math.sqrt(query.shape[1])))
    grid = positions * max((side - 1) / 2.0, 1.0)
    chebyshev = (grid[:, None] - grid[None]).abs().amax(dim=-1)
    allowed = chebyshev <= float(radius)
    squared_offset = (positions[:, None] - positions[None]).square().sum(dim=-1)
    cosine = torch.einsum(
        "btd,bksd->bkts",
        F.normalize(query, dim=-1),
        F.normalize(cache, dim=-1),
    )
    logits = (cosine - position_penalty * squared_offset[None, None]) / max(temperature, 1e-6)
    logits = logits.masked_fill(~allowed[None, None], torch.finfo(logits.dtype).min)
    q_weights = torch.softmax(logits, dim=-1)
    c_weights = torch.softmax(logits, dim=-2)
    # Geometric mean retains correspondences that are strong in both
    # directions and suppresses one-way matches caused by repeated texture.
    mutual = torch.sqrt((q_weights * c_weights).clamp_min(0.0))
    q_score = (mutual * cosine).sum(dim=-1) / mutual.sum(dim=-1).clamp_min(1e-8)
    c_score = (mutual * cosine).sum(dim=-2) / mutual.sum(dim=-2).clamp_min(1e-8)
    center = gaussian_token_weights(query.shape[1], center_sigma, query.device, query.dtype)
    return 0.5 * (
        torch.einsum("t,bkt->bk", center, q_score)
        + torch.einsum("t,bkt->bk", center, c_score)
    )


def parse_ttc_probabilities(
    prediction: dict[str, str],
    ttc_values: Iterable[int] = DEFAULT_TTC_VALUES,
) -> tuple[np.ndarray, float, float]:
    axis = np.asarray(list(ttc_values), dtype=np.float32)
    raw = prediction.get("probabilities", "")
    if raw:
        values = np.asarray([float(value) for value in raw.split(";")], dtype=np.float32)
        if values.shape != axis.shape:
            raise ValueError(f"TTC probability count {len(values)} does not match axis {len(axis)}")
        values = np.maximum(values, 1e-12)
        values /= values.sum()
    elif prediction.get("predicted_ttc", ""):
        estimate = float(prediction["predicted_ttc"])
        values = np.exp(-0.5 * ((axis - estimate) / 20.0) ** 2)
        values /= values.sum()
    else:
        raise ValueError("Deployable TTC prediction requires probabilities or predicted_ttc")
    expected = float(np.sum(values * axis))
    entropy = float(-np.sum(values * np.log(values)) / math.log(len(values)))
    return values.astype(np.float32), expected, entropy


def deployable_motion_feature(
    row: dict[str, str],
    x: float,
    y: float,
    trajectory: np.ndarray,
    trajectory_mask: np.ndarray,
    quality: dict[str, float],
    ttc_prediction: dict[str, str],
    ttc_values: Iterable[int] = DEFAULT_TTC_VALUES,
) -> np.ndarray:
    """Online-safe geometry, masked motion and predicted-TTC summary.

    True probe/contact-frame values are intentionally absent.
    """
    width, height = max(float(row["image_width"]), 1.0), max(float(row["image_height"]), 1.0)
    tip = np.asarray([float(row["tip_x"]), float(row["tip_y"])], dtype=np.float32)
    base = np.asarray([float(row["base_x"]), float(row["base_y"])], dtype=np.float32)
    point = np.asarray([x, y], dtype=np.float32)
    scale = max(width, height)
    valid = np.flatnonzero(trajectory_mask > 0.5)
    if len(valid):
        last = trajectory[valid[-1]]
        velocity, acceleration = last[8:10], last[10:12]
        speed, stability, cumulative = float(last[13]), float(last[14]), float(last[12])
    else:
        velocity = acceleration = np.zeros(2, dtype=np.float32)
        speed = stability = cumulative = 0.0
    probabilities, expected_ttc, entropy = parse_ttc_probabilities(ttc_prediction, ttc_values)
    geometry = np.asarray(
        [
            point[0] / width,
            point[1] / height,
            tip[0] / width,
            tip[1] / height,
            base[0] / width,
            base[1] / height,
            (point[0] - tip[0]) / width,
            (point[1] - tip[1]) / height,
            (tip[0] - base[0]) / width,
            (tip[1] - base[1]) / height,
            np.linalg.norm(point - tip) / scale,
            float(row.get("direction_x") or 0.0),
            float(row.get("direction_y") or 0.0),
            float(velocity[0]),
            float(velocity[1]),
            float(acceleration[0]),
            float(acceleration[1]),
            speed,
            stability,
            cumulative,
            quality["real_point_count"] / max(len(trajectory_mask), 1),
            quality["history_span_frames"] / max(len(trajectory_mask) - 1, 1),
            quality["padding_ratio"],
            quality["max_frame_gap"] / max(len(trajectory_mask), 1),
            quality["cumulative_displacement"] / max(scale, 1.0),
            expected_ttc / 100.0,
            entropy,
            1.0 - entropy,
        ],
        dtype=np.float32,
    )
    return np.concatenate((geometry, probabilities)).astype(np.float32)


class TactileLatentProjector(nn.Module):
    """Low-capacity visual-motion projector; no query tactile input exists."""

    def __init__(self, input_dim: int, latent_dim: int = TACTILE_LATENT_DIM, dropout: float = 0.1) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, latent_dim),
        )

    def forward(self, visual_motion: torch.Tensor) -> torch.Tensor:
        if visual_motion.shape[-1] != self.input_dim:
            raise ValueError(f"Expected {self.input_dim} online features, got {visual_motion.shape[-1]}")
        return self.network(visual_motion)


class DinoSafetyGate(nn.Module):
    def __init__(self, input_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


def final_holdout_keys(partition_csv: str | Path | None) -> set[tuple[str, str]]:
    if partition_csv is None:
        return set()
    if not Path(partition_csv).exists():
        raise FileNotFoundError(f"Phase4H final-partition manifest is required: {partition_csv}")
    return {
        (row["split"], row["record_id"])
        for row in read_csv_rows(partition_csv)
        if row.get("partition") == "final_holdout" or row.get("purpose") == "one_time_final_evaluation_only"
    }


def assert_development_only(
    rows: list[dict[str, str]],
    partition_csv: str | Path | None = None,
) -> None:
    sealed = final_holdout_keys(partition_csv)
    for row in rows:
        if is_final_holdout(row) or (row.get("split", ""), row["record_id"]) in sealed:
            raise RuntimeError(
                f"Phase4H refuses sealed final-holdout row {row.get('split')}/{row['record_id']}"
            )


def candidate_groups(
    rows: list[dict[str, str]],
    top_k: int,
) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["query_image_name"]].append(row)
    for name, group in groups.items():
        if len(group) != top_k:
            raise RuntimeError(f"Expected exactly Top-{top_k} candidates for {name}, got {len(group)}")
        keys = {(row["candidate_record_id"], row["candidate_image_name"]) for row in group}
        if len(keys) != top_k:
            raise RuntimeError(f"Duplicate candidate entries for {name}")
    return groups


def candidate_set_fingerprint(groups: dict[str, list[dict[str, str]]]) -> str:
    payload = [
        (query, sorted((row["candidate_record_id"], row["candidate_image_name"]) for row in group))
        for query, group in sorted(groups.items())
    ]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode("utf-8")).hexdigest()


def assert_candidate_identity(
    reference: dict[str, list[dict[str, str]]],
    candidate: dict[str, list[dict[str, str]]],
) -> None:
    if set(reference) != set(candidate):
        raise RuntimeError("Candidate query sets differ from the frozen V1 protocol")
    for query in reference:
        left = {(row["candidate_record_id"], row["candidate_image_name"]) for row in reference[query]}
        right = {(row["candidate_record_id"], row["candidate_image_name"]) for row in candidate[query]}
        if left != right:
            raise RuntimeError(f"Top-K candidate identity differs for {query}")


def record_hash_split(record_id: str, validation_fraction: float, seed: int) -> bool:
    digest = hashlib.sha256(f"{seed}:{record_id}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64)
    return value < validation_fraction
