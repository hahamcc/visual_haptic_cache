"""Factorized tactile representations for Phase4H.2 diagnostics."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


SHAPE_GRID = 8
SHAPE_DIM = SHAPE_GRID * SHAPE_GRID + 6
INTENSITY_DIM = 14
INTENSITY_FIELDS = (
    "global_energy",
    "active_mean",
    "active_std",
    "active_p50",
    "active_p75",
    "active_p90",
    "active_p95",
    "active_peak",
    "active_r_mean",
    "active_g_mean",
    "active_b_mean",
    "active_r_std",
    "active_g_std",
    "active_b_std",
)


def grayscale(diff: np.ndarray) -> np.ndarray:
    if diff.ndim != 3 or diff.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 tactile difference, got {diff.shape}")
    return (
        0.299 * diff[..., 0] + 0.587 * diff[..., 1] + 0.114 * diff[..., 2]
    ).astype(np.float32)


def factorized_tactile_latents(
    diff: np.ndarray,
    threshold: float = 0.04,
    grid: int = SHAPE_GRID,
) -> tuple[np.ndarray, np.ndarray]:
    """Split tactile supervision into spatial shape and deformation intensity."""
    gray = grayscale(diff)
    height, width = gray.shape
    cell_h, cell_w = height // grid, width // grid
    if cell_h == 0 or cell_w == 0:
        raise ValueError(f"Tactile image {gray.shape} is too small for {grid}x{grid}")
    mask = gray >= float(threshold)
    pooled_mask = (
        mask[: cell_h * grid, : cell_w * grid]
        .reshape(grid, cell_h, grid, cell_w)
        .mean(axis=(1, 3))
        .astype(np.float32)
    )
    area = float(mask.mean())
    ys, xs = np.where(mask)
    if len(xs):
        x = xs.astype(np.float32) / max(width - 1, 1)
        y = ys.astype(np.float32) / max(height - 1, 1)
        cx, cy = float(x.mean()), float(y.mean())
        moments = np.asarray(
            [
                np.mean((x - cx) ** 2),
                np.mean((y - cy) ** 2),
                np.mean((x - cx) * (y - cy)),
            ],
            dtype=np.float32,
        )
        active_gray = gray[mask]
        active_rgb = diff[mask]
        quantiles = np.quantile(
            active_gray,
            [0.50, 0.75, 0.90, 0.95],
            method="linear",
        ).astype(np.float32)
        active_statistics = np.asarray(
            [
                active_gray.mean(),
                active_gray.std(),
                *quantiles,
                active_gray.max(),
            ],
            dtype=np.float32,
        )
        rgb_mean = active_rgb.mean(axis=0).astype(np.float32)
        rgb_std = active_rgb.std(axis=0).astype(np.float32)
    else:
        cx, cy = 0.5, 0.5
        moments = np.zeros(3, dtype=np.float32)
        active_statistics = np.zeros(7, dtype=np.float32)
        rgb_mean = np.zeros(3, dtype=np.float32)
        rgb_std = np.zeros(3, dtype=np.float32)
    shape = np.concatenate(
        (
            pooled_mask.reshape(-1),
            np.asarray([area, cx, cy], dtype=np.float32),
            moments,
        )
    ).astype(np.float32)
    intensity = np.concatenate(
        (
            np.asarray([gray.mean()], dtype=np.float32),
            active_statistics,
            rgb_mean,
            rgb_std,
        )
    ).astype(np.float32)
    if shape.shape != (grid * grid + 6,):
        raise RuntimeError(f"Expected {grid * grid + 6}-D shape latent, got {shape.shape}")
    if intensity.shape != (INTENSITY_DIM,):
        raise RuntimeError(
            f"Expected {INTENSITY_DIM}-D intensity latent, got {intensity.shape}"
        )
    if not np.isfinite(shape).all() or not np.isfinite(intensity).all():
        raise RuntimeError("Factorized tactile latent contains non-finite values")
    return shape, intensity


class LowCapacityIntensityRegressor(nn.Module):
    """Small diagnostic head; the DINO backbone remains frozen."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        output_dim: int = INTENSITY_DIM,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, online_features: torch.Tensor) -> torch.Tensor:
        return self.network(online_features)


def standardized_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return mean squared distances from N query vectors to N×K candidates."""
    if left.ndim != 2 or right.ndim != 3 or left.shape[0] != right.shape[0]:
        raise ValueError(f"Incompatible latent shapes: {left.shape}, {right.shape}")
    return ((left[:, None] - right) ** 2).mean(axis=-1).astype(np.float32)
