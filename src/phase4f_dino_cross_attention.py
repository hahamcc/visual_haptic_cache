"""Frozen DINOv2 token extraction and spatial query--cache matching for Phase 4F.

The module deliberately has no tactile input in ``forward``.  Tactile data is
used only by the offline trainer to create ranking targets.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


class FrozenDinoV2(nn.Module):
    """DINOv2 ViT-S/14 patch-token adapter loaded through the official Hub API."""

    def __init__(self, model_name: str = "dinov2_vits14", image_size: int = 224) -> None:
        super().__init__()
        self.image_size = int(image_size)
        self.model = torch.hub.load("facebookresearch/dinov2", model_name)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.feature_dim = int(getattr(self.model, "embed_dim", 384))
        self.register_buffer("mean", torch.tensor((0.485, 0.456, 0.406))[None, :, None, None])
        self.register_buffer("std", torch.tensor((0.229, 0.224, 0.225))[None, :, None, None])

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        images = F.interpolate(images, (self.image_size, self.image_size), mode="bicubic", align_corners=False)
        images = (images - self.mean) / self.std
        features = self.model.forward_features(images)
        tokens = features.get("x_norm_patchtokens") if isinstance(features, dict) else None
        if tokens is None:
            raise RuntimeError("DINOv2 backbone did not return x_norm_patchtokens; expected official ViT-S/14 Hub model.")
        return tokens


def spatial_tokens(tokens: torch.Tensor, width: int = 16) -> torch.Tensor:
    """Append deterministic normalized 2-D coordinates to patch tokens."""
    if tokens.ndim != 3 or tokens.shape[1] != width * width:
        raise ValueError(f"Expected [batch,{width * width},channels] patch tokens, got {tuple(tokens.shape)}")
    axis = torch.linspace(-1.0, 1.0, width, device=tokens.device, dtype=tokens.dtype)
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    coordinates = torch.stack((x, y), dim=-1).reshape(1, width * width, 2).expand(tokens.shape[0], -1, -1)
    return torch.cat((tokens, coordinates), dim=-1)


class PairCrossAttention(nn.Module):
    def __init__(self, token_dim: int, hidden_dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.project = nn.Sequential(nn.Linear(token_dim + 2, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.query_to_cache = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.cache_to_query = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.norm_query = nn.LayerNorm(hidden_dim)
        self.norm_cache = nn.LayerNorm(hidden_dim)

    def forward(self, query: torch.Tensor, cache: torch.Tensor) -> torch.Tensor:
        batch, candidates, tokens, channels = cache.shape
        query = self.project(spatial_tokens(query))
        cache = self.project(spatial_tokens(cache.reshape(batch * candidates, tokens, channels))).reshape(batch, candidates, tokens, -1)
        query = query[:, None].expand(-1, candidates, -1, -1).reshape(batch * candidates, tokens, -1)
        cache = cache.reshape(batch * candidates, tokens, -1)
        q_to_c, _ = self.query_to_cache(query, cache, cache, need_weights=False)
        c_to_q, _ = self.cache_to_query(cache, query, query, need_weights=False)
        q_to_c, c_to_q = self.norm_query(query + q_to_c), self.norm_cache(cache + c_to_q)
        return torch.cat((query.mean(1), cache.mean(1), q_to_c.mean(1), c_to_q.mean(1)), dim=1).reshape(batch, candidates, -1)


class DinoSpatialCacheRanker(nn.Module):
    """Ranks a fixed geometry shortlist using two-scale spatial token alignment."""

    def __init__(self, dino_dim: int, geometry_dim: int, hidden_dim: int = 192, heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.detail = PairCrossAttention(dino_dim, hidden_dim, heads, dropout)
        self.context = PairCrossAttention(dino_dim, hidden_dim, heads, dropout)
        pair_dim = hidden_dim * 4 * 2 + geometry_dim * 3 + 1
        self.trunk = nn.Sequential(nn.Linear(pair_dim, 384), nn.GELU(), nn.Dropout(dropout), nn.Linear(384, 128), nn.GELU())
        self.embedding_head = nn.Linear(128, 1)
        self.ssim_head = nn.Linear(128, 1)
        self.iou_head = nn.Linear(128, 1)

    def forward(self, query_detail: torch.Tensor, cache_detail: torch.Tensor, query_context: torch.Tensor, cache_context: torch.Tensor, query_geometry: torch.Tensor, cache_geometry: torch.Tensor, hand_scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, candidates = cache_detail.shape[:2]
        detail = self.detail(query_detail, cache_detail)
        context = self.context(query_context, cache_context)
        query_geometry = query_geometry[:, None].expand(-1, candidates, -1)
        geometry = torch.cat((query_geometry, cache_geometry, torch.abs(query_geometry - cache_geometry)), dim=-1)
        hidden = self.trunk(torch.cat((detail, context, geometry, hand_scores[..., None]), dim=-1))
        return self.embedding_head(hidden).squeeze(-1), self.ssim_head(hidden).squeeze(-1), self.iou_head(hidden).squeeze(-1)


@dataclass(frozen=True)
class ScoreWeights:
    embedding: float
    ssim: float
    iou: float


SCORE_WEIGHT_OPTIONS = {
    "embedding_only": ScoreWeights(1.0, 0.0, 0.0),
    "balanced_light": ScoreWeights(0.75, 0.125, 0.125),
    "balanced_equal": ScoreWeights(0.50, 0.25, 0.25),
}


def composite_score(embedding: torch.Tensor, ssim: torch.Tensor, iou: torch.Tensor, weights: ScoreWeights) -> torch.Tensor:
    """Lower is better; SSIM/IoU heads are quality predictions and are negated."""
    return weights.embedding * embedding - weights.ssim * ssim - weights.iou * iou
