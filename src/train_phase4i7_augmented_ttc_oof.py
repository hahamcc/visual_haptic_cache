"""Train control and augmented strict-OOF temporal TTC models."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .config import load_config, project_path
from .phase4h_dino_adaptation import assert_development_only, record_hash_split
from .train_phase4b_predicted_box_cache_ranker import set_seed
from .train_phase4i5_temporal_progress_oof import (
    PROGRESS_NAMES,
    class_weights,
    load_temporal_cache,
    masked_mean_std,
)
from .temporal_progress import MaskedTrajectoryEncoder
from .utils import ensure_dir, read_csv_rows, write_csv_rows, write_json


OUTPUT_FIELDS = [
    "query_record_id",
    "query_image_name",
    "query_probe",
    "oof_fold",
    "phase4i7_source",
    "control_predicted_ttc",
    "augmented_predicted_ttc",
    "control_predicted_progress",
    "augmented_predicted_progress",
    "control_far_probability",
    "augmented_far_probability",
    "true_progress",
    "control_absolute_ttc_error",
    "augmented_absolute_ttc_error",
]


class TemporalMotionProgressModel(nn.Module):
    """Four-frame visual progress model fused with exact 32-frame motion."""

    def __init__(
        self,
        visual_dim: int,
        geometry_dim: int,
        motion_dim: int,
        projection_dim: int,
        visual_hidden_dim: int,
        motion_hidden_dim: int,
        ttc_classes: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.frame_projection = nn.Sequential(
            nn.Linear(visual_dim * 2 + geometry_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.visual_cell = nn.GRUCell(projection_dim, visual_hidden_dim)
        self.motion_encoder = MaskedTrajectoryEncoder(
            motion_dim,
            motion_hidden_dim,
        )
        fusion_dim = visual_hidden_dim + motion_hidden_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, visual_hidden_dim),
            nn.LayerNorm(visual_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.progress_head = nn.Linear(visual_hidden_dim, len(PROGRESS_NAMES))
        self.ordinal_head = nn.Linear(visual_hidden_dim, 2)
        self.ttc_head = nn.Linear(visual_hidden_dim, ttc_classes)

    def forward(
        self,
        visual: torch.Tensor,
        geometry: torch.Tensor,
        visual_valid: torch.Tensor,
        motion: torch.Tensor,
        motion_valid: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        delta = torch.zeros_like(visual)
        delta[:, 1:] = visual[:, 1:] - visual[:, :-1]
        frames = self.frame_projection(
            torch.cat((visual, delta, geometry), dim=-1)
        )
        visual_hidden = torch.zeros(
            visual.shape[0],
            self.visual_cell.hidden_size,
            device=visual.device,
            dtype=visual.dtype,
        )
        for index in range(frames.shape[1]):
            updated = self.visual_cell(frames[:, index], visual_hidden)
            mask = visual_valid[:, index : index + 1].to(visual.dtype)
            visual_hidden = mask * updated + (1.0 - mask) * visual_hidden
        motion_hidden = self.motion_encoder(motion, motion_valid)
        fused = self.fusion(torch.cat((visual_hidden, motion_hidden), dim=1))
        return {
            "progress_logits": self.progress_head(fused),
            "ordinal_logits": self.ordinal_head(fused),
            "ttc_logits": self.ttc_head(fused),
        }


def load_training_rows(cfg: dict) -> tuple[list[dict[str, str]], int]:
    original_all = read_csv_rows(project_path(cfg["original_samples_csv"]))
    assert_development_only(
        original_all,
        project_path(cfg["final_partition_csv"]),
    )
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
        raise RuntimeError("Phase4I.7 original fold manifest size mismatch")
    for row in original:
        try:
            row["oof_fold"] = fold_by_name[row["image_name"]]
        except KeyError as exc:
            raise RuntimeError(
                f"Phase4I.7 original fold missing {row['image_name']}"
            ) from exc
        row["phase4i7_source"] = "original"

    increment = [
        dict(row)
        for row in read_csv_rows(project_path(cfg["increment_samples_csv"]))
    ]
    for row in increment:
        if row.get("oof_fold", "") not in {"0", "1", "2"}:
            raise RuntimeError("Phase4I.7 increment is missing an OOF fold")
        row["phase4i7_source"] = "increment"
    original_records = {(row["split"], row["record_id"]) for row in original}
    increment_records = {(row["split"], row["record_id"]) for row in increment}
    if original_records & increment_records:
        raise RuntimeError("Phase4I.7 original and increment records overlap")
    return original + increment, len(original)


def targets_for(
    rows: list[dict[str, str]],
    ttc_values: list[int],
    near_maximum: int,
    mid_maximum: int,
) -> tuple[np.ndarray, np.ndarray]:
    lookup = {value: index for index, value in enumerate(ttc_values)}
    ttc_target = []
    progress_target = []
    for row in rows:
        probe = int(row["probe"])
        if probe not in lookup:
            raise RuntimeError(f"Phase4I.7 unknown probe {probe}")
        ttc_target.append(lookup[probe])
        progress_target.append(
            0 if probe <= near_maximum else 1 if probe <= mid_maximum else 2
        )
    return (
        np.asarray(progress_target, dtype=np.int64),
        np.asarray(ttc_target, dtype=np.int64),
    )


def record_weights(indices: np.ndarray, records: np.ndarray) -> np.ndarray:
    counts: dict[str, int] = defaultdict(int)
    for index in indices:
        counts[str(records[index])] += 1
    values = np.asarray(
        [1.0 / counts[str(records[index])] for index in indices],
        dtype=np.float32,
    )
    return values / max(float(values.mean()), 1e-8)


def normalization_statistics(
    visual: np.ndarray,
    geometry: np.ndarray,
    valid: np.ndarray,
    motion: np.ndarray,
    motion_valid: np.ndarray,
    fit: np.ndarray,
) -> dict[str, np.ndarray]:
    visual_mean, visual_std = masked_mean_std(visual, valid, fit)
    geometry_mean, geometry_std = masked_mean_std(geometry, valid, fit)
    motion_mean, motion_std = masked_mean_std(
        motion,
        motion_valid,
        fit,
    )
    return {
        "visual_mean": visual_mean,
        "visual_std": visual_std,
        "geometry_mean": geometry_mean,
        "geometry_std": geometry_std,
        "motion_mean": motion_mean,
        "motion_std": motion_std,
    }


def normalize_inputs(
    visual: np.ndarray,
    geometry: np.ndarray,
    motion: np.ndarray,
    statistics: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        (
            (visual - statistics["visual_mean"])
            / statistics["visual_std"]
        ).astype(np.float32),
        (
            (geometry - statistics["geometry_mean"])
            / statistics["geometry_std"]
        ).astype(np.float32),
        (
            (motion - statistics["motion_mean"])
            / statistics["motion_std"]
        ).astype(np.float32),
    )


def primary_loss(
    outputs: dict[str, torch.Tensor],
    progress: torch.Tensor,
    ttc: torch.Tensor,
    sample_weights: torch.Tensor,
    progress_weights: torch.Tensor,
    ttc_weights: torch.Tensor,
    cfg: dict,
) -> torch.Tensor:
    progress_loss = F.cross_entropy(
        outputs["progress_logits"],
        progress,
        weight=progress_weights,
        reduction="none",
    )
    ttc_loss = F.cross_entropy(
        outputs["ttc_logits"],
        ttc,
        weight=ttc_weights,
        reduction="none",
    )
    ordinal_target = torch.stack(
        ((progress >= 1).float(), (progress >= 2).float()),
        dim=1,
    )
    ordinal_loss = F.binary_cross_entropy_with_logits(
        outputs["ordinal_logits"],
        ordinal_target,
        reduction="none",
    ).mean(dim=1)
    per_query = (
        progress_loss
        + float(cfg["ttc_loss_weight"]) * ttc_loss
        + float(cfg["ordinal_loss_weight"]) * ordinal_loss
    )
    return (per_query * sample_weights).sum() / sample_weights.sum().clamp_min(
        1e-8
    )


def _record_probe_index(
    rows: list[dict[str, str]],
    indices: np.ndarray,
) -> dict[tuple[str, int], int]:
    output = {}
    for index in indices:
        row = rows[int(index)]
        key = (f"{row['split']}:{row['record_id']}", int(row["probe"]))
        if key in output:
            raise RuntimeError(f"Phase4I.7 duplicate record/probe query {key}")
        output[key] = int(index)
    return output


def auxiliary_pairs(
    rows: list[dict[str, str]],
    indices: np.ndarray,
    approved_pairs: list[dict[str, str]],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    lookup = _record_probe_index(rows, indices)
    monotonic = []
    records = sorted({key[0] for key in lookup})
    for record in records:
        if (record, 75) in lookup and (record, 100) in lookup:
            monotonic.append((lookup[(record, 75)], lookup[(record, 100)]))
    consistency = []
    record_split = {
        row["record_id"]: row["split"]
        for row in rows
        if row["phase4i7_source"] == "increment"
    }
    for pair in approved_pairs:
        if pair.get("approved", "0") != "1":
            continue
        left_record = pair["record_a"]
        right_record = pair["record_b"]
        if left_record not in record_split or right_record not in record_split:
            raise RuntimeError("Phase4I.7 approved pair record is missing")
        left_key = f"{record_split[left_record]}:{left_record}"
        right_key = f"{record_split[right_record]}:{right_record}"
        for probe in (75, 100):
            left = lookup.get((left_key, probe))
            right = lookup.get((right_key, probe))
            if left is not None and right is not None:
                consistency.append((left, right))
    return monotonic, consistency


def _model_output(
    model: TemporalMotionProgressModel,
    visual: np.ndarray,
    geometry: np.ndarray,
    valid: np.ndarray,
    motion: np.ndarray,
    motion_valid: np.ndarray,
    indices: np.ndarray,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return model(
        torch.from_numpy(visual[indices]).to(device),
        torch.from_numpy(geometry[indices]).to(device),
        torch.from_numpy(valid[indices]).to(device),
        torch.from_numpy(motion[indices]).to(device),
        torch.from_numpy(motion_valid[indices]).to(device),
    )


def auxiliary_epoch(
    model: TemporalMotionProgressModel,
    optimizer: torch.optim.Optimizer,
    visual: np.ndarray,
    geometry: np.ndarray,
    valid: np.ndarray,
    motion: np.ndarray,
    motion_valid: np.ndarray,
    monotonic_pairs: list[tuple[int, int]],
    consistency_pairs: list[tuple[int, int]],
    ttc_axis: torch.Tensor,
    cfg: dict,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    losses: dict[str, list[float]] = {
        "monotonic": [],
        "consistency": [],
    }
    pair_batch = int(cfg["pair_batch_size"])
    recipes = (
        (
            "monotonic",
            monotonic_pairs,
            float(cfg["monotonic_loss_weight"]),
        ),
        (
            "consistency",
            consistency_pairs,
            float(cfg["pair_consistency_loss_weight"]),
        ),
    )
    for kind, pairs, weight in recipes:
        if not pairs or weight <= 0:
            continue
        order = np.random.permutation(len(pairs))
        for start in range(0, len(order), pair_batch):
            selected = [pairs[int(index)] for index in order[start : start + pair_batch]]
            left = np.asarray([pair[0] for pair in selected], dtype=np.int32)
            right = np.asarray([pair[1] for pair in selected], dtype=np.int32)
            left_output = _model_output(
                model,
                visual,
                geometry,
                valid,
                motion,
                motion_valid,
                left,
                device,
            )
            right_output = _model_output(
                model,
                visual,
                geometry,
                valid,
                motion,
                motion_valid,
                right,
                device,
            )
            left_probability = torch.softmax(
                left_output["ttc_logits"], dim=1
            )
            right_probability = torch.softmax(
                right_output["ttc_logits"], dim=1
            )
            left_expected = left_probability @ ttc_axis
            right_expected = right_probability @ ttc_axis
            if kind == "monotonic":
                loss = (
                    F.relu(
                        float(cfg["monotonic_minimum_gap_frames"])
                        - (right_expected - left_expected)
                    ).mean()
                    / float(cfg["ttc_values"][-1])
                )
            else:
                probability_loss = F.mse_loss(
                    left_probability,
                    right_probability,
                )
                expected_loss = (
                    (left_expected - right_expected).abs().mean()
                    / float(cfg["ttc_values"][-1])
                )
                loss = 0.5 * probability_loss + 0.5 * expected_loss
            weighted = weight * loss
            optimizer.zero_grad(set_to_none=True)
            weighted.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(),
                float(cfg["gradient_clip"]),
            )
            optimizer.step()
            losses[kind].append(float(loss.detach().cpu()))
    return {
        f"{kind}_loss": float(np.mean(values)) if values else 0.0
        for kind, values in losses.items()
    }


def _checkpoint_compatible(path: Path, metadata: dict) -> bool:
    if not path.is_file():
        return False
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        return checkpoint.get("metadata") == metadata
    except (EOFError, OSError, RuntimeError, TypeError, ValueError):
        return False


def train_model(
    visual: np.ndarray,
    geometry: np.ndarray,
    valid: np.ndarray,
    motion: np.ndarray,
    motion_valid: np.ndarray,
    progress: np.ndarray,
    ttc: np.ndarray,
    records: np.ndarray,
    rows: list[dict[str, str]],
    fit: np.ndarray,
    early_stop: np.ndarray,
    approved_pairs: list[dict[str, str]],
    augmented: bool,
    cfg: dict,
    device: torch.device,
    checkpoint_path: Path,
    metadata: dict,
) -> tuple[TemporalMotionProgressModel, dict]:
    statistics = normalization_statistics(
        visual,
        geometry,
        valid,
        motion,
        motion_valid,
        fit,
    )
    normalized_visual, normalized_geometry, normalized_motion = normalize_inputs(
        visual,
        geometry,
        motion,
        statistics,
    )
    metadata = {
        **metadata,
        "visual_mean_sha256": _array_sha256(statistics["visual_mean"]),
        "geometry_mean_sha256": _array_sha256(statistics["geometry_mean"]),
        "motion_mean_sha256": _array_sha256(statistics["motion_mean"]),
    }
    model = TemporalMotionProgressModel(
        visual.shape[2],
        geometry.shape[2],
        motion.shape[2],
        int(cfg["projection_dim"]),
        int(cfg["hidden_dim"]),
        int(cfg["motion_hidden_dim"]),
        len(cfg["ttc_values"]),
        float(cfg["dropout"]),
    ).to(device)
    if _checkpoint_compatible(checkpoint_path, metadata):
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(checkpoint["model_state"])
        return model, {
            **checkpoint["training_report"],
            "reused_checkpoint": True,
        }

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    fit_sample_weights = record_weights(fit, records)
    early_weights = record_weights(early_stop, records)
    progress_weight = torch.from_numpy(
        class_weights(
            progress,
            fit,
            fit_sample_weights,
            len(PROGRESS_NAMES),
        )
    ).to(device)
    ttc_weight = torch.from_numpy(
        class_weights(
            ttc,
            fit,
            fit_sample_weights,
            len(cfg["ttc_values"]),
        )
    ).to(device)
    monotonic_pairs, consistency_pairs = (
        auxiliary_pairs(rows, fit, approved_pairs)
        if augmented
        else ([], [])
    )
    ttc_axis = torch.tensor(
        cfg["ttc_values"],
        dtype=torch.float32,
        device=device,
    )
    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    history = []
    best_state = None
    for epoch in range(1, int(cfg["epochs"]) + 1):
        model.train()
        order = np.random.permutation(len(fit))
        train_losses = []
        for start in range(0, len(order), int(cfg["train_batch_size"])):
            local = order[start : start + int(cfg["train_batch_size"])]
            batch = fit[local]
            output = _model_output(
                model,
                normalized_visual,
                normalized_geometry,
                valid,
                normalized_motion,
                motion_valid,
                batch,
                device,
            )
            loss = primary_loss(
                output,
                torch.from_numpy(progress[batch]).to(device),
                torch.from_numpy(ttc[batch]).to(device),
                torch.from_numpy(fit_sample_weights[local]).to(device),
                progress_weight,
                ttc_weight,
                cfg,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(),
                float(cfg["gradient_clip"]),
            )
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        auxiliary = auxiliary_epoch(
            model,
            optimizer,
            normalized_visual,
            normalized_geometry,
            valid,
            normalized_motion,
            motion_valid,
            monotonic_pairs,
            consistency_pairs,
            ttc_axis,
            cfg,
            device,
        )
        model.eval()
        with torch.no_grad():
            early_output = _model_output(
                model,
                normalized_visual,
                normalized_geometry,
                valid,
                normalized_motion,
                motion_valid,
                early_stop,
                device,
            )
            validation_loss = float(
                primary_loss(
                    early_output,
                    torch.from_numpy(progress[early_stop]).to(device),
                    torch.from_numpy(ttc[early_stop]).to(device),
                    torch.from_numpy(early_weights).to(device),
                    progress_weight,
                    ttc_weight,
                    cfg,
                ).cpu()
            )
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(train_losses)),
                "validation_loss": validation_loss,
                **auxiliary,
            }
        )
        if validation_loss < best_loss - 1e-6:
            best_loss = validation_loss
            best_epoch = epoch
            stale = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            stale += 1
        if stale >= int(cfg["early_stopping_patience"]):
            break
    if best_state is None:
        raise RuntimeError("Phase4I.7 did not produce a checkpoint")
    model.load_state_dict(best_state)
    training_report = {
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "epochs_ran": len(history),
        "monotonic_pairs": len(monotonic_pairs),
        "consistency_pairs": len(consistency_pairs),
        "history": history,
        "reused_checkpoint": False,
    }
    ensure_dir(checkpoint_path.parent)
    torch.save(
        {
            "model_state": best_state,
            "statistics": statistics,
            "metadata": metadata,
            "training_report": training_report,
        },
        checkpoint_path,
    )
    return model, training_report


def _array_sha256(array: np.ndarray) -> str:
    import hashlib

    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def probabilities_for(
    model: TemporalMotionProgressModel,
    visual: np.ndarray,
    geometry: np.ndarray,
    valid: np.ndarray,
    motion: np.ndarray,
    motion_valid: np.ndarray,
    indices: np.ndarray,
    checkpoint_path: Path,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    normalized_visual, normalized_geometry, normalized_motion = normalize_inputs(
        visual,
        geometry,
        motion,
        checkpoint["statistics"],
    )
    progress_values = []
    ttc_values = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch = indices[start : start + batch_size]
            output = _model_output(
                model,
                normalized_visual,
                normalized_geometry,
                valid,
                normalized_motion,
                motion_valid,
                batch,
                device,
            )
            progress_values.append(
                torch.softmax(output["progress_logits"], dim=1)
                .cpu()
                .numpy()
            )
            ttc_values.append(
                torch.softmax(output["ttc_logits"], dim=1).cpu().numpy()
            )
    return (
        np.concatenate(progress_values).astype(np.float32),
        np.concatenate(ttc_values).astype(np.float32),
    )


def nested_split(
    rows: list[dict[str, str]],
    fold: str,
    include_increment: bool,
    cfg: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eligible = np.asarray(
        [
            index
            for index, row in enumerate(rows)
            if include_increment or row["phase4i7_source"] == "original"
        ],
        dtype=np.int32,
    )
    held_out = np.asarray(
        [index for index, row in enumerate(rows) if row["oof_fold"] == fold],
        dtype=np.int32,
    )
    outer_fit = np.asarray(
        [
            index
            for index in eligible
            if rows[int(index)]["oof_fold"] != fold
        ],
        dtype=np.int32,
    )
    group_key = {}
    for index in outer_fit:
        row = rows[int(index)]
        key = f"{row['split']}:{row['record_id']}"
        if include_increment and row.get("approved_pair_id", ""):
            key = f"approved:{row['approved_pair_id']}"
        group_key[int(index)] = key
    early_stop = np.asarray(
        [
            index
            for index in outer_fit
            if record_hash_split(
                group_key[int(index)],
                float(cfg["early_stop_fraction"]),
                int(cfg["early_stop_seed"]) + int(fold),
            )
        ],
        dtype=np.int32,
    )
    early_set = set(early_stop.tolist())
    fit = np.asarray(
        [index for index in outer_fit if int(index) not in early_set],
        dtype=np.int32,
    )
    if not len(fit) or not len(early_stop) or not len(held_out):
        raise RuntimeError(f"Phase4I.7 fold {fold} has an empty split")
    fit_records = {
        f"{rows[int(index)]['split']}:{rows[int(index)]['record_id']}"
        for index in fit
    }
    early_records = {
        f"{rows[int(index)]['split']}:{rows[int(index)]['record_id']}"
        for index in early_stop
    }
    held_records = {
        f"{rows[int(index)]['split']}:{rows[int(index)]['record_id']}"
        for index in held_out
    }
    if fit_records & early_records or (fit_records | early_records) & held_records:
        raise RuntimeError(f"Phase4I.7 fold {fold} record leakage")
    return fit, early_stop, held_out


def metric_summary(
    progress_probability: np.ndarray,
    ttc_probability: np.ndarray,
    progress_target: np.ndarray,
    ttc_target: np.ndarray,
    ttc_axis: np.ndarray,
) -> dict:
    predicted_progress = progress_probability.argmax(axis=1)
    predicted_ttc = ttc_probability @ ttc_axis
    true_ttc = ttc_axis[ttc_target]
    far = progress_target == 2
    nonfar = ~far
    tp = int(((predicted_progress == 2) & far).sum())
    fp = int(((predicted_progress == 2) & nonfar).sum())
    return {
        "queries": int(len(progress_target)),
        "ttc_mae_frames": float(np.abs(predicted_ttc - true_ttc).mean()),
        "ttc_bias_frames": float((predicted_ttc - true_ttc).mean()),
        "progress_accuracy": float(
            (predicted_progress == progress_target).mean()
        ),
        "far_recall": tp / max(int(far.sum()), 1),
        "far_precision": tp / max(tp + fp, 1),
        "near_mid_retention": (
            float((predicted_progress[nonfar] != 2).mean())
            if nonfar.any()
            else None
        ),
        "adjacent_ttc_bucket_accuracy": float(
            (np.abs(ttc_probability.argmax(axis=1) - ttc_target) <= 1).mean()
        ),
    }


def bootstrap_delta(
    rows: list[dict[str, str]],
    indices: np.ndarray,
    control_progress: np.ndarray,
    augmented_progress: np.ndarray,
    control_ttc: np.ndarray,
    augmented_ttc: np.ndarray,
    progress_target: np.ndarray,
    ttc_target: np.ndarray,
    ttc_axis: np.ndarray,
    iterations: int,
    seed: int,
) -> dict:
    records = np.asarray(
        [
            f"{rows[int(index)]['split']}:{rows[int(index)]['record_id']}"
            for index in indices
        ]
    )
    unique = np.unique(records)
    control_expected = control_ttc[indices] @ ttc_axis
    augmented_expected = augmented_ttc[indices] @ ttc_axis
    truth = ttc_axis[ttc_target[indices]]
    mae_delta = (
        np.abs(augmented_expected - truth)
        - np.abs(control_expected - truth)
    )
    far = progress_target[indices] == 2
    control_far_correct = (
        control_progress[indices].argmax(axis=1) == 2
    ).astype(np.float32)
    augmented_far_correct = (
        augmented_progress[indices].argmax(axis=1) == 2
    ).astype(np.float32)
    recall_delta = augmented_far_correct - control_far_correct
    by_record = {
        record: np.flatnonzero(records == record)
        for record in unique
    }
    rng = np.random.default_rng(seed)
    mae_draws = []
    recall_draws = []
    for _ in range(iterations):
        draw = rng.choice(unique, size=len(unique), replace=True)
        selected = np.concatenate([by_record[record] for record in draw])
        mae_draws.append(float(mae_delta[selected].mean()))
        far_selected = selected[far[selected]]
        if len(far_selected):
            recall_draws.append(float(recall_delta[far_selected].mean()))
    return {
        "ttc_mae_delta_augmented_minus_control": float(mae_delta.mean()),
        "ttc_mae_delta_95_ci": [
            float(value)
            for value in np.quantile(mae_draws, [0.025, 0.975])
        ],
        "far_recall_delta_augmented_minus_control": (
            float(recall_delta[far].mean()) if far.any() else None
        ),
        "far_recall_delta_95_ci": (
            [
                float(value)
                for value in np.quantile(recall_draws, [0.025, 0.975])
            ]
            if recall_draws
            else None
        ),
    }


def pair_diagnostics(
    rows: list[dict[str, str]],
    indices: np.ndarray,
    control_ttc: np.ndarray,
    augmented_ttc: np.ndarray,
    ttc_axis: np.ndarray,
    approved_pairs: list[dict[str, str]],
) -> dict:
    monotonic, consistency = auxiliary_pairs(rows, indices, approved_pairs)
    control_expected = control_ttc @ ttc_axis
    augmented_expected = augmented_ttc @ ttc_axis

    def monotonic_report(values: np.ndarray) -> dict:
        gaps = np.asarray(
            [values[right] - values[left] for left, right in monotonic],
            dtype=np.float32,
        )
        return {
            "pairs": len(gaps),
            "mean_100_minus_75_frames": float(gaps.mean()) if len(gaps) else None,
            "ordered_rate": float((gaps > 0).mean()) if len(gaps) else None,
        }

    def consistency_report(values: np.ndarray) -> dict:
        gaps = np.asarray(
            [abs(values[right] - values[left]) for left, right in consistency],
            dtype=np.float32,
        )
        return {
            "pairs": len(gaps),
            "mean_same_probe_gap_frames": float(gaps.mean()) if len(gaps) else None,
        }

    return {
        "control": {
            "monotonic": monotonic_report(control_expected),
            "approved_pair_consistency": consistency_report(control_expected),
        },
        "augmented": {
            "monotonic": monotonic_report(augmented_expected),
            "approved_pair_consistency": consistency_report(augmented_expected),
        },
    }


def load_motion_cache(
    prefix: Path,
    query_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    cache_names = np.load(
        Path(f"{prefix}.names.npy"),
        allow_pickle=False,
    ).astype(str)
    lookup = {name: index for index, name in enumerate(cache_names)}
    if len(lookup) != len(cache_names):
        raise RuntimeError("Phase4I.7 motion cache has duplicate names")
    if any(name not in lookup for name in query_names):
        raise RuntimeError("Phase4I.7 motion cache misses queries")
    indices = np.asarray([lookup[name] for name in query_names], dtype=np.int32)
    motion_path = Path(f"{prefix}.motion.npy")
    valid_path = Path(f"{prefix}.motion_valid.npy")
    if not motion_path.is_file() or not valid_path.is_file():
        raise FileNotFoundError("Phase4I.7 motion cache is incomplete")
    motion = np.asarray(np.load(motion_path, mmap_mode="r")[indices])
    valid = np.asarray(np.load(valid_path, mmap_mode="r")[indices])
    if motion.ndim != 3 or valid.shape != motion.shape[:2]:
        raise RuntimeError("Phase4I.7 motion cache shapes are inconsistent")
    return motion.astype(np.float32), valid.astype(np.float32)


def run(config_path: str, section: str) -> dict:
    cfg = load_config(config_path)[section]
    set_seed(int(cfg["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows, original_count = load_training_rows(cfg)
    names = [row["image_name"] for row in rows]
    if len(names) != len(set(names)):
        raise RuntimeError("Phase4I.7 query names are not unique")
    feature_prefix = project_path(cfg["feature_cache_prefix"])
    visual, geometry, valid, padding = load_temporal_cache(
        feature_prefix,
        names,
    )
    motion, motion_valid = load_motion_cache(feature_prefix, names)
    if np.any(valid.sum(axis=1) < int(cfg["minimum_valid_frames"])):
        raise RuntimeError("Phase4I.7 feature cache has incomplete queries")
    ttc_axis = np.asarray(cfg["ttc_values"], dtype=np.float32)
    progress, ttc_target = targets_for(
        rows,
        [int(value) for value in cfg["ttc_values"]],
        int(cfg["near_probe_maximum"]),
        int(cfg["mid_probe_maximum"]),
    )
    records = np.asarray(
        [f"{row['split']}:{row['record_id']}" for row in rows]
    )
    approved_pairs = read_csv_rows(
        project_path(cfg["approved_pair_csv"])
    )
    feature_metadata_path = Path(
        f"{project_path(cfg['feature_cache_prefix'])}.json"
    )
    with feature_metadata_path.open("r", encoding="utf-8") as handle:
        feature_metadata = json.load(handle)
    folds = sorted({row["oof_fold"] for row in rows})
    if folds != ["0", "1", "2"]:
        raise RuntimeError(f"Phase4I.7 requires folds 0/1/2, got {folds}")

    shape_progress = (len(rows), len(PROGRESS_NAMES))
    shape_ttc = (len(rows), len(ttc_axis))
    probabilities = {
        arm: {
            "progress": np.zeros(shape_progress, dtype=np.float32),
            "ttc": np.zeros(shape_ttc, dtype=np.float32),
        }
        for arm in ("control", "augmented")
    }
    training_reports = []
    checkpoint_root = project_path(cfg["checkpoint_dir"])
    ensure_dir(checkpoint_root)
    for fold in folds:
        for arm in ("control", "augmented"):
            augmented = arm == "augmented"
            fit, early_stop, held_out = nested_split(
                rows,
                fold,
                augmented,
                cfg,
            )
            if len(np.unique(ttc_target[fit])) != len(ttc_axis):
                raise RuntimeError(
                    f"Phase4I.7 {arm} fold {fold} misses TTC classes"
                )
            held_progress = []
            held_ttc = []
            for seed in [int(value) for value in cfg["seeds"]]:
                print(
                    f"phase4i7 {arm} fold {fold} seed {seed}: training",
                    flush=True,
                )
                set_seed(seed)
                checkpoint_path = (
                    checkpoint_root / f"{arm}_fold_{fold}_seed_{seed}.pt"
                )
                model, training_report = train_model(
                    visual,
                    geometry,
                    valid,
                    motion,
                    motion_valid,
                    progress,
                    ttc_target,
                    records,
                    rows,
                    fit,
                    early_stop,
                    approved_pairs,
                    augmented,
                    cfg,
                    device,
                    checkpoint_path,
                    {
                        "scope": "phase4i7_strict_oof_augmented_ttc",
                        "arm": arm,
                        "fold": fold,
                        "seed": seed,
                        "feature_identity": feature_metadata[
                            "query_identity_sha256"
                        ],
                        "increment_fingerprint": feature_metadata[
                            "increment_fingerprint"
                        ],
                        "training_recipe": {
                            key: cfg[key]
                            for key in (
                                "projection_dim",
                                "hidden_dim",
                                "motion_hidden_dim",
                                "dropout",
                                "ttc_values",
                                "ttc_loss_weight",
                                "ordinal_loss_weight",
                                "monotonic_loss_weight",
                                "monotonic_minimum_gap_frames",
                                "pair_consistency_loss_weight",
                                "early_stop_fraction",
                                "early_stop_seed",
                                "learning_rate",
                                "weight_decay",
                            )
                        },
                        "query_true_probe_feature_used": False,
                        "query_tactile_input": False,
                    },
                )
                progress_probability, ttc_probability = probabilities_for(
                    model,
                    visual,
                    geometry,
                    valid,
                    motion,
                    motion_valid,
                    held_out,
                    checkpoint_path,
                    int(cfg["inference_batch_size"]),
                    device,
                )
                held_progress.append(progress_probability)
                held_ttc.append(ttc_probability)
                training_reports.append(
                    {
                        "arm": arm,
                        "fold": fold,
                        "seed": seed,
                        "fit_queries": len(fit),
                        "early_stop_queries": len(early_stop),
                        "held_out_queries": len(held_out),
                        **training_report,
                    }
                )
            probabilities[arm]["progress"][held_out] = np.mean(
                held_progress,
                axis=0,
            )
            probabilities[arm]["ttc"][held_out] = np.mean(
                held_ttc,
                axis=0,
            )

    original_indices = np.arange(original_count, dtype=np.int32)
    increment_indices = np.arange(
        original_count,
        len(rows),
        dtype=np.int32,
    )
    original_far = original_indices[progress[original_indices] == 2]
    slices = {
        "original_all": original_indices,
        "original_far": original_far,
        "increment_far": increment_indices,
    }
    summary = {}
    comparisons = {}
    for name, indices in slices.items():
        summary[name] = {
            arm: metric_summary(
                probabilities[arm]["progress"][indices],
                probabilities[arm]["ttc"][indices],
                progress[indices],
                ttc_target[indices],
                ttc_axis,
            )
            for arm in ("control", "augmented")
        }
        comparisons[name] = bootstrap_delta(
            rows,
            indices,
            probabilities["control"]["progress"],
            probabilities["augmented"]["progress"],
            probabilities["control"]["ttc"],
            probabilities["augmented"]["ttc"],
            progress,
            ttc_target,
            ttc_axis,
            int(cfg["bootstrap_iterations"]),
            int(cfg["bootstrap_seed"]),
        )

    increment_comparison = comparisons["increment_far"]
    original_comparison = comparisons["original_all"]
    original_far_comparison = comparisons["original_far"]
    point_pass = bool(
        increment_comparison["ttc_mae_delta_augmented_minus_control"] < 0
        and increment_comparison[
            "far_recall_delta_augmented_minus_control"
        ]
        >= 0
        and original_comparison[
            "ttc_mae_delta_augmented_minus_control"
        ]
        <= float(cfg["maximum_original_mae_regression_frames"])
        and original_far_comparison[
            "far_recall_delta_augmented_minus_control"
        ]
        >= -float(cfg["maximum_original_far_recall_regression"])
    )
    ci_pass = bool(
        increment_comparison["ttc_mae_delta_95_ci"][1] < 0
        and original_comparison["ttc_mae_delta_95_ci"][1]
        <= float(cfg["maximum_original_mae_regression_frames"])
    )
    accepted = bool(point_pass and ci_pass)

    control_predicted_ttc = probabilities["control"]["ttc"] @ ttc_axis
    augmented_predicted_ttc = probabilities["augmented"]["ttc"] @ ttc_axis
    output = []
    for index, row in enumerate(rows):
        control_progress = int(
            probabilities["control"]["progress"][index].argmax()
        )
        augmented_progress = int(
            probabilities["augmented"]["progress"][index].argmax()
        )
        truth = float(ttc_axis[ttc_target[index]])
        output.append(
            {
                "query_record_id": row["record_id"],
                "query_image_name": row["image_name"],
                "query_probe": row["probe"],
                "oof_fold": row["oof_fold"],
                "phase4i7_source": row["phase4i7_source"],
                "control_predicted_ttc": f"{control_predicted_ttc[index]:.6f}",
                "augmented_predicted_ttc": (
                    f"{augmented_predicted_ttc[index]:.6f}"
                ),
                "control_predicted_progress": PROGRESS_NAMES[
                    control_progress
                ],
                "augmented_predicted_progress": PROGRESS_NAMES[
                    augmented_progress
                ],
                "control_far_probability": (
                    f"{probabilities['control']['progress'][index, 2]:.9f}"
                ),
                "augmented_far_probability": (
                    f"{probabilities['augmented']['progress'][index, 2]:.9f}"
                ),
                "true_progress": PROGRESS_NAMES[int(progress[index])],
                "control_absolute_ttc_error": (
                    f"{abs(control_predicted_ttc[index] - truth):.6f}"
                ),
                "augmented_absolute_ttc_error": (
                    f"{abs(augmented_predicted_ttc[index] - truth):.6f}"
                ),
            }
        )
    write_csv_rows(
        project_path(cfg["query_output_csv"]),
        output,
        OUTPUT_FIELDS,
    )
    pair_report = pair_diagnostics(
        rows,
        increment_indices,
        probabilities["control"]["ttc"],
        probabilities["augmented"]["ttc"],
        ttc_axis,
        approved_pairs,
    )
    report = {
        "mode": "phase4i7_augmented_ttc_oof_v1",
        "device": str(device),
        "temporal_input": {
            "visual_frame_offsets": list(cfg["frame_offsets"]),
            "visual_crop_definition": "per-frame current sensor tip",
            "visual_crop_size": int(cfg["crop_size"]),
            "motion_history_frames": int(cfg["trajectory_history_frames"]),
            "motion_valid_fraction": float(motion_valid.mean()),
        },
        "queries": {
            "original": len(original_indices),
            "increment": len(increment_indices),
        },
        "summary": summary,
        "comparison": comparisons,
        "pair_diagnostics": pair_report,
        "acceptance": {
            "point_pass": point_pass,
            "ci_pass": ci_pass,
            "accepted": accepted,
            "contract": {
                "increment_far_mae_delta": "strictly below zero",
                "increment_far_recall_delta": "nonnegative",
                "maximum_original_mae_regression_frames": float(
                    cfg["maximum_original_mae_regression_frames"]
                ),
                "maximum_original_far_recall_regression": float(
                    cfg["maximum_original_far_recall_regression"]
                ),
                "increment_far_mae_ci_upper": "strictly below zero",
            },
        },
        "training": training_reports,
        "integrity": {
            "source": "strict development 3-fold record-level OOF",
            "control_training": "original development records only",
            "augmented_training": (
                "original development plus frozen Phase4I.6F increment"
            ),
            "approved_pairs": (
                "offline auxiliary consistency supervision only"
            ),
            "query_true_probe_feature_used": False,
            "true_probe_offline_target_used": True,
            "query_target_tip_feature_used": False,
            "query_tactile_input": False,
            "future_visual_frames_used": False,
            "sealed_final_holdout_rows_read": 0,
            "development_validation_rows_read": 0,
        },
        "next_action": (
            "freeze augmented TTC and evaluate its OOF signal in the cache gate"
            if accepted
            else (
                "retain the prior TTC model; inspect increment and pair "
                "diagnostics before changing model capacity"
            )
        ),
    }
    write_json(project_path(cfg["metrics_json"]), report)
    print(
        {
            "mode": report["mode"],
            "queries": report["queries"],
            "summary": report["summary"],
            "comparison": report["comparison"],
            "pair_diagnostics": report["pair_diagnostics"],
            "acceptance": report["acceptance"],
            "integrity": report["integrity"],
            "next_action": report["next_action"],
        }
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Phase4I.7 control/augmented TTC strict OOF."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--section",
        default="phase4i7_augmented_ttc_oof_v1",
    )
    args = parser.parse_args()
    run(args.config, args.section)


if __name__ == "__main__":
    main()
