"""Strict-OOF temporal visual contact-progress model and safe V1 gate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .config import load_config, project_path
from .evaluate_phase4h_factorized_intensity_oof import (
    fast_bootstrap_comparison,
)
from .phase4h_dino_adaptation import (
    assert_development_only,
    record_hash_split,
)
from .train_phase4b_predicted_box_cache_ranker import set_seed
from .train_phase4h_dino_gate import metric_summary
from .train_phase4i4_far_risk_gate_oof import (
    THRESHOLD_FIELDS,
    assert_csv_schema,
    choose_recall_threshold,
    record_balanced_weights,
    retrieval_rows,
    select_by_risk,
    threshold_csv_row,
)
from .train_phase4i_factorized_residual_cascade import (
    METRICS,
    ONLINE_PROGRESS_FIELDS,
    required_float,
)
from .utils import ensure_dir, read_csv_rows, write_csv_rows, write_json


PROGRESS_NAMES = ("near", "mid", "far")
QUERY_FIELDS = [
    "query_record_id",
    "query_image_name",
    "query_probe",
    "oof_fold",
    "temporal_near_probability",
    "temporal_mid_probability",
    "temporal_far_class_probability",
    "temporal_far_ordinal_probability",
    "temporal_far_risk_probability",
    "far_risk_threshold",
    "predicted_far_risk",
    "true_far_label",
    "predicted_progress_class",
    "true_progress_class",
    "predicted_ttc",
    "true_ttc",
    "temporal_frame_valid_fraction",
    "temporal_padding_ratio",
    "phase4i3_selected_cache_image_name",
    "v1_selected_cache_image_name",
    "gated_selected_cache_image_name",
    "gated_selection_source",
    "gated_ranker_oracle_embedding_rank",
    *[f"gated_{metric}" for metric in METRICS],
    "deployment_accepted",
    "final_selection_source",
    "selected_cache_image_name",
    "ranker_oracle_embedding_rank",
    *METRICS,
]


def progress_targets(
    rows: list[dict[str, str]],
    near_maximum: int,
    mid_maximum: int,
    ttc_values: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    if near_maximum >= mid_maximum:
        raise ValueError("Phase4I.5 near/mid boundaries are not ordered")
    axis = [int(value) for value in ttc_values]
    if len(axis) != len(set(axis)):
        raise ValueError("Phase4I.5 TTC values must be unique")
    lookup = {value: index for index, value in enumerate(axis)}
    progress, ttc = [], []
    for row in rows:
        probe = int(row["query_probe"])
        if probe not in lookup:
            raise RuntimeError(f"Unknown Phase4I.5 probe value: {probe}")
        progress.append(
            0 if probe <= near_maximum else 1 if probe <= mid_maximum else 2
        )
        ttc.append(lookup[probe])
    return np.asarray(progress, dtype=np.int64), np.asarray(ttc, dtype=np.int64)


def online_progress_features(rows: list[dict[str, str]]) -> np.ndarray:
    values = np.asarray(
        [
            [required_float(row, field) for field in ONLINE_PROGRESS_FIELDS]
            for row in rows
        ],
        dtype=np.float32,
    )
    if not np.isfinite(values).all():
        raise RuntimeError("Phase4I.5 online features contain nonfinite values")
    return values


def load_temporal_cache(
    prefix: Path,
    query_names: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cache_names = np.load(
        Path(f"{prefix}.names.npy"),
        allow_pickle=False,
    ).astype(str)
    if len(set(cache_names.tolist())) != len(cache_names):
        raise RuntimeError("Phase4I.5 feature cache contains duplicate names")
    lookup = {name: index for index, name in enumerate(cache_names)}
    if any(name not in lookup for name in query_names):
        raise RuntimeError("Phase4I.5 feature cache misses OOF queries")
    indices = np.asarray([lookup[name] for name in query_names], dtype=np.int32)
    arrays = []
    for suffix in ("features", "geometry", "valid", "padding"):
        path = Path(f"{prefix}.{suffix}.npy")
        if not path.is_file():
            raise FileNotFoundError(path)
        arrays.append(np.asarray(np.load(path, mmap_mode="r")[indices]))
    visual, geometry, valid, padding = arrays
    if (
        visual.ndim != 3
        or geometry.shape[:2] != visual.shape[:2]
        or valid.shape != visual.shape[:2]
        or padding.shape != visual.shape[:2]
    ):
        raise RuntimeError("Phase4I.5 temporal cache shapes are inconsistent")
    if not np.isfinite(geometry).all() or not np.isfinite(padding).all():
        raise RuntimeError("Phase4I.5 temporal cache contains nonfinite values")
    return (
        visual.astype(np.float32),
        geometry.astype(np.float32),
        valid.astype(np.float32),
        padding.astype(np.float32),
    )


def validate_temporal_cache_metadata(prefix: Path, cfg: dict) -> dict:
    path = Path(f"{prefix}.json")
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    expected = {
        "frame_offsets": [int(value) for value in cfg["frame_offsets"]],
        "crop_size": int(cfg["crop_size"]),
        "dino_model": str(cfg["dino_model"]),
        "dino_image_size": int(cfg["dino_image_size"]),
        "dino_layer": int(cfg["dino_layer"]),
        "center_sigma": float(cfg["center_sigma"]),
        "query_true_probe_feature_used": False,
        "query_tactile_input": False,
        "future_visual_frames_used": False,
    }
    mismatch = {
        key: {"expected": value, "actual": metadata.get(key)}
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatch:
        raise RuntimeError(
            f"Phase4I.5 temporal feature metadata mismatch: {mismatch}"
        )
    return metadata


def masked_mean_std(
    values: np.ndarray,
    valid: np.ndarray,
    indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    selected = values[indices][valid[indices] > 0.5]
    if not len(selected):
        raise RuntimeError("Phase4I.5 normalization has no valid frames")
    mean = selected.mean(axis=0).astype(np.float32)
    std = selected.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


def class_weights(
    targets: np.ndarray,
    indices: np.ndarray,
    sample_weights: np.ndarray,
    classes: int,
) -> np.ndarray:
    totals = np.asarray(
        [
            sample_weights[targets[indices] == class_index].sum()
            for class_index in range(classes)
        ],
        dtype=np.float32,
    )
    if np.any(totals <= 0):
        raise RuntimeError("Phase4I.5 training split misses a target class")
    inverse = totals.sum() / (classes * totals)
    return (inverse / inverse.mean()).astype(np.float32)


class TemporalProgressModel(nn.Module):
    """Frozen-DINO sequence head with online scalar progress context."""

    def __init__(
        self,
        visual_dim: int,
        geometry_dim: int,
        online_dim: int,
        projection_dim: int,
        hidden_dim: int,
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
        self.temporal_cell = nn.GRUCell(projection_dim, hidden_dim)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + online_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.progress_head = nn.Linear(hidden_dim, len(PROGRESS_NAMES))
        self.ordinal_head = nn.Linear(hidden_dim, 2)
        self.ttc_head = nn.Linear(hidden_dim, ttc_classes)

    def forward(
        self,
        visual: torch.Tensor,
        geometry: torch.Tensor,
        valid_mask: torch.Tensor,
        online: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        delta = torch.zeros_like(visual)
        delta[:, 1:] = visual[:, 1:] - visual[:, :-1]
        frames = self.frame_projection(
            torch.cat((visual, delta, geometry), dim=-1)
        )
        hidden = torch.zeros(
            visual.shape[0],
            self.temporal_cell.hidden_size,
            device=visual.device,
            dtype=visual.dtype,
        )
        for index in range(frames.shape[1]):
            updated = self.temporal_cell(frames[:, index], hidden)
            mask = valid_mask[:, index : index + 1].to(visual.dtype)
            hidden = mask * updated + (1.0 - mask) * hidden
        fused = self.fusion(torch.cat((hidden, online), dim=1))
        return {
            "progress_logits": self.progress_head(fused),
            "ordinal_logits": self.ordinal_head(fused),
            "ttc_logits": self.ttc_head(fused),
        }


def model_loss(
    outputs: dict[str, torch.Tensor],
    progress: torch.Tensor,
    ttc: torch.Tensor,
    sample_weights: torch.Tensor,
    progress_weights: torch.Tensor,
    ttc_weights: torch.Tensor,
    ttc_loss_weight: float,
    ordinal_loss_weight: float,
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
        + ttc_loss_weight * ttc_loss
        + ordinal_loss_weight * ordinal_loss
    )
    return (per_query * sample_weights).sum() / sample_weights.sum().clamp_min(
        1e-8
    )


def normalize_inputs(
    visual: np.ndarray,
    geometry: np.ndarray,
    online: np.ndarray,
    visual_mean: np.ndarray,
    visual_std: np.ndarray,
    geometry_mean: np.ndarray,
    geometry_std: np.ndarray,
    online_mean: np.ndarray,
    online_std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        ((visual - visual_mean) / visual_std).astype(np.float32),
        ((geometry - geometry_mean) / geometry_std).astype(np.float32),
        ((online - online_mean) / online_std).astype(np.float32),
    )


def training_statistics(
    visual: np.ndarray,
    geometry: np.ndarray,
    valid: np.ndarray,
    online: np.ndarray,
    fit: np.ndarray,
) -> dict[str, np.ndarray]:
    visual_mean, visual_std = masked_mean_std(visual, valid, fit)
    geometry_mean, geometry_std = masked_mean_std(geometry, valid, fit)
    online_mean = online[fit].mean(axis=0).astype(np.float32)
    online_std = online[fit].std(axis=0).astype(np.float32)
    online_std[online_std < 1e-6] = 1.0
    return {
        "visual_mean": visual_mean,
        "visual_std": visual_std,
        "geometry_mean": geometry_mean,
        "geometry_std": geometry_std,
        "online_mean": online_mean,
        "online_std": online_std,
    }


def train_model(
    visual: np.ndarray,
    geometry: np.ndarray,
    valid: np.ndarray,
    online: np.ndarray,
    progress: np.ndarray,
    ttc: np.ndarray,
    records: np.ndarray,
    fit: np.ndarray,
    early_stop: np.ndarray,
    cfg: dict,
    device: torch.device,
    checkpoint_path: Path,
    metadata: dict,
) -> tuple[TemporalProgressModel, dict]:
    statistics = training_statistics(
        visual,
        geometry,
        valid,
        online,
        fit,
    )
    normalized = normalize_inputs(
        visual,
        geometry,
        online,
        **statistics,
    )
    fit_weights = record_balanced_weights(fit, records)
    early_weights = record_balanced_weights(early_stop, records)
    progress_class_weights = class_weights(
        progress,
        fit,
        fit_weights,
        len(PROGRESS_NAMES),
    )
    ttc_class_weights = class_weights(
        ttc,
        fit,
        fit_weights,
        len(cfg["ttc_values"]),
    )
    model = TemporalProgressModel(
        visual.shape[2],
        geometry.shape[2],
        online.shape[1],
        int(cfg["projection_dim"]),
        int(cfg["hidden_dim"]),
        len(cfg["ttc_values"]),
        float(cfg["dropout"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    progress_weight_tensor = torch.from_numpy(
        progress_class_weights
    ).to(device)
    ttc_weight_tensor = torch.from_numpy(ttc_class_weights).to(device)
    best_loss, best_epoch, stale, history = float("inf"), 0, 0, []
    for epoch in range(1, int(cfg["epochs"]) + 1):
        model.train()
        order = np.random.permutation(len(fit))
        losses = []
        for start in range(0, len(order), int(cfg["train_batch_size"])):
            local = order[start : start + int(cfg["train_batch_size"])]
            batch = fit[local]
            outputs = model(
                torch.from_numpy(normalized[0][batch]).to(device),
                torch.from_numpy(normalized[1][batch]).to(device),
                torch.from_numpy(valid[batch]).to(device),
                torch.from_numpy(normalized[2][batch]).to(device),
            )
            loss = model_loss(
                outputs,
                torch.from_numpy(progress[batch]).to(device),
                torch.from_numpy(ttc[batch]).to(device),
                torch.from_numpy(fit_weights[local]).to(device),
                progress_weight_tensor,
                ttc_weight_tensor,
                float(cfg["ttc_loss_weight"]),
                float(cfg["ordinal_loss_weight"]),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(),
                float(cfg["gradient_clip"]),
            )
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            outputs = model(
                torch.from_numpy(normalized[0][early_stop]).to(device),
                torch.from_numpy(normalized[1][early_stop]).to(device),
                torch.from_numpy(valid[early_stop]).to(device),
                torch.from_numpy(normalized[2][early_stop]).to(device),
            )
            validation_loss = float(
                model_loss(
                    outputs,
                    torch.from_numpy(progress[early_stop]).to(device),
                    torch.from_numpy(ttc[early_stop]).to(device),
                    torch.from_numpy(early_weights).to(device),
                    progress_weight_tensor,
                    ttc_weight_tensor,
                    float(cfg["ttc_loss_weight"]),
                    float(cfg["ordinal_loss_weight"]),
                ).cpu()
            )
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "validation_loss": validation_loss,
            }
        )
        if validation_loss < best_loss - 1e-6:
            best_loss, best_epoch, stale = validation_loss, epoch, 0
            ensure_dir(checkpoint_path.parent)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "statistics": statistics,
                    "metadata": metadata,
                },
                checkpoint_path,
            )
        else:
            stale += 1
        if stale >= int(cfg["early_stopping_patience"]):
            break
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state"])
    return model, {
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "epochs_ran": len(history),
        "history": history,
    }


def probabilities_for(
    model: TemporalProgressModel,
    visual: np.ndarray,
    geometry: np.ndarray,
    valid: np.ndarray,
    online: np.ndarray,
    indices: np.ndarray,
    device: torch.device,
    checkpoint_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    normalized = normalize_inputs(
        visual,
        geometry,
        online,
        **checkpoint["statistics"],
    )
    model.eval()
    with torch.no_grad():
        output = model(
            torch.from_numpy(normalized[0][indices]).to(device),
            torch.from_numpy(normalized[1][indices]).to(device),
            torch.from_numpy(valid[indices]).to(device),
            torch.from_numpy(normalized[2][indices]).to(device),
        )
        progress_probability = torch.softmax(
            output["progress_logits"],
            dim=1,
        )
        ordinal_probability = torch.sigmoid(output["ordinal_logits"])
        ttc_probability = torch.softmax(output["ttc_logits"], dim=1)
    return (
        progress_probability.cpu().numpy().astype(np.float32),
        ordinal_probability.cpu().numpy().astype(np.float32),
        ttc_probability.cpu().numpy().astype(np.float32),
    )


def nested_indices(
    rows: list[dict[str, str]],
    records: np.ndarray,
    fold: str,
    cfg: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    held_out = np.asarray(
        [
            index
            for index, row in enumerate(rows)
            if row["oof_fold"] == fold
        ],
        dtype=np.int32,
    )
    outer_fit = np.asarray(
        [
            index
            for index, row in enumerate(rows)
            if row["oof_fold"] != fold
        ],
        dtype=np.int32,
    )
    threshold_calibration = np.asarray(
        [
            index
            for index in outer_fit
            if record_hash_split(
                records[index],
                float(cfg["threshold_calibration_fraction"]),
                int(cfg["threshold_calibration_seed"]) + int(fold),
            )
        ],
        dtype=np.int32,
    )
    calibration_set = set(threshold_calibration.tolist())
    fit_and_early = np.asarray(
        [index for index in outer_fit if index not in calibration_set],
        dtype=np.int32,
    )
    early_stop = np.asarray(
        [
            index
            for index in fit_and_early
            if record_hash_split(
                records[index],
                float(cfg["early_stop_fraction"]),
                int(cfg["early_stop_seed"]) + int(fold),
            )
        ],
        dtype=np.int32,
    )
    early_set = set(early_stop.tolist())
    fit = np.asarray(
        [index for index in fit_and_early if index not in early_set],
        dtype=np.int32,
    )
    splits = (fit, early_stop, threshold_calibration, held_out)
    if any(not len(indices) for indices in splits):
        raise RuntimeError(f"Phase4I.5 fold {fold} has an empty nested split")
    record_sets = [set(records[indices].tolist()) for indices in splits]
    if any(
        record_sets[left] & record_sets[right]
        for left in range(len(record_sets))
        for right in range(left + 1, len(record_sets))
    ):
        raise RuntimeError(f"Phase4I.5 fold {fold} nested records overlap")
    return splits


def confusion_report(
    progress_target: np.ndarray,
    predicted_progress: np.ndarray,
    far_target: np.ndarray,
    predicted_risk: np.ndarray,
) -> dict:
    matrix = np.zeros((len(PROGRESS_NAMES), len(PROGRESS_NAMES)), dtype=int)
    for target, prediction in zip(
        progress_target,
        predicted_progress,
        strict=True,
    ):
        matrix[int(target), int(prediction)] += 1
    recalls = {
        name: int(matrix[index, index]) / max(int(matrix[index].sum()), 1)
        for index, name in enumerate(PROGRESS_NAMES)
    }
    actual = far_target.astype(bool)
    tp = int((predicted_risk & actual).sum())
    fp = int((predicted_risk & ~actual).sum())
    tn = int((~predicted_risk & ~actual).sum())
    fn = int((~predicted_risk & actual).sum())
    return {
        "progress_confusion_matrix_true_rows": matrix.tolist(),
        "progress_accuracy": float((progress_target == predicted_progress).mean()),
        "progress_recall": recalls,
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "far_recall": tp / max(tp + fn, 1),
        "far_precision": tp / max(tp + fp, 1),
        "near_mid_retention": tn / max(tn + fp, 1),
        "risk_rate": float(predicted_risk.mean()),
    }


def run(config_path: str, section: str) -> dict:
    cfg = load_config(config_path)[section]
    set_seed(int(cfg["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples = read_csv_rows(project_path(cfg["samples_csv"]))
    assert_development_only(samples, project_path(cfg["final_partition_csv"]))
    development_names = {
        row["image_name"]
        for row in samples
        if row["dataset_split"] == "train"
    }
    rows = read_csv_rows(project_path(cfg["phase4i3_query_csv"]))
    names = [row["query_image_name"] for row in rows]
    if len(set(names)) != len(rows) or set(names) != development_names:
        raise RuntimeError("Phase4I.5 queries do not match development OOF")
    feature_prefix = project_path(cfg["feature_cache_prefix"])
    feature_metadata = validate_temporal_cache_metadata(feature_prefix, cfg)
    visual, geometry, valid, padding = load_temporal_cache(feature_prefix, names)
    minimum_valid = int(cfg["minimum_valid_frames"])
    valid_counts = valid.sum(axis=1)
    if np.any(valid_counts < minimum_valid):
        raise RuntimeError(
            "Phase4I.5 feature cache violates minimum valid-frame contract"
        )
    online = online_progress_features(rows)
    progress, ttc_target = progress_targets(
        rows,
        int(cfg["near_probe_maximum"]),
        int(cfg["mid_probe_maximum"]),
        [int(value) for value in cfg["ttc_values"]],
    )
    far_target = (progress == 2).astype(np.float32)
    records = np.asarray([row["query_record_id"] for row in rows])
    folds = sorted({row["oof_fold"] for row in rows})
    if len(folds) != 3:
        raise RuntimeError(f"Phase4I.5 requires three OOF folds, got {folds}")

    progress_probability = np.zeros(
        (len(rows), len(PROGRESS_NAMES)),
        dtype=np.float32,
    )
    ordinal_probability = np.zeros((len(rows), 2), dtype=np.float32)
    ttc_probability = np.zeros(
        (len(rows), len(cfg["ttc_values"])),
        dtype=np.float32,
    )
    thresholds = np.zeros(len(rows), dtype=np.float32)
    threshold_reports = []
    training_reports = []
    checkpoint_root = project_path(cfg["checkpoint_dir"])
    ensure_dir(checkpoint_root)
    for fold in folds:
        fit, early_stop, calibration, held_out = nested_indices(
            rows,
            records,
            fold,
            cfg,
        )
        if len(np.unique(progress[fit])) != len(PROGRESS_NAMES):
            raise RuntimeError(
                f"Phase4I.5 fold {fold} fit misses progress classes"
            )
        if len(np.unique(ttc_target[fit])) != len(cfg["ttc_values"]):
            raise RuntimeError(f"Phase4I.5 fold {fold} fit misses TTC classes")
        for split_name, indices in (
            ("early_stop", early_stop),
            ("threshold_calibration", calibration),
        ):
            if len(np.unique(far_target[indices])) != 2:
                raise RuntimeError(
                    f"Phase4I.5 fold {fold} {split_name} "
                    "requires far and non-far samples"
                )
        calibration_far_values = []
        held_progress_values = []
        held_ordinal_values = []
        held_ttc_values = []
        for seed in [int(value) for value in cfg["seeds"]]:
            print(
                f"phase4i5 fold {fold} seed {seed}: "
                "training temporal progress model",
                flush=True,
            )
            set_seed(seed)
            checkpoint_path = checkpoint_root / f"fold_{fold}_seed_{seed}.pt"
            model, training_report = train_model(
                visual,
                geometry,
                valid,
                online,
                progress,
                ttc_target,
                records,
                fit,
                early_stop,
                cfg,
                device,
                checkpoint_path,
                {
                    "scope": "strict_oof_temporal_visual_progress",
                    "fold": fold,
                    "seed": seed,
                    "frame_offsets": list(cfg["frame_offsets"]),
                    "query_true_probe_feature_used": False,
                    "true_probe_offline_target_used": True,
                    "query_tactile_input": False,
                },
            )
            calibration_output = probabilities_for(
                model,
                visual,
                geometry,
                valid,
                online,
                calibration,
                device,
                checkpoint_path,
            )
            held_output = probabilities_for(
                model,
                visual,
                geometry,
                valid,
                online,
                held_out,
                device,
                checkpoint_path,
            )
            calibration_far_values.append(
                0.5
                * (
                    calibration_output[0][:, 2]
                    + calibration_output[1][:, 1]
                )
            )
            held_progress_values.append(held_output[0])
            held_ordinal_values.append(held_output[1])
            held_ttc_values.append(held_output[2])
            training_reports.append(
                {
                    "fold": fold,
                    "seed": seed,
                    "fit_queries": len(fit),
                    "early_stop_queries": len(early_stop),
                    "threshold_calibration_queries": len(calibration),
                    "held_out_queries": len(held_out),
                    **training_report,
                }
            )
        calibration_far = np.mean(
            calibration_far_values,
            axis=0,
        ).astype(np.float32)
        selected, options = choose_recall_threshold(
            calibration_far,
            far_target[calibration],
            float(cfg["minimum_far_recall"]),
        )
        threshold = float(selected["threshold"])
        progress_probability[held_out] = np.mean(
            held_progress_values,
            axis=0,
        )
        ordinal_probability[held_out] = np.mean(
            held_ordinal_values,
            axis=0,
        )
        ttc_probability[held_out] = np.mean(held_ttc_values, axis=0)
        thresholds[held_out] = threshold
        for option in options:
            threshold_reports.append(
                {
                    "held_out_fold": fold,
                    **option,
                    "selected": option is selected,
                }
            )
        print(
            f"phase4i5 fold {fold}: threshold={threshold:.6f} "
            f"calibration_recall={selected['recall']:.3f} "
            f"calibration_fpr={selected['false_positive_rate']:.3f}",
            flush=True,
        )

    far_probability = 0.5 * (
        progress_probability[:, 2] + ordinal_probability[:, 1]
    )
    predicted_risk = far_probability >= thresholds
    predicted_progress = progress_probability.argmax(axis=1)
    ttc_axis = np.asarray(cfg["ttc_values"], dtype=np.float32)
    predicted_ttc = ttc_probability @ ttc_axis
    true_ttc = ttc_axis[ttc_target]
    classification = confusion_report(
        progress,
        predicted_progress,
        far_target,
        predicted_risk,
    )
    classification["ttc_mae_frames"] = float(
        np.abs(predicted_ttc - true_ttc).mean()
    )
    classification["by_oof_fold"] = {}
    for fold in folds:
        indices = np.asarray(
            [
                index
                for index, row in enumerate(rows)
                if row["oof_fold"] == fold
            ],
            dtype=np.int32,
        )
        fold_report = confusion_report(
            progress[indices],
            predicted_progress[indices],
            far_target[indices],
            predicted_risk[indices],
        )
        fold_report["ttc_mae_frames"] = float(
            np.abs(predicted_ttc[indices] - true_ttc[indices]).mean()
        )
        classification["by_oof_fold"][fold] = fold_report
    classification_contract_pass = bool(
        classification["far_recall"] >= float(cfg["minimum_far_recall"])
        and classification["near_mid_retention"]
        >= float(cfg["minimum_near_mid_retention"])
    )

    v1_rows = retrieval_rows(rows, "v1")
    phase4i3_rows = retrieval_rows(rows, "phase4i3")
    gated_rows = select_by_risk(v1_rows, phase4i3_rows, predicted_risk)
    comparison = fast_bootstrap_comparison(
        v1_rows,
        gated_rows,
        {
            "bootstrap_iterations": int(cfg["bootstrap_iterations"]),
            "bootstrap_seed": int(cfg["bootstrap_seed"]),
        },
    )
    accepted = bool(classification_contract_pass and comparison["accepted"])
    final_rows = gated_rows if accepted else v1_rows

    output = []
    for index, (source, v1, current, gated, final) in enumerate(
        zip(rows, v1_rows, phase4i3_rows, gated_rows, final_rows, strict=True)
    ):
        gate_source = "v1_far_risk" if predicted_risk[index] else "phase4i3"
        output.append(
            {
                "query_record_id": source["query_record_id"],
                "query_image_name": source["query_image_name"],
                "query_probe": source["query_probe"],
                "oof_fold": source["oof_fold"],
                "temporal_near_probability": (
                    f"{progress_probability[index, 0]:.9f}"
                ),
                "temporal_mid_probability": (
                    f"{progress_probability[index, 1]:.9f}"
                ),
                "temporal_far_class_probability": (
                    f"{progress_probability[index, 2]:.9f}"
                ),
                "temporal_far_ordinal_probability": (
                    f"{ordinal_probability[index, 1]:.9f}"
                ),
                "temporal_far_risk_probability": (
                    f"{far_probability[index]:.9f}"
                ),
                "far_risk_threshold": f"{thresholds[index]:.9f}",
                "predicted_far_risk": str(int(predicted_risk[index])),
                "true_far_label": str(int(far_target[index])),
                "predicted_progress_class": PROGRESS_NAMES[
                    int(predicted_progress[index])
                ],
                "true_progress_class": PROGRESS_NAMES[int(progress[index])],
                "predicted_ttc": f"{predicted_ttc[index]:.6f}",
                "true_ttc": f"{true_ttc[index]:.6f}",
                "temporal_frame_valid_fraction": (
                    f"{valid[index].mean():.6f}"
                ),
                "temporal_padding_ratio": (
                    f"{padding[index][valid[index] > 0.5].mean():.6f}"
                ),
                "phase4i3_selected_cache_image_name": current[
                    "selected_cache_image_name"
                ],
                "v1_selected_cache_image_name": v1[
                    "selected_cache_image_name"
                ],
                "gated_selected_cache_image_name": gated[
                    "selected_cache_image_name"
                ],
                "gated_selection_source": gate_source,
                "gated_ranker_oracle_embedding_rank": gated[
                    "ranker_oracle_embedding_rank"
                ],
                **{
                    f"gated_{metric}": gated[metric]
                    for metric in METRICS
                },
                "deployment_accepted": str(int(accepted)),
                "final_selection_source": (
                    gate_source if accepted else "v1"
                ),
                "selected_cache_image_name": final[
                    "selected_cache_image_name"
                ],
                "ranker_oracle_embedding_rank": final[
                    "ranker_oracle_embedding_rank"
                ],
                **{metric: final[metric] for metric in METRICS},
            }
        )
    threshold_output = [
        threshold_csv_row(row) for row in threshold_reports
    ]
    assert_csv_schema(output, QUERY_FIELDS, "Phase4I.5 query output")
    assert_csv_schema(
        threshold_output,
        THRESHOLD_FIELDS,
        "Phase4I.5 threshold output",
    )
    write_csv_rows(
        project_path(cfg["query_output_csv"]),
        output,
        QUERY_FIELDS,
    )
    write_csv_rows(
        project_path(cfg["threshold_output_csv"]),
        threshold_output,
        THRESHOLD_FIELDS,
    )

    report = {
        "mode": "phase4i5_temporal_visual_progress_oof_v1",
        "device": str(device),
        "temporal_input": {
            "frame_offsets": list(cfg["frame_offsets"]),
            "crop_definition": "tip-to-C2-prediction corridor midpoint",
            "crop_size": int(cfg["crop_size"]),
            "dino_model": str(cfg["dino_model"]),
            "dino_layer": int(cfg["dino_layer"]),
            "valid_frame_fraction": float(valid.mean()),
            "mean_padding_ratio": float(padding.mean()),
            "cache_query_identity_sha256": feature_metadata[
                "query_identity_sha256"
            ],
        },
        "classification_contract": {
            "minimum_far_recall": float(cfg["minimum_far_recall"]),
            "minimum_near_mid_retention": float(
                cfg["minimum_near_mid_retention"]
            ),
            "oof": classification,
            "passed": classification_contract_pass,
        },
        "gated": {
            "summary": {
                "all": metric_summary(gated_rows),
                "far_probe75_100": metric_summary(
                    gated_rows,
                    lambda row: int(row["query_probe"])
                    > int(cfg["mid_probe_maximum"]),
                ),
            },
            "vs_v1": comparison,
        },
        "deployment_accepted": accepted,
        "deployed": {
            "source": "phase4i5" if accepted else "v1",
            "summary": {
                "all": metric_summary(final_rows),
                "far_probe75_100": metric_summary(
                    final_rows,
                    lambda row: int(row["query_probe"])
                    > int(cfg["mid_probe_maximum"]),
                ),
            },
        },
        "training": training_reports,
        "integrity": {
            "source": "strict development 3-fold record-level OOF only",
            "nested_split": (
                "fit / early-stop / threshold-calibration / held-out "
                "records are mutually exclusive"
            ),
            "query_true_probe_feature_used": False,
            "true_probe_offline_target_used": True,
            "query_tactile_input": False,
            "future_visual_frames_used": False,
            "c2_contact_box": "unchanged",
            "top32_candidates": "unchanged Phase4I.3",
            "sealed_final_holdout_rows_read": 0,
            "development_validation_rows_read": 0,
        },
        "next_action": (
            "freeze Phase4I.5 and implement independent development validation"
            if accepted
            else (
                "retain V1; if classification passed, Phase4I.3 retrieval "
                "benefit is too weak; otherwise collect targeted temporal far records"
            )
        ),
    }
    write_json(project_path(cfg["metrics_json"]), report)
    print(
        {
            "mode": report["mode"],
            "classification_contract": report["classification_contract"],
            "gated": report["gated"],
            "deployment_accepted": report["deployment_accepted"],
            "deployed": report["deployed"],
            "integrity": report["integrity"],
            "next_action": report["next_action"],
        }
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train strict-OOF temporal visual contact-progress model."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--section",
        default="phase4i5_temporal_progress_oof_v1",
    )
    args = parser.parse_args()
    run(args.config, args.section)


if __name__ == "__main__":
    main()
