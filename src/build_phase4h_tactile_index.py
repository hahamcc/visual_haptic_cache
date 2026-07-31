"""Build the fixed 77-D candidate tactile index with strict OOF statistics."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

from .config import load_config, project_path
from .evaluate_oracle_tactile_retrieval import tactile_difference
from .phase4h_dino_adaptation import TACTILE_LATENT_DIM, assert_development_only, tactile_latent
from .train_phase4b_predicted_box_cache_ranker import prediction_map
from .utils import ensure_dir, read_csv_rows, write_json


def fingerprint_rows(rows: list[dict[str, str]]) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda item: item["image_name"]):
        digest.update(
            f"{row['split']}|{row['record_id']}|{row['image_name']}|{row['touch_path']}\n".encode("utf-8")
        )
    return digest.hexdigest()


def safe_stats(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean, std = values.mean(axis=0), values.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def build(config_path: str, section: str) -> dict:
    cfg = load_config(config_path)[section]
    samples_path = project_path(cfg["samples_csv"])
    partition_path = project_path(cfg["final_partition_csv"])
    rows = read_csv_rows(samples_path)
    assert_development_only(rows, partition_path)
    rows = [row for row in rows if row["dataset_split"] == "train"]
    predictions = prediction_map(
        read_csv_rows(project_path(cfg["oof_predictions_csv"])),
        rows,
        "train",
        "Phase4H tactile-index OOF",
    )
    folds = sorted({prediction["oof_fold"] for prediction in predictions.values()})
    fold_by_name = {name: prediction["oof_fold"] for name, prediction in predictions.items()}
    touch_cache: dict[str, np.ndarray] = {}
    values = []
    for index, row in enumerate(rows, start=1):
        diff = tactile_difference(row["touch_path"], touch_cache, int(cfg["tactile_size"]))
        values.append(tactile_latent(diff, float(cfg["tactile_mask_threshold"])))
        if index % 200 == 0 or index == len(rows):
            print(f"phase4h tactile index: {index}/{len(rows)}", flush=True)
    raw = np.stack(values).astype(np.float32)
    if raw.shape != (len(rows), TACTILE_LATENT_DIM):
        raise RuntimeError(f"Unexpected tactile index shape: {raw.shape}")
    full_mean, full_std = safe_stats(raw)
    fold_means, fold_stds = [], []
    for fold in folds:
        fit = np.asarray([fold_by_name[row["image_name"]] != fold for row in rows], dtype=bool)
        if not fit.any():
            raise RuntimeError(f"OOF fold {fold} has no tactile-index fit rows")
        mean, std = safe_stats(raw[fit])
        fold_means.append(mean)
        fold_stds.append(std)
    output_path = project_path(cfg["index_npz"])
    ensure_dir(output_path.parent)
    np.savez_compressed(
        output_path,
        image_names=np.asarray([row["image_name"] for row in rows]),
        record_ids=np.asarray([row["record_id"] for row in rows]),
        split_ids=np.asarray([row["split"] for row in rows]),
        oof_folds=np.asarray([fold_by_name[row["image_name"]] for row in rows]),
        tactile_latents=raw,
        fold_names=np.asarray(folds),
        fold_means=np.stack(fold_means),
        fold_stds=np.stack(fold_stds),
        full_mean=full_mean,
        full_std=full_std,
        latent_dim=np.asarray([TACTILE_LATENT_DIM], dtype=np.int32),
        tactile_mask_threshold=np.asarray([float(cfg["tactile_mask_threshold"])], dtype=np.float32),
    )
    report = {
        "mode": "phase4h_fixed_77d_tactile_index_v1",
        "rows": len(rows),
        "records": len({(row["split"], row["record_id"]) for row in rows}),
        "latent_dim": TACTILE_LATENT_DIM,
        "oof_folds": folds,
        "index_npz": str(output_path),
        "dataset_fingerprint": fingerprint_rows(rows),
        "statistics": {
            "full_fit_rows": len(rows),
            "fold_fit_rows": {
                fold: int(sum(fold_by_name[row["image_name"]] != fold for row in rows))
                for fold in folds
            },
        },
        "integrity": {
            "sealed_final_holdout_rows_read": 0,
            "fold_statistics_use_heldout_fold": False,
            "query_tactile_usage": "offline supervision and evaluation only",
            "online_candidate_tactile_usage": "precomputed cache index only",
        },
    }
    write_json(project_path(cfg["metrics_json"]), report)
    print(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the strict Phase4H 77-D tactile cache index.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--section", default="phase4h_tactile_index_v1")
    args = parser.parse_args()
    build(args.config, args.section)


if __name__ == "__main__":
    main()
