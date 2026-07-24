"""Freeze and audit the Phase4H development-only experiment contract."""
from __future__ import annotations

import argparse
import hashlib
import inspect
from pathlib import Path

import numpy as np

from .config import load_config, project_path
from .phase4h_dino_adaptation import (
    PHASE4H_QUERY_FORBIDDEN_FIELDS,
    TactileLatentProjector,
    assert_candidate_identity,
    assert_development_only,
    candidate_groups,
    candidate_set_fingerprint,
)
from .utils import read_csv_rows, write_json


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def query_summary(rows: list[dict[str, str]]) -> dict:
    output = {"queries": len(rows)}
    for name, subset in (
        ("all", rows),
        ("near_probe5_20", [row for row in rows if int(row["query_probe"]) <= 20]),
        ("mid_probe30_50", [row for row in rows if 30 <= int(row["query_probe"]) <= 50]),
        ("far_probe75_100", [row for row in rows if int(row["query_probe"]) >= 75]),
    ):
        output[name] = {
            "queries": len(subset),
            "tactile_diff_mae": float(np.mean([float(row["tactile_diff_mae"]) for row in subset])) if subset else None,
            "tactile_ssim": float(np.mean([float(row["tactile_ssim"]) for row in subset])) if subset else None,
            "tactile_mask_iou": float(np.mean([float(row["tactile_mask_iou"]) for row in subset])) if subset else None,
            "oracle_top1": float(np.mean([int(row["ranker_oracle_embedding_rank"]) == 1 for row in subset])) if subset else None,
            "oracle_top3": float(np.mean([int(row["ranker_oracle_embedding_rank"]) <= 3 for row in subset])) if subset else None,
        }
    return output


def audit(config_path: str, section: str) -> dict:
    cfg = load_config(config_path)[section]
    paths = {
        name: project_path(cfg[name])
        for name in (
            "samples_csv",
            "oof_predictions_csv",
            "v1_query_csv",
            "v1_candidate_csv",
            "dino_query_csv",
            "dino_candidate_csv",
        )
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Phase4H protocol inputs are missing: {missing}")
    partition_path = project_path(cfg["final_partition_csv"])
    if not partition_path.exists():
        raise FileNotFoundError(
            f"Phase4H requires the fixed V2 partition manifest before development: {partition_path}"
        )

    samples = read_csv_rows(paths["samples_csv"])
    assert_development_only(samples, partition_path)
    train_rows = [row for row in samples if row["dataset_split"] == "train"]
    train_names = {row["image_name"] for row in train_rows}
    if len(train_names) != len(train_rows):
        raise RuntimeError("Development-train image names are not unique")

    predictions = [
        row for row in read_csv_rows(paths["oof_predictions_csv"])
        if row.get("dataset_split") == "train" and row["image_name"] in train_names
    ]
    if {row["image_name"] for row in predictions} != train_names or len(predictions) != len(train_names):
        raise RuntimeError("Frozen C2 OOF predictions do not cover development-train exactly once")
    fold_by_record: dict[str, str] = {}
    for prediction in predictions:
        fold = prediction.get("oof_fold", "")
        if not fold:
            raise RuntimeError("C2 OOF prediction is missing oof_fold")
        record = prediction["record_id"]
        if record in fold_by_record and fold_by_record[record] != fold:
            raise RuntimeError(f"Record {record} appears in more than one OOF fold")
        fold_by_record[record] = fold
    if len(set(fold_by_record.values())) != 3:
        raise RuntimeError(f"Phase4H requires exactly three record-level OOF folds, got {set(fold_by_record.values())}")

    top_k = int(cfg["geometry_filter_k"])
    v1_groups = candidate_groups(read_csv_rows(paths["v1_candidate_csv"]), top_k)
    dino_groups = candidate_groups(read_csv_rows(paths["dino_candidate_csv"]), top_k)
    assert_candidate_identity(v1_groups, dino_groups)
    if set(v1_groups) != train_names:
        raise RuntimeError("Frozen V1 candidate queries do not cover development-train exactly")
    sample_by_name = {row["image_name"]: row for row in train_rows}
    for query, group in v1_groups.items():
        query_record = sample_by_name[query]["record_id"]
        if any(row["candidate_record_id"] == query_record for row in group):
            raise RuntimeError(f"Same-record candidate leaked into {query}")

    forward_parameters = set(inspect.signature(TactileLatentProjector.forward).parameters)
    forbidden_parameters = forward_parameters & PHASE4H_QUERY_FORBIDDEN_FIELDS
    if forbidden_parameters:
        raise RuntimeError(f"Online projector exposes forbidden tactile/target inputs: {forbidden_parameters}")
    v1_queries = read_csv_rows(paths["v1_query_csv"])
    dino_queries = read_csv_rows(paths["dino_query_csv"])
    if {row["query_image_name"] for row in v1_queries} != train_names:
        raise RuntimeError("Frozen V1 query table does not match development-train")
    if {row["query_image_name"] for row in dino_queries} != train_names:
        raise RuntimeError("Frozen DINO query table does not match development-train")

    report = {
        "mode": "phase4h_frozen_protocol_audit_v1",
        "passed": True,
        "counts": {
            "development_samples": len(samples),
            "development_train_queries": len(train_rows),
            "records": len({(row["split"], row["record_id"]) for row in samples}),
            "oof_folds": len(set(fold_by_record.values())),
            "candidates_per_query": top_k,
        },
        "frozen_baselines": {
            "v1_multiscale": query_summary(v1_queries),
            "dino_patch_similarity": query_summary(dino_queries),
            "cross_attention": "rejected; excluded from Phase4H model selection",
        },
        "integrity": {
            "candidate_identity_passed": True,
            "candidate_set_fingerprint": candidate_set_fingerprint(v1_groups),
            "same_record_candidates": 0,
            "sealed_final_holdout_rows_read": 0,
            "query_forward_parameters": sorted(forward_parameters),
            "query_forbidden_parameters": sorted(forbidden_parameters),
            "query_true_probe_usage": "evaluation bins only",
            "query_tactile_usage": "offline supervision and evaluation only",
        },
        "input_fingerprints": {
            name: {"path": str(path), "sha256": file_sha256(path)}
            for name, path in paths.items()
        }
        | {
            "final_partition_csv": {
                "path": str(partition_path),
                "sha256": file_sha256(partition_path),
            }
        },
    }
    write_json(project_path(cfg["audit_json"]), report)
    print(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit and freeze the Phase4H development-only protocol.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--section", default="phase4h_protocol_v1")
    args = parser.parse_args()
    audit(args.config, args.section)


if __name__ == "__main__":
    main()
