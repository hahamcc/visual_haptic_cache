"""Select record-disjoint temporal-far pairs from robust Phase4I.6 pools."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from .build_phase4h_dino_ablation import encode_rows
from .config import load_config, project_path
from .phase4f_dino_cross_attention import FrozenDinoV2
from .phase4h_dino_adaptation import contact_crop_reflect, final_holdout_keys
from .utils import ensure_dir, read_csv_rows, write_csv_rows, write_json


MOTION_FIELDS = [
    "robust_mean_speed_px",
    "robust_speed_cv",
    "robust_normalized_speed_slope",
    "robust_mean_acceleration_px",
    "robust_direction_stability",
    "robust_cumulative_turn_radians",
    "robust_pause_ratio",
    "robust_cumulative_displacement_px",
]

PAIR_FIELDS = [
    "pair_rank",
    "planned_pair_id",
    "record_a",
    "record_b",
    "image_a",
    "image_b",
    "detail_similarity",
    "context_similarity",
    "visual_similarity",
    "motion_distance",
    "reciprocal_visual_rank",
    "pair_score",
    "pair_design",
    "selection_status",
    "manual_same_object_region_approved",
    "review_image_path",
    "notes",
]

SELECTION_FIELDS = [
    "selection_rank",
    "planned_pair_id",
    "pair_variant",
    "pair_design",
    "source_pool",
    "split",
    "record_id",
    "image_name",
    "robust_motion_profile",
    "track_quality_passed",
    *MOTION_FIELDS,
    "speed_bin",
    "slope_bin",
    "stability_bin",
    "turn_bin",
    "pause_bin",
    "paired_record_id",
    "visual_similarity",
    "motion_distance",
    "selection_status",
    "manual_same_object_region_approved",
]

BIN_FIELDS = {
    "speed_bin": "robust_mean_speed_px",
    "slope_bin": "robust_normalized_speed_slope",
    "stability_bin": "robust_direction_stability",
    "turn_bin": "robust_cumulative_turn_radians",
    "pause_bin": "robust_pause_ratio",
}


def merge_passed_records(
    pools: list[tuple[str, list[dict[str, str]]]],
) -> list[dict[str, str]]:
    merged = []
    seen: set[tuple[str, str]] = set()
    for pool_name, rows in pools:
        for source in rows:
            if source["track_quality_passed"] != "1":
                continue
            key = (source["split"], source["record_id"])
            if key in seen:
                raise RuntimeError(
                    f"Phase4I.6E robust pools overlap at {key[0]}/{key[1]}"
                )
            seen.add(key)
            merged.append({**source, "source_pool": pool_name})
    merged.sort(key=lambda row: (row["split"], row["record_id"]))
    return merged


def quantile_bin_assignments(
    rows: list[dict[str, str]],
    fields: dict[str, str],
    bins: int,
) -> tuple[dict[str, np.ndarray], dict[str, list[float]]]:
    if bins < 2:
        raise ValueError("Phase4I.6E requires at least two quantile bins")
    assignments: dict[str, np.ndarray] = {}
    edges_by_name: dict[str, list[float]] = {}
    quantiles = np.linspace(0.0, 1.0, bins + 1)[1:-1]
    for output_name, field in fields.items():
        values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
        edges = np.quantile(values, quantiles)
        assignments[output_name] = np.searchsorted(
            edges, values, side="right"
        ).astype(np.int64)
        edges_by_name[output_name] = [float(value) for value in edges]
    return assignments, edges_by_name


def standardized_motion(rows: list[dict[str, str]]) -> np.ndarray:
    values = np.asarray(
        [[float(row[field]) for field in MOTION_FIELDS] for row in rows],
        dtype=np.float32,
    )
    median = np.median(values, axis=0)
    q25, q75 = np.quantile(values, [0.25, 0.75], axis=0)
    scale = np.maximum(q75 - q25, 1e-6)
    return ((values - median) / scale).astype(np.float32)


def normalize_rows(values: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norm, 1e-8)


def visual_rank_matrix(similarity: np.ndarray) -> np.ndarray:
    count = len(similarity)
    ranks = np.full((count, count), count, dtype=np.int64)
    for index in range(count):
        order = [
            candidate
            for candidate in np.argsort(-similarity[index], kind="stable")
            if candidate != index
        ]
        for rank, candidate in enumerate(order, start=1):
            ranks[index, candidate] = rank
    return ranks


def _coverage_bonus(
    left: int,
    right: int,
    assignments: dict[str, np.ndarray],
    counts: dict[str, Counter],
    target_per_bin: float,
) -> float:
    bonus = 0.0
    terms = 0
    for name, values in assignments.items():
        for index in (left, right):
            deficit = max(target_per_bin - counts[name][int(values[index])], 0.0)
            bonus += deficit / max(target_per_bin, 1.0)
            terms += 1
    return bonus / max(terms, 1)


def greedy_pair_selection(
    record_ids: list[str],
    visual_similarity: np.ndarray,
    motion: np.ndarray,
    assignments: dict[str, np.ndarray],
    target_pairs: int,
    top_k_schedule: list[int],
    motion_weight: float,
    diversity_weight: float,
    quantile_bins: int,
) -> tuple[list[dict[str, float | int]], int]:
    count = len(record_ids)
    if target_pairs * 2 > count:
        raise RuntimeError(
            f"Phase4I.6E needs {target_pairs * 2} records, only {count} exist"
        )
    ranks = visual_rank_matrix(visual_similarity)
    delta = motion[:, None, :] - motion[None, :, :]
    motion_distance = np.linalg.norm(delta, axis=2) / math.sqrt(
        max(motion.shape[1], 1)
    )
    target_per_bin = target_pairs * 2 / float(quantile_bins)
    for top_k in top_k_schedule:
        available = set(range(count))
        selected: list[dict[str, float | int]] = []
        counts = {name: Counter() for name in assignments}
        while len(selected) < target_pairs:
            best: tuple[float, float, str, str, int, int] | None = None
            for left in sorted(available):
                for right in sorted(index for index in available if index > left):
                    reciprocal_rank = max(
                        int(ranks[left, right]),
                        int(ranks[right, left]),
                    )
                    if reciprocal_rank > top_k:
                        continue
                    contrast = float(motion_distance[left, right])
                    bonus = _coverage_bonus(
                        left,
                        right,
                        assignments,
                        counts,
                        target_per_bin,
                    )
                    score = (
                        float(visual_similarity[left, right])
                        + motion_weight * math.tanh(contrast / 2.0)
                        + diversity_weight * bonus
                    )
                    candidate = (
                        score,
                        float(visual_similarity[left, right]),
                        record_ids[left],
                        record_ids[right],
                        left,
                        right,
                    )
                    if best is None or candidate > best:
                        best = candidate
            if best is None:
                break
            score, _, _, _, left, right = best
            selected.append(
                {
                    "left": left,
                    "right": right,
                    "reciprocal_rank": max(
                        int(ranks[left, right]),
                        int(ranks[right, left]),
                    ),
                    "visual_similarity": float(
                        visual_similarity[left, right]
                    ),
                    "motion_distance": float(
                        motion_distance[left, right]
                    ),
                    "pair_score": float(score),
                }
            )
            available.remove(left)
            available.remove(right)
            for name, values in assignments.items():
                counts[name][int(values[left])] += 1
                counts[name][int(values[right])] += 1
        if len(selected) == target_pairs:
            return selected, top_k
    raise RuntimeError(
        "Phase4I.6E could not form the requested disjoint visual pairs"
    )


def pair_design(
    left: int,
    right: int,
    standardized: np.ndarray,
) -> str:
    field = MOTION_FIELDS[int(np.argmax(np.abs(standardized[left] - standardized[right])))]
    if field in {
        "robust_mean_speed_px",
        "robust_speed_cv",
        "robust_mean_acceleration_px",
    }:
        return "same_region_different_speed"
    if field in {
        "robust_normalized_speed_slope",
        "robust_cumulative_displacement_px",
    }:
        return "same_region_progress_contrast"
    return "same_region_motion_pattern_contrast"


def sample_index(
    sample_pools: list[list[dict[str, str]]],
    allowed: set[tuple[str, str]],
    probes: set[int],
) -> tuple[
    dict[tuple[str, str, int], dict[str, str]],
    list[dict[str, str]],
]:
    indexed: dict[tuple[str, str, int], dict[str, str]] = {}
    all_rows = []
    for rows in sample_pools:
        for row in rows:
            key2 = (row["split"], row["record_id"])
            probe = int(row["probe"])
            if key2 not in allowed or probe not in probes:
                continue
            key = (*key2, probe)
            if key in indexed:
                raise RuntimeError(f"Phase4I.6E duplicate sample {key}")
            indexed[key] = row
            all_rows.append(row)
    expected = len(allowed) * len(probes)
    if len(indexed) != expected:
        raise RuntimeError(
            f"Phase4I.6E needs one sample per record/probe: "
            f"{len(indexed)} != {expected}"
        )
    return indexed, all_rows


def save_pair_review(
    left: dict[str, str],
    right: dict[str, str],
    output: Path,
    detail_size: int,
    context_size: int,
    label: str,
) -> None:
    panels = []
    for row in (left, right):
        x, y = float(row["target_tip_x"]), float(row["target_tip_y"])
        detail, _ = contact_crop_reflect(
            row["vision_path"], x, y, detail_size
        )
        context, _ = contact_crop_reflect(
            row["vision_path"], x, y, context_size
        )
        panels.append(
            (
                Image.fromarray((detail * 255.0).astype(np.uint8)).resize(
                    (192, 192), Image.Resampling.BILINEAR
                ),
                Image.fromarray((context * 255.0).astype(np.uint8)).resize(
                    (192, 192), Image.Resampling.BILINEAR
                ),
            )
        )
    canvas = Image.new("RGB", (768, 235), "white")
    draw = ImageDraw.Draw(canvas)
    for row_index, (detail, context) in enumerate(panels):
        offset = row_index * 384
        canvas.paste(detail, (offset, 0))
        canvas.paste(context, (offset + 192, 0))
        row = left if row_index == 0 else right
        draw.text(
            (offset + 4, 197),
            f"{row['record_id']}  detail | context",
            fill="black",
        )
    draw.text((4, 217), label, fill="black")
    ensure_dir(output.parent)
    canvas.save(output, quality=92)


def save_review_contact_sheet(
    review_paths: list[Path],
    output: Path,
    columns: int = 2,
) -> None:
    cell_width, cell_height = 384, 118
    rows = int(math.ceil(len(review_paths) / max(columns, 1)))
    sheet = Image.new(
        "RGB",
        (cell_width * columns, cell_height * rows),
        "white",
    )
    for index, path in enumerate(review_paths):
        with Image.open(path) as handle:
            image = handle.convert("RGB").resize(
                (cell_width, cell_height),
                Image.Resampling.LANCZOS,
            )
        x = (index % columns) * cell_width
        y = (index // columns) * cell_height
        sheet.paste(image, (x, y))
    ensure_dir(output.parent)
    sheet.save(output, quality=92)


def select(config_path: str, section: str) -> dict:
    cfg = load_config(config_path)[section]
    robust_paths = [project_path(path) for path in cfg["robust_record_csvs"]]
    sample_paths = [project_path(path) for path in cfg["candidate_sample_csvs"]]
    pool_names = [str(value) for value in cfg["pool_names"]]
    if len(robust_paths) != len(sample_paths) or len(pool_names) != len(
        robust_paths
    ):
        raise RuntimeError("Phase4I.6E pool configuration lengths differ")
    robust = merge_passed_records(
        [
            (pool_name, read_csv_rows(path))
            for pool_name, path in zip(pool_names, robust_paths, strict=True)
        ]
    )
    target_records = int(cfg["target_records"])
    if target_records % 2:
        raise RuntimeError("Phase4I.6E target_records must be even")
    if len(robust) < target_records:
        raise RuntimeError(
            f"Phase4I.6E robust pool is too small: {len(robust)} < "
            f"{target_records}"
        )
    keys = {(row["split"], row["record_id"]) for row in robust}
    sealed = final_holdout_keys(project_path(cfg["final_partition_csv"]))
    overlap = sorted(keys & sealed)
    if overlap:
        raise RuntimeError(
            f"Phase4I.6E robust pool overlaps final holdout: {overlap[:3]}"
        )
    probes = {int(value) for value in cfg["probe_focus"]}
    samples, sample_rows = sample_index(
        [read_csv_rows(path) for path in sample_paths],
        keys,
        probes,
    )
    descriptor_probe = int(cfg["descriptor_probe"])
    descriptor_rows = [
        samples[(row["split"], row["record_id"], descriptor_probe)]
        for row in robust
    ]
    coordinates = [
        (float(row["target_tip_x"]), float(row["target_tip_y"]))
        for row in descriptor_rows
    ]
    descriptor_identity = {
        "model": str(cfg["dino_model"]),
        "input_size": int(cfg["dino_input_size"]),
        "layer_recipe": str(cfg["dino_layer_recipe"]),
        "record_images": [row["image_name"] for row in descriptor_rows],
        "target_coordinates": coordinates,
    }
    descriptor_fingerprint = hashlib.sha256(
        json.dumps(
            descriptor_identity,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    requested_device = str(cfg.get("device", "auto"))
    device = torch.device(
        "cuda"
        if requested_device == "auto" and torch.cuda.is_available()
        else ("cpu" if requested_device == "auto" else requested_device)
    )
    print(
        f"phase4i6e: loading frozen {cfg['dino_model']} on {device}",
        flush=True,
    )
    backbone = FrozenDinoV2(
        str(cfg["dino_model"]),
        int(cfg["dino_input_size"]),
    ).to(device)
    pooled = {}
    padding = {}
    for name, size in (
        ("detail", int(cfg["detail_crop_size"])),
        ("context", int(cfg["context_crop_size"])),
    ):
        _, pooled[name], padding[name] = encode_rows(
            backbone,
            descriptor_rows,
            coordinates,
            size,
            "reflect",
            "raw",
            str(cfg["dino_layer_recipe"]),
            {},
            device,
            int(cfg["batch_size"]),
            float(cfg["center_sigma"]),
            f"phase4i6e {name}",
            Path(
                f"{project_path(cfg[f'{name}_cache_prefix'])}_"
                f"{descriptor_fingerprint[:16]}"
            ),
        )
        pooled[name] = normalize_rows(np.asarray(pooled[name]))
    detail_similarity = pooled["detail"] @ pooled["detail"].T
    context_similarity = pooled["context"] @ pooled["context"].T
    visual_similarity = (
        float(cfg["detail_weight"]) * detail_similarity
        + float(cfg["context_weight"]) * context_similarity
    )
    np.fill_diagonal(visual_similarity, -1.0)
    motion = standardized_motion(robust)
    assignments, bin_edges = quantile_bin_assignments(
        robust,
        BIN_FIELDS,
        int(cfg["quantile_bins"]),
    )
    selected_pairs, selected_top_k = greedy_pair_selection(
        [row["record_id"] for row in robust],
        visual_similarity,
        motion,
        assignments,
        target_records // 2,
        [int(value) for value in cfg["visual_top_k_schedule"]],
        float(cfg["motion_contrast_weight"]),
        float(cfg["diversity_weight"]),
        int(cfg["quantile_bins"]),
    )
    pair_rows = []
    selection_rows = []
    selected_sample_rows = []
    review_paths = []
    review_dir = project_path(cfg["review_dir"])
    for pair_index, pair in enumerate(selected_pairs, start=1):
        left_index, right_index = int(pair["left"]), int(pair["right"])
        left, right = robust[left_index], robust[right_index]
        left_sample = descriptor_rows[left_index]
        right_sample = descriptor_rows[right_index]
        planned_pair_id = f"pair_{pair_index:03d}"
        design = pair_design(left_index, right_index, motion)
        review_path = review_dir / f"{planned_pair_id}_{left['record_id']}_{right['record_id']}.jpg"
        save_pair_review(
            left_sample,
            right_sample,
            review_path,
            int(cfg["detail_crop_size"]),
            int(cfg["context_crop_size"]),
            (
                f"visual={float(pair['visual_similarity']):.4f}  "
                f"motion={float(pair['motion_distance']):.4f}  "
                f"{design}"
            ),
        )
        review_paths.append(review_path)
        pair_rows.append(
            {
                "pair_rank": str(pair_index),
                "planned_pair_id": planned_pair_id,
                "record_a": left["record_id"],
                "record_b": right["record_id"],
                "image_a": left_sample["image_name"],
                "image_b": right_sample["image_name"],
                "detail_similarity": f"{float(detail_similarity[left_index, right_index]):.9f}",
                "context_similarity": f"{float(context_similarity[left_index, right_index]):.9f}",
                "visual_similarity": f"{float(pair['visual_similarity']):.9f}",
                "motion_distance": f"{float(pair['motion_distance']):.9f}",
                "reciprocal_visual_rank": str(pair["reciprocal_rank"]),
                "pair_score": f"{float(pair['pair_score']):.9f}",
                "pair_design": design,
                "selection_status": "proposed_visual_review_required",
                "manual_same_object_region_approved": "0",
                "review_image_path": str(review_path),
                "notes": "",
            }
        )
        for variant, index, row, paired in (
            ("A", left_index, left, right),
            ("B", right_index, right, left),
        ):
            selection = {
                "selection_rank": str(len(selection_rows) + 1),
                "planned_pair_id": planned_pair_id,
                "pair_variant": variant,
                "pair_design": design,
                "source_pool": row["source_pool"],
                "split": row["split"],
                "record_id": row["record_id"],
                "image_name": row["image_name"],
                "robust_motion_profile": row["robust_motion_profile"],
                "track_quality_passed": row["track_quality_passed"],
                **{field: row[field] for field in MOTION_FIELDS},
                **{
                    output_name: str(int(values[index]))
                    for output_name, values in assignments.items()
                },
                "paired_record_id": paired["record_id"],
                "visual_similarity": f"{float(pair['visual_similarity']):.9f}",
                "motion_distance": f"{float(pair['motion_distance']):.9f}",
                "selection_status": "proposed_visual_review_required",
                "manual_same_object_region_approved": "0",
            }
            selection_rows.append(selection)
            for probe in sorted(probes):
                sample = dict(samples[(row["split"], row["record_id"], probe)])
                sample.update(
                    {
                        "planned_pair_id": planned_pair_id,
                        "pair_variant": variant,
                        "pair_design": design,
                        "selection_status": (
                            "proposed_visual_review_required"
                        ),
                        "manual_same_object_region_approved": "0",
                    }
                )
                selected_sample_rows.append(sample)
    review_contact_sheet = project_path(cfg["review_contact_sheet"])
    save_review_contact_sheet(review_paths, review_contact_sheet)
    write_csv_rows(project_path(cfg["pair_output_csv"]), pair_rows, PAIR_FIELDS)
    write_csv_rows(
        project_path(cfg["selection_output_csv"]),
        selection_rows,
        SELECTION_FIELDS,
    )
    sample_extra_fields = [
        "planned_pair_id",
        "pair_variant",
        "pair_design",
        "selection_status",
        "manual_same_object_region_approved",
    ]
    sample_fields = list(sample_rows[0].keys()) + [
        field for field in sample_extra_fields if field not in sample_rows[0]
    ]
    write_csv_rows(
        project_path(cfg["selected_samples_csv"]),
        selected_sample_rows,
        sample_fields,
    )
    selected_counts = {
        name: dict(
            Counter(
                int(selection[name])
                for selection in selection_rows
            )
        )
        for name in BIN_FIELDS
    }
    summary = {
        "mode": "phase4i6e_visual_motion_pair_selection_v1",
        "robust_input_records": len(robust),
        "source_pool_counts": dict(
            Counter(row["source_pool"] for row in robust)
        ),
        "selected_records": len(selection_rows),
        "selected_pairs": len(pair_rows),
        "selected_far_queries": len(selected_sample_rows),
        "selected_reciprocal_top_k": selected_top_k,
        "descriptor_fingerprint": descriptor_fingerprint,
        "pair_design_counts": dict(
            Counter(row["pair_design"] for row in pair_rows)
        ),
        "selected_quantile_bin_counts": selected_counts,
        "quantile_edges": bin_edges,
        "visual_similarity": {
            "minimum": min(float(row["visual_similarity"]) for row in pair_rows),
            "median": float(
                np.median(
                    [float(row["visual_similarity"]) for row in pair_rows]
                )
            ),
            "maximum": max(float(row["visual_similarity"]) for row in pair_rows),
        },
        "motion_distance": {
            "minimum": min(float(row["motion_distance"]) for row in pair_rows),
            "median": float(
                np.median(
                    [float(row["motion_distance"]) for row in pair_rows]
                )
            ),
            "maximum": max(float(row["motion_distance"]) for row in pair_rows),
        },
        "manual_review": {
            "required": True,
            "approved_pairs": 0,
            "reason": (
                "DINO visual similarity is a proposal signal and cannot prove "
                "same-object/same-contact-region identity"
            ),
        },
        "integrity": {
            "source": "frozen Phase4I.6C/6D robust development pools only",
            "query_tactile_input": False,
            "tactile_labels_read": 0,
            "model_retrained": False,
            "threshold_retuned": False,
            "contact_descriptor_usage": (
                "offline data selection only; target contact coordinates are "
                "not an online model input"
            ),
            "sealed_final_holdout_rows_read": 0,
            "development_validation_outcomes_read": 0,
        },
        "outputs": {
            "pair_csv": str(project_path(cfg["pair_output_csv"])),
            "selection_csv": str(project_path(cfg["selection_output_csv"])),
            "selected_samples_csv": str(
                project_path(cfg["selected_samples_csv"])
            ),
            "review_dir": str(review_dir),
            "review_contact_sheet": str(review_contact_sheet),
        },
        "next_action": (
            "review all 30 pair images and approve only true same-object/"
            "same-contact-region pairs before building the TTC training split"
        ),
    }
    write_json(project_path(cfg["metrics_json"]), summary)
    print(summary)
    for row in pair_rows[: int(cfg["print_pair_limit"])]:
        print(
            row["planned_pair_id"],
            row["record_a"],
            row["record_b"],
            "visual=",
            row["visual_similarity"],
            "motion=",
            row["motion_distance"],
            "rank=",
            row["reciprocal_visual_rank"],
            "design=",
            row["pair_design"],
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select Phase4I.6E visual-motion far record pairs."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--section",
        default="phase4i6e_visual_motion_pair_selection_v1",
    )
    args = parser.parse_args()
    select(args.config, args.section)


if __name__ == "__main__":
    main()
