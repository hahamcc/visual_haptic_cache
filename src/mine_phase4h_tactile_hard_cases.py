"""Mine record-disjoint tactile-cache hard cases after a failed Phase4H OOF run."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

from .config import load_config, project_path
from .phase4h_dino_adaptation import (
    assert_candidate_identity,
    assert_development_only,
    candidate_groups,
)
from .utils import ensure_dir, read_csv_rows, write_csv_rows, write_json


QUERY_FIELDS = [
    "query_record_id",
    "query_image_name",
    "query_probe",
    "oof_fold",
    "primary_recipe",
    "v1_selected_cache_record_id",
    "v1_selected_cache_image_name",
    "frozen_dino_selected_cache_record_id",
    "frozen_dino_selected_cache_image_name",
    "aligned_dino_selected_cache_record_id",
    "aligned_dino_selected_cache_image_name",
    "v1_mae",
    "frozen_dino_mae",
    "aligned_dino_mae",
    "frozen_mae_delta_vs_v1",
    "aligned_mae_delta_vs_v1",
    "v1_ssim",
    "frozen_dino_ssim",
    "aligned_dino_ssim",
    "frozen_ssim_delta_vs_v1",
    "aligned_ssim_delta_vs_v1",
    "v1_iou",
    "frozen_dino_iou",
    "aligned_dino_iou",
    "frozen_iou_delta_vs_v1",
    "aligned_iou_delta_vs_v1",
    "frozen_oracle_rank",
    "aligned_oracle_rank",
    "dino_top8_oracle17_32_count",
    "unique_hard_negative_cache_records",
    "frozen_strict_triple_win",
    "spatial_gain_mae_harm",
    "aligned_projector_harm",
    "far_failure",
    "priority_score",
    "case_types",
    "acquisition_reason",
]

PAIR_FIELDS = [
    "query_record_id",
    "query_image_name",
    "query_probe",
    "oof_fold",
    "primary_recipe",
    "candidate_record_id",
    "candidate_image_name",
    "dino_candidate_rank",
    "candidate_oracle_embedding_rank",
    "candidate_score",
    "detail_patch_score",
    "context_patch_score",
    "wide_patch_score",
    "candidate_tactile_embedding_distance",
    "candidate_tactile_ssim",
    "candidate_tactile_mask_iou",
    "normalized_contact_offset",
    "pair_priority_score",
    "hard_negative_type",
]

RECORD_FIELDS = [
    "reference_record_id",
    "queries",
    "probes",
    "far_queries",
    "hard_negative_pairs",
    "unique_hard_negative_cache_records",
    "spatial_gain_mae_harm_queries",
    "aligned_projector_harm_queries",
    "strict_triple_win_queries",
    "priority_score",
    "selected_as_collection_reference",
    "reference_query_image_names",
    "acquisition_recipe",
]

COLLECTION_FIELDS = [
    "collection_slot",
    "new_record_id",
    "reference_record_id",
    "reference_query_image_names",
    "probe_focus",
    "minimum_contact_regions",
    "repeats_per_region",
    "motion_conditions",
    "intensity_conditions",
    "hard_negative_pair_goal",
    "record_disjoint_required",
    "acquisition_reason",
]


def load_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def finite(row: dict[str, str], key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"Non-finite {key} for {row.get('query_image_name', '<unknown>')}")
    return value


def unique_by_query(rows: list[dict[str, str]], label: str) -> dict[str, dict[str, str]]:
    output: dict[str, dict[str, str]] = {}
    for row in rows:
        name = row["query_image_name"]
        if name in output:
            raise RuntimeError(f"Duplicate {label} query row: {name}")
        output[name] = row
    return output


def require_same_queries(*named: tuple[str, dict[str, dict[str, str]]]) -> list[str]:
    reference_label, reference = named[0]
    expected = set(reference)
    for label, rows in named[1:]:
        if set(rows) != expected:
            missing = sorted(expected - set(rows))[:3]
            extra = sorted(set(rows) - expected)[:3]
            raise RuntimeError(
                f"{label} queries differ from {reference_label}: missing={missing}, extra={extra}"
            )
    return sorted(expected)


def is_strict_triple_win(base: dict[str, str], current: dict[str, str]) -> bool:
    return bool(
        finite(current, "tactile_diff_mae") < finite(base, "tactile_diff_mae")
        and finite(current, "tactile_ssim") >= finite(base, "tactile_ssim")
        and finite(current, "tactile_mask_iou") >= finite(base, "tactile_mask_iou")
    )


def contact_offset(
    query: dict[str, str],
    candidate: dict[str, str],
) -> float:
    qx = float(query["target_tip_x"]) / max(float(query["image_width"]), 1.0)
    qy = float(query["target_tip_y"]) / max(float(query["image_height"]), 1.0)
    cx = float(candidate["target_tip_x"]) / max(float(candidate["image_width"]), 1.0)
    cy = float(candidate["target_tip_y"]) / max(float(candidate["image_height"]), 1.0)
    return float(math.hypot(qx - cx, qy - cy))


def mine_pairs_for_query(
    query_name: str,
    candidates: list[dict[str, str]],
    samples_by_name: dict[str, dict[str, str]],
    max_pairs: int,
    far_probe_min: int,
) -> list[dict[str, str]]:
    hard = [
        row
        for row in candidates
        if int(row["candidate_rank"]) <= 8
        and int(row["candidate_oracle_embedding_rank"]) >= 17
    ]
    hard.sort(
        key=lambda row: (
            -int(row["candidate_oracle_embedding_rank"]),
            int(row["candidate_rank"]),
            row["candidate_image_name"],
        )
    )
    query = samples_by_name[query_name]
    output = []
    for row in hard[:max_pairs]:
        candidate = samples_by_name[row["candidate_image_name"]]
        dino_rank = int(row["candidate_rank"])
        oracle_rank = int(row["candidate_oracle_embedding_rank"])
        far_bonus = 5.0 if int(row["query_probe"]) >= far_probe_min else 0.0
        output.append(
            {
                "query_record_id": row["query_record_id"],
                "query_image_name": query_name,
                "query_probe": row["query_probe"],
                "oof_fold": row["oof_fold"],
                "primary_recipe": row["recipe_name"],
                "candidate_record_id": row["candidate_record_id"],
                "candidate_image_name": row["candidate_image_name"],
                "dino_candidate_rank": str(dino_rank),
                "candidate_oracle_embedding_rank": str(oracle_rank),
                "candidate_score": row["candidate_score"],
                "detail_patch_score": row.get("detail_patch_score", ""),
                "context_patch_score": row.get("context_patch_score", ""),
                "wide_patch_score": row.get("wide_patch_score", ""),
                "candidate_tactile_embedding_distance": row[
                    "candidate_tactile_embedding_distance"
                ],
                "candidate_tactile_ssim": row.get("candidate_tactile_ssim", ""),
                "candidate_tactile_mask_iou": row.get("candidate_tactile_mask_iou", ""),
                "normalized_contact_offset": f"{contact_offset(query, candidate):.6f}",
                "pair_priority_score": f"{oracle_rank - dino_rank + far_bonus:.6f}",
                "hard_negative_type": "dino_top8_tactile_oracle17_32",
            }
        )
    return output


def classify_query(
    v1: dict[str, str],
    frozen: dict[str, str],
    aligned: dict[str, str],
    hard_pairs: list[dict[str, str]],
    cfg: dict,
) -> dict[str, str]:
    frozen_mae_delta = finite(frozen, "tactile_diff_mae") - finite(v1, "tactile_diff_mae")
    aligned_mae_delta = finite(aligned, "tactile_diff_mae") - finite(v1, "tactile_diff_mae")
    frozen_ssim_delta = finite(frozen, "tactile_ssim") - finite(v1, "tactile_ssim")
    aligned_ssim_delta = finite(aligned, "tactile_ssim") - finite(v1, "tactile_ssim")
    frozen_iou_delta = finite(frozen, "tactile_mask_iou") - finite(v1, "tactile_mask_iou")
    aligned_iou_delta = finite(aligned, "tactile_mask_iou") - finite(v1, "tactile_mask_iou")
    epsilon = float(cfg["mae_harm_epsilon"])
    spatial_gain_mae_harm = bool(
        frozen_iou_delta >= float(cfg["minimum_spatial_iou_gain"])
        and frozen_mae_delta > epsilon
    )
    aligned_projector_harm = bool(
        finite(aligned, "tactile_diff_mae")
        > finite(frozen, "tactile_diff_mae") + epsilon
        or finite(aligned, "tactile_mask_iou")
        < finite(frozen, "tactile_mask_iou") - float(cfg["minimum_aligned_iou_drop"])
    )
    far_failure = bool(
        int(v1["query_probe"]) >= int(cfg["far_probe_min"])
        and (
            aligned_mae_delta > epsilon
            or aligned_ssim_delta < -float(cfg["minimum_far_ssim_drop"])
        )
    )
    triple = is_strict_triple_win(v1, frozen)
    weights = cfg["priority_weights"]
    priority = (
        min(len(hard_pairs), int(cfg["max_hard_pairs_per_query"]))
        * float(weights["hard_negative_pair"])
        + float(weights["spatial_gain_mae_harm"]) * int(spatial_gain_mae_harm)
        + float(weights["aligned_projector_harm"]) * int(aligned_projector_harm)
        + float(weights["far_failure"]) * int(far_failure)
        + float(weights["triple_win_reference"]) * int(triple)
    )
    case_types = []
    if hard_pairs:
        case_types.append("visual_tactile_hard_negative")
    if spatial_gain_mae_harm:
        case_types.append("spatial_gain_intensity_harm")
    if aligned_projector_harm:
        case_types.append("aligned_projector_overfit_harm")
    if far_failure:
        case_types.append("far_failure")
    if triple:
        case_types.append("frozen_dino_strict_triple_win_reference")
    reasons = {
        "visual_tactile_hard_negative": "capture visually similar regions with deliberately different tactile response",
        "spatial_gain_intensity_harm": "repeat the same region under low, medium, and high deformation intensity",
        "aligned_projector_overfit_harm": "add a new record-disjoint repetition of this visual-contact configuration",
        "far_failure": "repeat with probe75/100 and varied motion speed or direction",
        "frozen_dino_strict_triple_win_reference": "retain as a positive spatial-correspondence reference",
    }
    cache_records = sorted({row["candidate_record_id"] for row in hard_pairs})
    return {
        "query_record_id": v1["query_record_id"],
        "query_image_name": v1["query_image_name"],
        "query_probe": v1["query_probe"],
        "oof_fold": v1["oof_fold"],
        "primary_recipe": frozen["recipe_name"],
        "v1_selected_cache_record_id": v1["selected_cache_record_id"],
        "v1_selected_cache_image_name": v1["selected_cache_image_name"],
        "frozen_dino_selected_cache_record_id": frozen["selected_cache_record_id"],
        "frozen_dino_selected_cache_image_name": frozen["selected_cache_image_name"],
        "aligned_dino_selected_cache_record_id": aligned["selected_cache_record_id"],
        "aligned_dino_selected_cache_image_name": aligned["selected_cache_image_name"],
        "v1_mae": v1["tactile_diff_mae"],
        "frozen_dino_mae": frozen["tactile_diff_mae"],
        "aligned_dino_mae": aligned["tactile_diff_mae"],
        "frozen_mae_delta_vs_v1": f"{frozen_mae_delta:.9f}",
        "aligned_mae_delta_vs_v1": f"{aligned_mae_delta:.9f}",
        "v1_ssim": v1["tactile_ssim"],
        "frozen_dino_ssim": frozen["tactile_ssim"],
        "aligned_dino_ssim": aligned["tactile_ssim"],
        "frozen_ssim_delta_vs_v1": f"{frozen_ssim_delta:.9f}",
        "aligned_ssim_delta_vs_v1": f"{aligned_ssim_delta:.9f}",
        "v1_iou": v1["tactile_mask_iou"],
        "frozen_dino_iou": frozen["tactile_mask_iou"],
        "aligned_dino_iou": aligned["tactile_mask_iou"],
        "frozen_iou_delta_vs_v1": f"{frozen_iou_delta:.9f}",
        "aligned_iou_delta_vs_v1": f"{aligned_iou_delta:.9f}",
        "frozen_oracle_rank": frozen["ranker_oracle_embedding_rank"],
        "aligned_oracle_rank": aligned["ranker_oracle_embedding_rank"],
        "dino_top8_oracle17_32_count": str(len(hard_pairs)),
        "unique_hard_negative_cache_records": str(len(cache_records)),
        "frozen_strict_triple_win": str(int(triple)),
        "spatial_gain_mae_harm": str(int(spatial_gain_mae_harm)),
        "aligned_projector_harm": str(int(aligned_projector_harm)),
        "far_failure": str(int(far_failure)),
        "priority_score": f"{priority:.6f}",
        "case_types": "|".join(case_types) if case_types else "non_priority",
        "acquisition_reason": "; ".join(reasons[name] for name in case_types),
    }


def aggregate_records(
    query_rows: list[dict[str, str]],
    pair_rows: list[dict[str, str]],
    target_reference_records: int,
) -> list[dict[str, str]]:
    queries: dict[str, list[dict[str, str]]] = defaultdict(list)
    pairs: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in query_rows:
        queries[row["query_record_id"]].append(row)
    for row in pair_rows:
        pairs[row["query_record_id"]].append(row)
    summaries = []
    for record_id, rows in queries.items():
        record_pairs = pairs.get(record_id, [])
        ranked = sorted(rows, key=lambda row: -float(row["priority_score"]))
        priority = sum(float(row["priority_score"]) for row in rows)
        cache_records = {row["candidate_record_id"] for row in record_pairs}
        reasons = Counter(
            case_type
            for row in rows
            for case_type in row["case_types"].split("|")
            if case_type != "non_priority"
        )
        summaries.append(
            {
                "reference_record_id": record_id,
                "queries": str(len(rows)),
                "probes": "|".join(
                    str(value)
                    for value in sorted({int(row["query_probe"]) for row in rows})
                ),
                "far_queries": str(sum(int(row["far_failure"]) for row in rows)),
                "hard_negative_pairs": str(len(record_pairs)),
                "unique_hard_negative_cache_records": str(len(cache_records)),
                "spatial_gain_mae_harm_queries": str(
                    sum(int(row["spatial_gain_mae_harm"]) for row in rows)
                ),
                "aligned_projector_harm_queries": str(
                    sum(int(row["aligned_projector_harm"]) for row in rows)
                ),
                "strict_triple_win_queries": str(
                    sum(int(row["frozen_strict_triple_win"]) for row in rows)
                ),
                "priority_score": f"{priority:.6f}",
                "selected_as_collection_reference": "0",
                "reference_query_image_names": "|".join(
                    row["query_image_name"] for row in ranked[:3]
                ),
                "acquisition_recipe": "|".join(
                    name for name, _ in reasons.most_common()
                ),
            }
        )
    summaries.sort(
        key=lambda row: (
            -float(row["priority_score"]),
            -int(row["hard_negative_pairs"]),
            row["reference_record_id"],
        )
    )
    for row in summaries[:target_reference_records]:
        row["selected_as_collection_reference"] = "1"
    return summaries


def build_collection_plan(
    record_rows: list[dict[str, str]],
    cfg: dict,
) -> list[dict[str, str]]:
    references = [
        row for row in record_rows if row["selected_as_collection_reference"] == "1"
    ]
    if not references:
        raise RuntimeError("No Phase4H hard-case reference records were selected")
    output = []
    for index in range(int(cfg["target_new_records"])):
        reference = references[index % len(references)]
        probes = [int(value) for value in reference["probes"].split("|")]
        probe_focus = (
            "75|100"
            if int(reference["far_queries"]) > 0
            else "|".join(str(value) for value in probes)
        )
        output.append(
            {
                "collection_slot": f"{index + 1:03d}",
                "new_record_id": "",
                "reference_record_id": reference["reference_record_id"],
                "reference_query_image_names": reference["reference_query_image_names"],
                "probe_focus": probe_focus,
                "minimum_contact_regions": str(cfg["minimum_contact_regions"]),
                "repeats_per_region": str(cfg["repeats_per_region"]),
                "motion_conditions": "straight|turning|accelerating_or_decelerating",
                "intensity_conditions": "low|medium|high",
                "hard_negative_pair_goal": str(cfg["hard_negative_pairs_per_new_record"]),
                "record_disjoint_required": "1",
                "acquisition_reason": reference["acquisition_recipe"],
            }
        )
    return output


def panel_image(path: str, size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB").resize(size, Image.Resampling.BILINEAR)


def draw_contact_cross(
    image: Image.Image,
    x: float,
    y: float,
    source_width: float,
    source_height: float,
) -> None:
    draw = ImageDraw.Draw(image)
    px = x / max(source_width, 1.0) * image.width
    py = y / max(source_height, 1.0) * image.height
    draw.line((px - 7, py, px + 7, py), fill="red", width=2)
    draw.line((px, py - 7, px, py + 7), fill="red", width=2)


def save_debug_sheet(
    query_row: dict[str, str],
    v1: dict[str, str],
    frozen: dict[str, str],
    aligned: dict[str, str],
    samples_by_name: dict[str, dict[str, str]],
    output_path: Path,
) -> None:
    selections = [
        ("query", query_row["query_image_name"]),
        ("V1", v1["selected_cache_image_name"]),
        ("frozen DINO", frozen["selected_cache_image_name"]),
        ("aligned DINO", aligned["selected_cache_image_name"]),
    ]
    width, vision_height, touch_height, header = 192, 128, 128, 26
    canvas = Image.new("RGB", (width * len(selections), header + vision_height + touch_height), "white")
    draw = ImageDraw.Draw(canvas)
    for column, (label, image_name) in enumerate(selections):
        sample = samples_by_name[image_name]
        vision = panel_image(sample["vision_path"], (width, vision_height))
        if label == "query":
            x, y = float(v1["pred_x"]), float(v1["pred_y"])
        else:
            x, y = float(sample["target_tip_x"]), float(sample["target_tip_y"])
        draw_contact_cross(
            vision,
            x,
            y,
            float(sample["image_width"]),
            float(sample["image_height"]),
        )
        touch = panel_image(sample["touch_path"], (width, touch_height))
        left = column * width
        canvas.paste(vision, (left, header))
        canvas.paste(touch, (left, header + vision_height))
        draw.text((left + 5, 6), label, fill="black")
    ensure_dir(output_path.parent)
    canvas.save(output_path)


def count_flags(rows: list[dict[str, str]]) -> dict[str, int]:
    return {
        key: sum(int(row[key]) for row in rows)
        for key in (
            "frozen_strict_triple_win",
            "spatial_gain_mae_harm",
            "aligned_projector_harm",
            "far_failure",
        )
    }


def mine(config_path: str, section: str) -> dict:
    cfg = load_config(config_path)[section]
    samples = read_csv_rows(project_path(cfg["samples_csv"]))
    assert_development_only(samples, project_path(cfg["final_partition_csv"]))
    development = [row for row in samples if row["dataset_split"] == "train"]
    samples_by_name = {row["image_name"]: row for row in development}
    if len(samples_by_name) != len(development):
        raise RuntimeError("Phase4H hard-case mining requires unique development image names")

    oof_report = load_json(project_path(cfg["oof_evaluation_json"]))
    if bool(cfg.get("require_failed_oof", True)) and oof_report.get(
        "ready_for_development_validation", False
    ):
        raise RuntimeError("Phase4H OOF passed; failure-branch mining is not applicable")
    frontier = load_json(project_path(cfg["frontier_json"]))
    primary_recipe = str(frontier["primary_recipe"])

    v1 = unique_by_query(
        read_csv_rows(project_path(cfg["v1_query_csv"])), "V1"
    )
    frozen = unique_by_query(
        [
            row
            for row in read_csv_rows(project_path(cfg["ablation_query_csv"]))
            if row["recipe_name"] == primary_recipe
        ],
        "frozen DINO",
    )
    aligned = unique_by_query(
        read_csv_rows(project_path(cfg["aligned_query_csv"])), "aligned DINO"
    )
    names = require_same_queries(("V1", v1), ("frozen DINO", frozen), ("aligned DINO", aligned))
    if set(names) != set(samples_by_name):
        raise RuntimeError("Phase4H hard-case query set does not exactly match development-train")

    top_k = int(cfg["geometry_filter_k"])
    v1_candidate_groups = candidate_groups(
        read_csv_rows(project_path(cfg["v1_candidate_csv"])), top_k
    )
    frozen_candidate_groups = candidate_groups(
        [
            row
            for row in read_csv_rows(project_path(cfg["ablation_candidate_csv"]))
            if row["recipe_name"] == primary_recipe
        ],
        top_k,
    )
    assert_candidate_identity(v1_candidate_groups, frozen_candidate_groups)

    all_pairs: list[dict[str, str]] = []
    pairs_by_query: dict[str, list[dict[str, str]]] = {}
    for name in names:
        pairs = mine_pairs_for_query(
            name,
            frozen_candidate_groups[name],
            samples_by_name,
            int(cfg["max_hard_pairs_per_query"]),
            int(cfg["far_probe_min"]),
        )
        pairs_by_query[name] = pairs
        all_pairs.extend(pairs)

    query_rows = [
        classify_query(v1[name], frozen[name], aligned[name], pairs_by_query[name], cfg)
        for name in names
    ]
    query_rows.sort(
        key=lambda row: (-float(row["priority_score"]), row["query_image_name"])
    )
    all_pairs.sort(
        key=lambda row: (-float(row["pair_priority_score"]), row["query_image_name"])
    )
    record_rows = aggregate_records(
        query_rows,
        all_pairs,
        int(cfg["target_reference_records"]),
    )
    collection_rows = build_collection_plan(record_rows, cfg)

    write_csv_rows(project_path(cfg["query_output_csv"]), query_rows, QUERY_FIELDS)
    write_csv_rows(project_path(cfg["pair_output_csv"]), all_pairs, PAIR_FIELDS)
    write_csv_rows(project_path(cfg["record_output_csv"]), record_rows, RECORD_FIELDS)
    write_csv_rows(
        project_path(cfg["collection_plan_csv"]),
        collection_rows,
        COLLECTION_FIELDS,
    )

    debug_dir = project_path(cfg["debug_dir"])
    debug_limit = int(cfg["maximum_debug_cases"])
    debug_written = 0
    for row in query_rows:
        if debug_written >= debug_limit:
            break
        if row["case_types"] == "non_priority":
            continue
        name = row["query_image_name"]
        save_debug_sheet(
            row,
            v1[name],
            frozen[name],
            aligned[name],
            samples_by_name,
            debug_dir / f"{debug_written + 1:03d}_{Path(name).stem}.jpg",
        )
        debug_written += 1

    selected_records = [
        row for row in record_rows if row["selected_as_collection_reference"] == "1"
    ]
    selected_record_ids = {row["reference_record_id"] for row in selected_records}
    selected_pairs = [
        row for row in all_pairs if row["query_record_id"] in selected_record_ids
    ]
    planned_new_pair_goal = (
        len(collection_rows) * int(cfg["hard_negative_pairs_per_new_record"])
    )
    if planned_new_pair_goal < int(cfg["target_new_hard_negative_pairs"]):
        raise RuntimeError(
            "Phase4H collection plan does not meet the configured hard-negative target: "
            f"{planned_new_pair_goal} < {int(cfg['target_new_hard_negative_pairs'])}"
        )
    summary = {
        "mode": "phase4h_failed_oof_tactile_hard_case_mining_v1",
        "primary_recipe": primary_recipe,
        "queries": len(query_rows),
        "hard_negative_pairs": len(all_pairs),
        "hard_negative_query_records": len(
            {row["query_record_id"] for row in all_pairs}
        ),
        "case_counts": count_flags(query_rows),
        "case_counts_by_probe": {
            str(probe): count_flags(
                [row for row in query_rows if int(row["query_probe"]) == probe]
            )
            for probe in sorted({int(row["query_probe"]) for row in query_rows})
        },
        "collection_contract": {
            "target_new_record_disjoint_records": int(cfg["target_new_records"]),
            "target_new_hard_negative_pairs": int(cfg["target_new_hard_negative_pairs"]),
            "reference_records_selected": len(selected_records),
            "reference_hard_negative_pairs": len(selected_pairs),
            "collection_slots_written": len(collection_rows),
            "planned_new_hard_negative_pair_goal": planned_new_pair_goal,
            "hard_negative_pair_target_satisfied_by_plan": True,
            "new_record_ids_must_not_reuse_reference_ids": True,
            "minimum_contact_regions": int(cfg["minimum_contact_regions"]),
            "repeats_per_region": int(cfg["repeats_per_region"]),
        },
        "diagnosis": {
            "capacity_change_allowed": False,
            "lora_allowed": False,
            "temporal_branch_allowed": False,
            "next_action": (
                "collect record-disjoint same-object/different-region and "
                "same-region/different-intensity examples using collection_plan_csv"
            ),
        },
        "outputs": {
            "query_csv": str(project_path(cfg["query_output_csv"])),
            "pair_csv": str(project_path(cfg["pair_output_csv"])),
            "record_csv": str(project_path(cfg["record_output_csv"])),
            "collection_plan_csv": str(project_path(cfg["collection_plan_csv"])),
            "debug_dir": str(debug_dir),
            "debug_sheets": debug_written,
        },
        "integrity": {
            "source": "strict Phase4H development OOF only",
            "same_top32_candidate_set": True,
            "query_tactile_usage": "offline hard-case labels and visualization only",
            "sealed_final_holdout_rows_read": 0,
            "development_validation_rows_read": 0,
        },
    }
    write_json(project_path(cfg["metrics_json"]), summary)
    print(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mine Phase4H tactile-cache hard cases and a record-disjoint collection plan."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--section", default="phase4h_tactile_hard_case_mining_v1"
    )
    args = parser.parse_args()
    mine(args.config, args.section)


if __name__ == "__main__":
    main()
