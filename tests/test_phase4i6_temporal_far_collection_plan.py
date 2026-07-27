from __future__ import annotations

import unittest

from src.plan_phase4i6_temporal_far_collection import (
    build_collection_plan,
    build_reference_rows,
    classify_reference,
    validate_collection_plan,
)


def audit_row(
    name: str,
    record: str,
    probe: int,
    changed_harm: bool = False,
    high_confidence: bool = False,
    predicted_ttc: float = 20.0,
) -> dict[str, str]:
    return {
        "query_record_id": record,
        "query_image_name": name,
        "query_probe": str(probe),
        "oof_fold": "0",
        "phase4i3_changed_cache": str(int(changed_harm)),
        "retrieval_outcome": (
            "spatial_only_gain" if changed_harm else "identity_unchanged"
        ),
        "mae_or_ssim_harm": str(int(changed_harm)),
        "high_confidence_false_negative": str(int(high_confidence)),
        "far_miss_margin": "0.2",
        "true_ttc": str(probe),
        "temporal_predicted_ttc": str(predicted_ttc),
        "temporal_ttc_absolute_error": str(abs(probe - predicted_ttc)),
        "base_predicted_ttc": "25",
        "ttc_entropy": "0.8",
        "trajectory_stability": "0.5",
        "visual_first_last_cosine_distance": "0.03",
        "corridor_approach_reduction": "0.1",
    }


def config() -> dict:
    return {
        "severe_ttc_underestimate_frames": 20.0,
        "priority_weights": {
            "changed_harm": 100.0,
            "high_confidence_false_negative": 20.0,
            "severe_ttc_underestimate": 10.0,
            "underestimate_per_frame": 0.1,
            "probe100": 2.0,
        },
        "target_reference_records": 2,
        "target_new_records": 6,
        "target_new_far_queries": 12,
        "probe_focus": [75, 100],
        "shared_motion_profiles": [
            "straight_constant_velocity",
            "straight_accelerating",
            "turning_constant_speed",
        ],
        "pair_designs": [
            {
                "name": "same_region_different_speed",
                "variant_a": "slow_approach",
                "variant_b": "fast_approach",
            },
            {
                "name": "same_speed_different_start_distance",
                "variant_a": "short_remaining_distance",
                "variant_b": "long_remaining_distance",
            },
        ],
        "minimum_real_point_count": 32,
        "minimum_history_span": 31,
        "maximum_padding_ratio": 0,
        "maximum_frame_gap": 1,
    }


class Phase4I6TemporalFarCollectionPlanTests(unittest.TestCase):
    def test_changed_harm_is_highest_priority_reference(self) -> None:
        cfg = config()
        changed = classify_reference(
            audit_row("0_rec_a_probe100.jpg", "rec_a", 100, True),
            cfg,
        )
        ordinary = classify_reference(
            audit_row("0_rec_b_probe075.jpg", "rec_b", 75),
            cfg,
        )
        self.assertGreater(
            float(changed["priority_score"]),
            float(ordinary["priority_score"]),
        )
        self.assertIn(
            "changed_cache_mae_or_ssim_harm",
            changed["case_types"],
        )

    def test_reference_selection_is_record_disjoint(self) -> None:
        cfg = config()
        rows = [
            audit_row(
                "0_rec_a_probe100.jpg",
                "rec_a",
                100,
                changed_harm=True,
            ),
            audit_row("0_rec_a_probe075.jpg", "rec_a", 75),
            audit_row(
                "0_rec_b_probe100.jpg",
                "rec_b",
                100,
                high_confidence=True,
            ),
            audit_row("0_rec_c_probe075.jpg", "rec_c", 75),
        ]
        references = build_reference_rows(rows, cfg)
        selected = [
            row
            for row in references
            if row["selected_as_collection_reference"] == "1"
        ]
        self.assertEqual(len(selected), 2)
        self.assertEqual(
            len({row["query_record_id"] for row in selected}),
            2,
        )
        self.assertEqual(selected[0]["query_record_id"], "rec_a")

    def test_collection_plan_is_paired_and_balanced(self) -> None:
        cfg = config()
        references = build_reference_rows(
            [
                audit_row(
                    "0_rec_a_probe100.jpg",
                    "rec_a",
                    100,
                    changed_harm=True,
                ),
                audit_row(
                    "0_rec_b_probe075.jpg",
                    "rec_b",
                    75,
                    high_confidence=True,
                ),
            ],
            cfg,
        )
        plan = build_collection_plan(references, cfg)
        self.assertEqual(len(plan), 6)
        self.assertEqual(plan[0]["planned_pair_id"], plan[1]["planned_pair_id"])
        self.assertEqual(plan[0]["pair_variant"], "A")
        self.assertEqual(plan[1]["pair_variant"], "B")
        self.assertEqual(plan[0]["new_record_id"], "")
        self.assertEqual(plan[0]["record_disjoint_required"], "1")
        self.assertEqual(plan[0]["probe_focus"], "75|100")
        self.assertEqual(
            {row["shared_motion_profile"] for row in plan},
            {
                "straight_constant_velocity",
                "straight_accelerating",
                "turning_constant_speed",
            },
        )
        self.assertEqual(
            sum(int(row["expected_far_queries"]) for row in plan),
            12,
        )

    def test_collection_plan_rejects_pair_profile_mismatch(self) -> None:
        cfg = config()
        references = build_reference_rows(
            [
                audit_row("0_rec_a_probe100.jpg", "rec_a", 100),
                audit_row("0_rec_b_probe075.jpg", "rec_b", 75),
            ],
            cfg,
        )
        plan = build_collection_plan(references, cfg)
        plan[1]["shared_motion_profile"] = "turning_constant_speed"
        with self.assertRaisesRegex(RuntimeError, "shared_motion_profile"):
            validate_collection_plan(plan)

    def test_collection_plan_requires_even_record_count(self) -> None:
        cfg = config()
        cfg["target_new_records"] = 5
        references = build_reference_rows(
            [
                audit_row("0_rec_a_probe100.jpg", "rec_a", 100),
                audit_row("0_rec_b_probe075.jpg", "rec_b", 75),
            ],
            cfg,
        )
        with self.assertRaisesRegex(RuntimeError, "must be even"):
            build_collection_plan(references, cfg)


if __name__ == "__main__":
    unittest.main()
