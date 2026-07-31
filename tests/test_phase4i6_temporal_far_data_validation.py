from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.validate_phase4i6_temporal_far_data import (
    build_record_rows,
    evaluate_query,
    validate_completed_plan,
)


def plan_row(slot: str, variant: str, record_id: str) -> dict[str, str]:
    return {
        "collection_slot": slot,
        "new_record_id": record_id,
        "new_split": "2",
        "object_id": "object_a",
        "contact_region_id": "region_a",
        "planned_pair_id": "pair_001",
        "pair_variant": variant,
        "pair_design": "same_region_different_speed",
        "reference_record_id": "rec_reference",
        "reference_query_image_name": "reference.jpg",
        "probe_focus": "75|100",
        "shared_motion_profile": "straight_constant_velocity",
        "paired_variant": (
            "slow_approach" if variant == "A" else "fast_approach"
        ),
        "target_failure_mode": "true_far_predicted_as_near_or_mid",
        "minimum_real_point_count": "32",
        "minimum_history_span": "31",
        "maximum_padding_ratio": "0",
        "maximum_frame_gap": "1",
        "record_disjoint_required": "1",
        "reference_record_reuse_forbidden": "1",
        "same_object_region_pair_required": "1",
        "expected_far_queries": "2",
        "collection_status": "collected",
        "notes": "",
    }


class Phase4I6TemporalFarDataValidationTests(unittest.TestCase):
    def test_completed_plan_accepts_disjoint_pair(self) -> None:
        rows = [
            plan_row("001", "A", "rec_new_a"),
            plan_row("002", "B", "rec_new_b"),
        ]
        validate_completed_plan(
            rows,
            {("0", "rec_old")},
            {"rec_old"},
            {("2", "rec_final")},
            2,
        )

    def test_completed_plan_rejects_pair_region_mismatch(self) -> None:
        rows = [
            plan_row("001", "A", "rec_new_a"),
            plan_row("002", "B", "rec_new_b"),
        ]
        rows[1]["contact_region_id"] = "region_b"
        with self.assertRaisesRegex(RuntimeError, "contact_region_id"):
            validate_completed_plan(rows, set(), set(), set(), 2)

    def test_completed_plan_rejects_development_record_id(self) -> None:
        rows = [
            plan_row("001", "A", "rec_old"),
            plan_row("002", "B", "rec_new_b"),
        ]
        with self.assertRaisesRegex(RuntimeError, "development record IDs"):
            validate_completed_plan(
                rows,
                {("0", "rec_old")},
                {"rec_old"},
                set(),
                2,
            )

    def test_query_quality_uses_exact_32_frame_track(self) -> None:
        plan = plan_row("001", "A", "rec_new_a")
        with tempfile.TemporaryDirectory() as directory:
            vision = Path(directory) / "vision.jpg"
            touch = Path(directory) / "touch.jpg"
            vision.write_bytes(b"x")
            touch.write_bytes(b"x")
            sample = {
                "split": "2",
                "record_id": "rec_new_a",
                "probe": "75",
                "image_name": "query.jpg",
                "frame_id": "100",
                "contact_frame_detected": "175",
                "sequence_ready": "1",
                "vision_path": str(vision),
                "touch_path": str(touch),
                "image_width": "768",
                "image_height": "512",
            }
            tracks = {
                ("2", "rec_new_a"): [
                    {
                        "frame_id": frame,
                        "tip_x": float(frame),
                        "tip_y": 10.0,
                        "base_x": float(frame),
                        "base_y": 20.0,
                    }
                    for frame in range(69, 101)
                ]
            }
            row = evaluate_query(
                plan,
                sample,
                tracks,
                {
                    "history_frames": 32,
                    "spatial_scale_px": 48.0,
                    "speed_scale_px": 4.0,
                    "minimum_real_point_count": 32,
                    "minimum_history_span": 31,
                    "maximum_padding_ratio": 0.0,
                    "maximum_frame_gap": 1,
                    "require_paths_exist": True,
                },
            )
        self.assertEqual(row["passed"], "1")
        self.assertEqual(row["real_point_count"], "32.000000")
        self.assertEqual(row["history_span"], "31.000000")

    def test_record_requires_both_far_probes(self) -> None:
        plan = plan_row("001", "A", "rec_new_a")
        query = {
            "collection_slot": "001",
            "probe": "75",
            "passed": "1",
        }
        records = build_record_rows([plan], [query], {75, 100})
        self.assertEqual(records[0]["record_passed"], "0")
        self.assertIn(
            "required_probe_set_incomplete",
            records[0]["failure_reasons"],
        )


if __name__ == "__main__":
    unittest.main()
