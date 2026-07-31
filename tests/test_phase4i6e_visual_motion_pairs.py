from __future__ import annotations

import unittest

import numpy as np

from src.select_phase4i6e_visual_motion_pairs import (
    greedy_pair_selection,
    merge_passed_records,
    pair_design,
    quantile_bin_assignments,
)


def robust_row(record_id: str, passed: str = "1") -> dict[str, str]:
    return {
        "split": "3",
        "record_id": record_id,
        "track_quality_passed": passed,
        "robust_mean_speed_px": "1",
        "robust_speed_cv": "1",
        "robust_normalized_speed_slope": "0",
        "robust_mean_acceleration_px": "1",
        "robust_direction_stability": "1",
        "robust_cumulative_turn_radians": "0",
        "robust_pause_ratio": "0",
        "robust_cumulative_displacement_px": "1",
    }


class Phase4I6eVisualMotionPairTests(unittest.TestCase):
    def test_merge_keeps_only_passed_and_rejects_overlap(self) -> None:
        merged = merge_passed_records(
            [
                ("a", [robust_row("rec_03000"), robust_row("rec_03001", "0")]),
                ("b", [robust_row("rec_03300")]),
            ]
        )
        self.assertEqual(
            [row["record_id"] for row in merged],
            ["rec_03000", "rec_03300"],
        )
        with self.assertRaisesRegex(RuntimeError, "overlap"):
            merge_passed_records(
                [
                    ("a", [robust_row("rec_03000")]),
                    ("b", [robust_row("rec_03000")]),
                ]
            )

    def test_quantile_bins_are_deterministic(self) -> None:
        rows = [robust_row(f"rec_{index:05d}") for index in range(8)]
        for index, row in enumerate(rows):
            row["robust_mean_speed_px"] = str(index)
        assignments, edges = quantile_bin_assignments(
            rows,
            {"speed_bin": "robust_mean_speed_px"},
            4,
        )
        self.assertEqual(assignments["speed_bin"].tolist(), [0, 0, 1, 1, 2, 2, 3, 3])
        self.assertEqual(len(edges["speed_bin"]), 3)

    def test_pair_selection_is_disjoint_and_visual_local(self) -> None:
        records = ["a", "b", "c", "d"]
        similarity = np.asarray(
            [
                [-1.0, 0.95, 0.10, 0.20],
                [0.95, -1.0, 0.20, 0.10],
                [0.10, 0.20, -1.0, 0.90],
                [0.20, 0.10, 0.90, -1.0],
            ],
            dtype=np.float32,
        )
        motion = np.asarray(
            [[0.0], [1.0], [0.0], [1.0]],
            dtype=np.float32,
        )
        assignments = {
            "speed_bin": np.asarray([0, 1, 0, 1], dtype=np.int64)
        }
        selected, top_k = greedy_pair_selection(
            records,
            similarity,
            motion,
            assignments,
            target_pairs=2,
            top_k_schedule=[1, 3],
            motion_weight=0.1,
            diversity_weight=0.1,
            quantile_bins=2,
            minimum_motion_distance=0.5,
        )
        endpoints = [
            int(pair[key])
            for pair in selected
            for key in ("left", "right")
        ]
        self.assertEqual(len(endpoints), len(set(endpoints)))
        self.assertEqual(
            {
                frozenset((records[int(pair["left"])], records[int(pair["right"])]))
                for pair in selected
            },
            {frozenset(("a", "b")), frozenset(("c", "d"))},
        )
        self.assertEqual(top_k, 1)

    def test_pair_selection_enforces_motion_contrast(self) -> None:
        records = ["a", "b", "c", "d"]
        similarity = np.asarray(
            [
                [-1.0, 0.99, 0.80, 0.10],
                [0.99, -1.0, 0.10, 0.80],
                [0.80, 0.10, -1.0, 0.99],
                [0.10, 0.80, 0.99, -1.0],
            ],
            dtype=np.float32,
        )
        motion = np.asarray([[0.0], [0.1], [1.0], [1.1]], dtype=np.float32)
        assignments = {
            "speed_bin": np.asarray([0, 0, 1, 1], dtype=np.int64)
        }
        selected, top_k = greedy_pair_selection(
            records,
            similarity,
            motion,
            assignments,
            target_pairs=2,
            top_k_schedule=[1, 3],
            motion_weight=0.1,
            diversity_weight=0.0,
            quantile_bins=2,
            minimum_motion_distance=0.5,
        )
        self.assertEqual(top_k, 3)
        self.assertTrue(
            all(float(pair["motion_distance"]) >= 0.5 for pair in selected)
        )

    def test_pair_design_uses_largest_motion_contrast(self) -> None:
        standardized = np.zeros((2, 8), dtype=np.float32)
        standardized[1, 0] = 4.0
        self.assertEqual(
            pair_design(0, 1, standardized),
            "same_region_different_speed",
        )


if __name__ == "__main__":
    unittest.main()
