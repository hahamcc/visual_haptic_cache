from __future__ import annotations

import unittest

import numpy as np

from src.select_phase4i3_monotonic_ttc_scales_oof import (
    RECORD_FIELDS,
    SLICE_FIELDS,
    SCALE_FIELDS,
    apply_group_scales,
    choose_scale,
    delta_summary,
    monotonic_scale_grid,
    phase4i2_diagnostics,
    scale_csv_row,
    scale_is_safe,
)


def retrieval_row(
    name: str,
    mae: float,
    ssim: float,
    iou: float,
    rank: int,
) -> dict[str, str]:
    return {
        "query_image_name": name,
        "tactile_diff_mae": str(mae),
        "tactile_ssim": str(ssim),
        "tactile_mask_iou": str(iou),
        "ranker_oracle_embedding_rank": str(rank),
    }


class Phase4I3MonotonicTTScaleTests(unittest.TestCase):
    def test_scale_grid_is_complete_monotonic_and_bounded(self) -> None:
        grid = monotonic_scale_grid([0.0, 0.25, 0.5, 1.0])
        self.assertEqual(len(grid), 20)
        self.assertIn((0.0, 0.0, 0.0), grid)
        self.assertIn((1.0, 1.0, 1.0), grid)
        self.assertTrue(
            all(far <= mid <= near for near, mid, far in grid)
        )
        with self.assertRaises(ValueError):
            monotonic_scale_grid([-0.1, 1.0])

    def test_group_scales_apply_only_to_intensity_residual(self) -> None:
        v1 = np.zeros((3, 2), dtype=np.float32)
        intensity = np.asarray(
            [[1, 2], [3, 4], [5, 6]],
            dtype=np.float32,
        )
        weights = np.asarray([0.2, 0.4, 0.6], dtype=np.float32)
        groups = np.asarray([0, 1, 2], dtype=np.int32)
        scores, scaled, selected = apply_group_scales(
            v1,
            intensity,
            weights,
            groups,
            (1.0, 0.5, 0.0),
        )
        self.assertTrue(np.allclose(selected, [1.0, 0.5, 0.0]))
        self.assertTrue(np.allclose(scaled, [0.2, 0.2, 0.0]))
        self.assertTrue(
            np.allclose(scores, intensity * scaled[:, None])
        )
        with self.assertRaises(ValueError):
            apply_group_scales(
                v1,
                intensity[:2],
                weights,
                groups,
                (1.0, 0.5, 0.0),
            )

    def test_safety_requires_non_degradation_in_both_regimes(self) -> None:
        safe = {
            "tactile_diff_mae": -0.01,
            "tactile_ssim": 0.01,
            "tactile_mask_iou": 0.01,
            "oracle_top1": 0.0,
        }
        harmful = dict(safe)
        harmful["tactile_diff_mae"] = 0.001
        self.assertTrue(scale_is_safe(safe, safe, 1e-12))
        self.assertFalse(scale_is_safe(safe, harmful, 1e-12))

    def test_scale_selection_prefers_best_safe_then_conservative_tie(self) -> None:
        rows = [
            {
                "near_scale": 1.0,
                "mid_scale": 0.5,
                "far_scale": 0.25,
                "safe": True,
                "objective": -0.01,
            },
            {
                "near_scale": 0.5,
                "mid_scale": 0.25,
                "far_scale": 0.0,
                "safe": True,
                "objective": -0.01,
            },
            {
                "near_scale": 1.0,
                "mid_scale": 1.0,
                "far_scale": 1.0,
                "safe": False,
                "objective": -0.02,
            },
        ]
        selected = choose_scale(rows)
        self.assertEqual(selected["far_scale"], 0.0)
        self.assertEqual(selected["near_scale"], 0.5)

    def test_delta_summary_uses_common_oracle_top1_definition(self) -> None:
        base = [
            retrieval_row("a", 0.02, 0.7, 0.2, 2),
            retrieval_row("b", 0.02, 0.7, 0.2, 1),
        ]
        current = [
            retrieval_row("a", 0.01, 0.8, 0.3, 1),
            retrieval_row("b", 0.03, 0.6, 0.1, 2),
        ]
        delta = delta_summary(base, current)
        self.assertAlmostEqual(delta["tactile_diff_mae"], 0.0)
        self.assertAlmostEqual(delta["tactile_ssim"], 0.0)
        self.assertAlmostEqual(delta["tactile_mask_iou"], 0.0)
        self.assertAlmostEqual(delta["oracle_top1"], 0.0)

    def test_diagnostics_use_predicted_ttc_not_true_probe(self) -> None:
        rows = []
        for index, ttc in enumerate((10.0, 40.0, 80.0, 90.0)):
            rows.append(
                {
                    "query_record_id": f"rec_{index // 2}",
                    "query_image_name": f"q_{index}",
                    "oof_fold": str(index % 2),
                    "query_probe": "5",
                    "predicted_ttc": str(ttc),
                    "robust_attenuation": str(0.7 + index * 0.01),
                    "ttc_entropy": str(index * 0.1),
                    "trajectory_stability": str(1.0 - index * 0.1),
                    "v1_tactile_diff_mae": "0.02",
                    "robust_tactile_diff_mae": "0.01",
                    "v1_tactile_ssim": "0.7",
                    "robust_tactile_ssim": "0.8",
                    "v1_tactile_mask_iou": "0.2",
                    "robust_tactile_mask_iou": "0.3",
                    "v1_ranker_oracle_embedding_rank": "2",
                    "robust_ranker_oracle_embedding_rank": "1",
                    "v1_selected_cache_image_name": "v1",
                    "robust_selected_cache_image_name": "robust",
                }
            )
        slices, records, report = phase4i2_diagnostics(
            rows,
            (30.0, 60.0),
            10,
        )
        ttc_slices = [
            row["slice"]
            for row in slices
            if row["dimension"] == "predicted_ttc_group"
        ]
        self.assertEqual(
            ttc_slices,
            ["near_lt30", "mid_30_60", "far_ge60"],
        )
        self.assertEqual(len(records), 2)
        self.assertEqual(report["query_count"], 4)
        self.assertTrue(
            all(set(row) == set(SLICE_FIELDS) for row in slices)
        )
        self.assertTrue(
            all(set(row) == set(RECORD_FIELDS) for row in records)
        )

    def test_scale_csv_schema_uses_full_metric_names(self) -> None:
        delta = {
            "tactile_diff_mae": -0.01,
            "tactile_ssim": 0.01,
            "tactile_mask_iou": 0.02,
            "oracle_top1": 0.03,
        }
        output = scale_csv_row(
            {
                "held_out_fold": "0",
                "near_scale": 1.0,
                "mid_scale": 0.5,
                "far_scale": 0.0,
                "safe": True,
                "objective": -0.02,
                "all_delta": delta,
                "predicted_far_delta": delta,
                "selected": True,
            }
        )
        self.assertEqual(set(output), set(SCALE_FIELDS))


if __name__ == "__main__":
    unittest.main()
