from __future__ import annotations

import unittest

import numpy as np

from src.audit_phase4i6c_robust_motion_quality import (
    median_smooth,
    pose_quality_diagnostics,
)


def config() -> dict:
    return {
        "maximum_plausible_step_px": 8.0,
        "maximum_single_step_px": 24.0,
        "maximum_jump_ratio": 0.1,
        "maximum_tip_base_length_cv": 0.15,
        "minimum_velocity_coherence": 0.0,
        "maximum_velocity_residual_p90_px": 5.0,
        "maximum_velocity_residual_max_px": 12.0,
        "minimum_track_confidence": 0.25,
        "coherence_step_threshold_px": 0.25,
        "median_filter_radius": 1,
        "pause_step_threshold_px": 0.25,
    }


class Phase4I6cRobustMotionQualityTests(unittest.TestCase):
    def test_rigid_translation_passes(self) -> None:
        tip = np.stack(
            [np.arange(32, dtype=np.float32), np.zeros(32)],
            axis=1,
        )
        base = tip + np.asarray([0.0, 10.0], dtype=np.float32)
        confidence = np.ones(32, dtype=np.float32)
        output = pose_quality_diagnostics(
            tip,
            base,
            confidence,
            confidence,
            config(),
        )
        self.assertTrue(output["passed"])
        self.assertAlmostEqual(output["tip_base_length_cv"], 0.0)
        self.assertAlmostEqual(output["tip_base_velocity_coherence"], 1.0)

    def test_tip_only_jump_fails_rigid_quality(self) -> None:
        tip = np.stack(
            [np.arange(32, dtype=np.float32), np.zeros(32)],
            axis=1,
        )
        base = tip + np.asarray([0.0, 10.0], dtype=np.float32)
        tip[15] += np.asarray([30.0, 20.0], dtype=np.float32)
        confidence = np.ones(32, dtype=np.float32)
        output = pose_quality_diagnostics(
            tip,
            base,
            confidence,
            confidence,
            config(),
        )
        self.assertFalse(output["passed"])
        self.assertIn(
            "tip_base_velocity_residual_max_exceeded",
            output["failure_reasons"],
        )

    def test_median_filter_suppresses_isolated_center_jump(self) -> None:
        points = np.stack(
            [np.arange(7, dtype=np.float32), np.zeros(7)],
            axis=1,
        )
        points[3, 0] = 100.0
        smoothed = median_smooth(points, 1)
        self.assertEqual(smoothed[3, 0], 4.0)

    def test_low_intermediate_confidence_fails(self) -> None:
        tip = np.stack(
            [np.arange(32, dtype=np.float32), np.zeros(32)],
            axis=1,
        )
        base = tip + np.asarray([0.0, 10.0], dtype=np.float32)
        tip_confidence = np.ones(32, dtype=np.float32)
        tip_confidence[8] = 0.1
        base_confidence = np.ones(32, dtype=np.float32)
        output = pose_quality_diagnostics(
            tip,
            base,
            tip_confidence,
            base_confidence,
            config(),
        )
        self.assertFalse(output["passed"])
        self.assertIn(
            "tip_confidence_below_threshold",
            output["failure_reasons"],
        )


if __name__ == "__main__":
    unittest.main()
