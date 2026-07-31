from __future__ import annotations

import unittest

import numpy as np

from src.audit_phase4i6b_temporal_far_candidate_pool import (
    classify_motion,
    motion_diagnostics,
)


def config() -> dict:
    return {
        "pause_ratio_threshold": 0.2,
        "turning_stability_threshold": 0.85,
        "turning_radians_threshold": 0.8,
        "variable_speed_cv_threshold": 0.35,
        "speed_slope_threshold": 0.15,
    }


class Phase4I6bTemporalFarCandidatePoolTests(unittest.TestCase):
    def test_classifies_straight_constant_motion(self) -> None:
        points = np.stack(
            [np.arange(32, dtype=np.float32), np.zeros(32)],
            axis=1,
        )
        diagnostics = motion_diagnostics(points, 0.25)
        self.assertEqual(
            classify_motion(diagnostics, config()),
            "straight_constant_velocity",
        )
        self.assertAlmostEqual(diagnostics["direction_stability"], 1.0)

    def test_classifies_straight_acceleration(self) -> None:
        speed = np.linspace(0.5, 3.0, 31, dtype=np.float32)
        x = np.concatenate([[0.0], np.cumsum(speed)])
        points = np.stack([x, np.zeros(32)], axis=1)
        diagnostics = motion_diagnostics(points, 0.25)
        self.assertEqual(
            classify_motion(diagnostics, config()),
            "straight_accelerating",
        )

    def test_classifies_turning_motion(self) -> None:
        angle = np.linspace(0.0, np.pi / 2.0, 32, dtype=np.float32)
        points = np.stack([np.cos(angle), np.sin(angle)], axis=1) * 20.0
        diagnostics = motion_diagnostics(points, 0.01)
        self.assertIn(
            classify_motion(diagnostics, config()),
            ("turning_constant_speed", "turning_variable_speed"),
        )
        self.assertGreater(diagnostics["cumulative_turn_radians"], 0.8)

    def test_classifies_pause_resume(self) -> None:
        x = np.concatenate(
            [
                np.arange(10, dtype=np.float32),
                np.full(10, 9.0, dtype=np.float32),
                np.arange(10.0, 22.0, dtype=np.float32),
            ]
        )
        points = np.stack([x, np.zeros(32)], axis=1)
        diagnostics = motion_diagnostics(points, 0.25)
        self.assertEqual(
            classify_motion(diagnostics, config()),
            "pause_resume",
        )


if __name__ == "__main__":
    unittest.main()
