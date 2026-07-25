from __future__ import annotations

import inspect
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.evaluate_phase4h_factorized_intensity_oof import (
    bootstrap_prediction_error,
    predict_regressor,
    select_from_shortlist,
    train_regressor,
)
from src.phase4h_factorized_tactile import (
    INTENSITY_DIM,
    SHAPE_DIM,
    LowCapacityIntensityRegressor,
    factorized_tactile_latents,
    standardized_distance,
)


class Phase4HFactorizedTactileTests(unittest.TestCase):
    def test_shape_is_separate_from_intensity_for_fixed_mask(self) -> None:
        low = np.zeros((96, 96, 3), dtype=np.float32)
        high = np.zeros_like(low)
        low[32:64, 40:56] = 0.2
        high[32:64, 40:56] = 0.4
        low_shape, low_intensity = factorized_tactile_latents(low, 0.04)
        high_shape, high_intensity = factorized_tactile_latents(high, 0.04)
        self.assertEqual(low_shape.shape, (SHAPE_DIM,))
        self.assertEqual(low_intensity.shape, (INTENSITY_DIM,))
        self.assertTrue(np.array_equal(low_shape, high_shape))
        self.assertGreater(high_intensity[0], low_intensity[0])
        self.assertGreater(high_intensity[1], low_intensity[1])

    def test_empty_contact_is_finite(self) -> None:
        shape, intensity = factorized_tactile_latents(
            np.zeros((96, 96, 3), dtype=np.float32),
            0.04,
        )
        self.assertTrue(np.isfinite(shape).all())
        self.assertTrue(np.isfinite(intensity).all())

    def test_predictor_forward_has_no_tactile_argument(self) -> None:
        parameters = set(
            inspect.signature(LowCapacityIntensityRegressor.forward).parameters
        )
        self.assertEqual(parameters, {"self", "online_features"})
        model = LowCapacityIntensityRegressor(12, hidden_dim=8)
        self.assertEqual(model(torch.randn(3, 12)).shape, (3, INTENSITY_DIM))

    def test_standardized_distance_and_shortlist(self) -> None:
        query = np.asarray([[0.0, 0.0]], dtype=np.float32)
        candidates = np.asarray(
            [[[5.0, 5.0], [1.0, 1.0], [0.0, 0.0]]],
            dtype=np.float32,
        )
        distance = standardized_distance(query, candidates)
        ranks = np.asarray([[1, 2, 3]], dtype=np.int32)
        self.assertEqual(int(distance.argmin(axis=1)[0]), 2)
        self.assertEqual(int(select_from_shortlist(distance, ranks, 2)[0]), 1)

    def test_bootstrap_detects_consistent_improvement(self) -> None:
        rows = [
            {"record_id": f"rec_{index:05d}", "probe": "100"}
            for index in range(20)
        ]
        result = bootstrap_prediction_error(
            rows,
            np.full(20, 0.2, dtype=np.float32),
            np.full(20, 0.4, dtype=np.float32),
            iterations=100,
            seed=7,
        )
        self.assertTrue(result["accepted"])
        self.assertLess(result["all"]["bootstrap_95_ci"][1], 0)
        self.assertLess(result["far_probe75_100"]["bootstrap_95_ci"][1], 0)

    def test_low_capacity_training_smoke(self) -> None:
        rng = np.random.default_rng(3)
        features = rng.normal(size=(16, 5)).astype(np.float32)
        projection = rng.normal(size=(5, INTENSITY_DIM)).astype(np.float32)
        targets = (features @ projection * 0.1).astype(np.float32)
        rows = [{"record_id": f"rec_{index:05d}"} for index in range(16)]
        fit = np.arange(12, dtype=np.int32)
        validation = np.arange(12, 16, dtype=np.int32)
        config = {
            "hidden_dim": 8,
            "dropout": 0.0,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "batch_size": 4,
            "epochs": 3,
            "early_stopping_patience": 2,
            "gradient_clip": 1.0,
        }
        torch.manual_seed(3)
        with tempfile.TemporaryDirectory() as directory:
            model, report = train_regressor(
                features,
                targets,
                fit,
                validation,
                rows,
                config,
                torch.device("cpu"),
                Path(directory) / "best.pt",
                {"scope": "unit-test"},
            )
            prediction = predict_regressor(
                model,
                features,
                validation,
                batch_size=2,
                device=torch.device("cpu"),
            )
        self.assertEqual(prediction.shape, (4, INTENSITY_DIM))
        self.assertGreaterEqual(report["best_epoch"], 1)
        self.assertTrue(np.isfinite(prediction).all())


if __name__ == "__main__":
    unittest.main()
