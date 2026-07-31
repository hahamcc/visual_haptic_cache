from __future__ import annotations

import inspect
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.train_phase4i2_ttc_robust_intensity_residual import (
    IntensityResidualRanker,
    feature_names,
    predicted_ttc_groups,
    query_features,
    robust_query_weights,
    train_ranker,
    uncertainty_attenuation,
)
from src.train_phase4i_factorized_residual_cascade import (
    ONLINE_PROGRESS_FIELDS,
    query_standardize,
)


def progress_rows(count: int) -> np.ndarray:
    values = np.zeros(
        (count, len(ONLINE_PROGRESS_FIELDS)),
        dtype=np.float32,
    )
    values[:, ONLINE_PROGRESS_FIELDS.index("predicted_ttc")] = np.linspace(
        10.0,
        90.0,
        count,
    )
    return values


class Phase4I2TTCRobustIntensityResidualTests(unittest.TestCase):
    def test_online_contract_excludes_true_probe_and_tactile(self) -> None:
        parameters = set(
            inspect.signature(IntensityResidualRanker.forward).parameters
        )
        self.assertEqual(
            parameters,
            {
                "self",
                "v1_scores",
                "intensity_scores",
                "features",
                "attenuation",
            },
        )
        for forbidden in ("probe", "tactile", "touch", "contact_frame"):
            self.assertFalse(
                any(forbidden in name for name in feature_names()),
                msg=f"forbidden online feature: {forbidden}",
            )

    def test_query_features_are_finite_and_expected_shape(self) -> None:
        rng = np.random.default_rng(42)
        v1 = query_standardize(
            rng.normal(size=(6, 32)).astype(np.float32)
        )
        intensity = query_standardize(
            rng.normal(size=(6, 32)).astype(np.float32)
        )
        progress = progress_rows(6)
        values = query_features(v1, intensity, progress)
        self.assertEqual(values.shape, (6, len(feature_names())))
        self.assertTrue(np.isfinite(values).all())

    def test_predicted_ttc_bucket_boundaries(self) -> None:
        progress = progress_rows(5)
        index = ONLINE_PROGRESS_FIELDS.index("predicted_ttc")
        progress[:, index] = [0.0, 29.9, 30.0, 59.9, 60.0]
        groups = predicted_ttc_groups(progress, (30.0, 60.0))
        self.assertTrue(
            np.array_equal(groups, np.asarray([0, 0, 1, 1, 2]))
        )

    def test_robust_weights_balance_groups_after_record_balance(self) -> None:
        progress = progress_rows(8)
        index = ONLINE_PROGRESS_FIELDS.index("predicted_ttc")
        progress[:, index] = [10, 10, 10, 20, 40, 50, 70, 90]
        records = np.asarray(
            ["a", "a", "a", "b", "c", "d", "e", "f"]
        )
        indices = np.arange(8, dtype=np.int32)
        weights = robust_query_weights(
            indices,
            records,
            progress,
            (30.0, 60.0),
            True,
        )
        groups = predicted_ttc_groups(progress, (30.0, 60.0))
        totals = [
            float(weights[groups == group].sum())
            for group in range(3)
        ]
        self.assertTrue(np.allclose(totals, totals[0], atol=1e-6))
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=6)

    def test_uncertainty_attenuation_penalizes_risk(self) -> None:
        progress = progress_rows(3)
        ttc = ONLINE_PROGRESS_FIELDS.index("predicted_ttc")
        entropy = ONLINE_PROGRESS_FIELDS.index("ttc_entropy")
        trajectory_padding = ONLINE_PROGRESS_FIELDS.index(
            "trajectory_padding_ratio"
        )
        crop_padding = ONLINE_PROGRESS_FIELDS.index("query_padding_ratio")
        progress[:, ttc] = [10.0, 90.0, 90.0]
        progress[:, entropy] = [0.0, 0.8, 0.8]
        progress[:, trajectory_padding] = [0.0, 0.0, 0.5]
        progress[:, crop_padding] = [0.0, 0.0, 0.25]
        values = uncertainty_attenuation(
            progress,
            {
                "entropy_attenuation": 0.35,
                "padding_attenuation": 0.5,
                "far_entropy_attenuation": 0.25,
                "minimum_attenuation": 0.25,
            },
        )
        self.assertEqual(values[0], 1.0)
        self.assertGreater(values[0], values[1])
        self.assertGreater(values[1], values[2])
        self.assertTrue(np.logical_and(values >= 0.25, values <= 1).all())

    def test_intensity_weight_is_nonnegative_bounded_and_attenuated(self) -> None:
        model = IntensityResidualRanker(
            feature_dim=5,
            maximum_weight=0.4,
            initial_weight_logit=0.0,
        )
        v1 = torch.zeros((2, 4))
        intensity = torch.ones((2, 4))
        attenuation = torch.tensor([1.0, 0.5])
        scores, raw, effective = model(
            v1,
            intensity,
            torch.zeros((2, 5)),
            attenuation,
        )
        self.assertTrue(torch.all(raw >= 0))
        self.assertTrue(torch.all(raw <= 0.4))
        self.assertTrue(torch.allclose(effective, raw * attenuation))
        self.assertTrue(torch.allclose(scores, effective[:, None] * intensity))

    def test_robust_training_smoke(self) -> None:
        rng = np.random.default_rng(7)
        query_count, candidate_count = 18, 8
        arrays = {
            "v1": query_standardize(
                rng.normal(size=(query_count, candidate_count)).astype(
                    np.float32
                )
            ),
            "intensity": query_standardize(
                rng.normal(size=(query_count, candidate_count)).astype(
                    np.float32
                )
            ),
            "target": np.abs(
                rng.normal(size=(query_count, candidate_count))
            ).astype(np.float32),
        }
        progress = progress_rows(query_count)
        features = query_features(
            arrays["v1"],
            arrays["intensity"],
            progress,
        )
        attenuation = np.ones(query_count, dtype=np.float32)
        records = np.asarray([f"rec_{index:03d}" for index in range(query_count)])
        validation = np.asarray([0, 6, 9, 12, 15, 17], dtype=np.int32)
        fit = np.setdiff1d(
            np.arange(query_count, dtype=np.int32),
            validation,
        )
        cfg = {
            "predicted_ttc_groups": [30.0, 60.0],
            "maximum_intensity_weight": 0.5,
            "initial_weight_logit": -3.0,
            "learning_rate": 0.001,
            "weight_decay": 0.001,
            "target_temperature": 0.02,
            "score_temperature": 1.0,
            "residual_regularization": 0.05,
            "batch_size": 4,
            "epochs": 2,
            "early_stopping_patience": 2,
            "gradient_clip": 1.0,
        }
        with tempfile.TemporaryDirectory() as directory:
            model, report = train_ranker(
                arrays,
                features,
                progress,
                attenuation,
                fit,
                validation,
                records,
                True,
                cfg,
                torch.device("cpu"),
                Path(directory) / "model.pt",
                {"scope": "unit_test"},
            )
        self.assertIsInstance(model, IntensityResidualRanker)
        self.assertGreaterEqual(report["best_epoch"], 1)
        self.assertTrue(np.isfinite(report["best_validation_loss"]))


if __name__ == "__main__":
    unittest.main()
