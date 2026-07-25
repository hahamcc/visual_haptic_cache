from __future__ import annotations

import unittest
import tempfile
import inspect
from pathlib import Path

import numpy as np
import torch

from src.train_phase4i_factorized_residual_cascade import (
    FactorizedResidualCascade,
    build_gate_oof_splits,
    cascade_feature_names,
    cascade_query_features,
    gate_feature_names,
    gate_features,
    normalized_entropy,
    predict_cascade,
    query_standardize,
    strict_triple_labels,
    train_cascade,
)


class Phase4IFactorizedResidualCascadeTests(unittest.TestCase):
    def test_query_standardization_is_per_query(self) -> None:
        values = np.asarray(
            [[1.0, 2.0, 3.0], [10.0, 14.0, 18.0]],
            dtype=np.float32,
        )
        standardized = query_standardize(values)
        self.assertTrue(
            np.allclose(standardized.mean(axis=1), 0.0, atol=1e-6)
        )
        self.assertTrue(
            np.allclose(standardized.std(axis=1), 1.0, atol=1e-6)
        )

    def test_residual_weights_are_nonnegative_and_bounded(self) -> None:
        model = FactorizedResidualCascade(
            feature_dim=11,
            maximum_dino_weight=0.4,
            maximum_intensity_weight=0.3,
            initial_weight_logit=-3.0,
        )
        scores, weights = model(
            torch.zeros((5, 32)),
            torch.randn((5, 32)),
            torch.randn((5, 32)),
            torch.randn((5, 11)),
        )
        self.assertEqual(scores.shape, (5, 32))
        self.assertEqual(weights.shape, (5, 2))
        self.assertTrue(torch.all(weights >= 0))
        self.assertTrue(torch.all(weights[:, 0] <= 0.4))
        self.assertTrue(torch.all(weights[:, 1] <= 0.3))

    def test_online_feature_builders_are_finite(self) -> None:
        rng = np.random.default_rng(9)
        v1 = query_standardize(rng.normal(size=(8, 32)).astype(np.float32))
        dino = query_standardize(rng.normal(size=(8, 32)).astype(np.float32))
        intensity = query_standardize(
            rng.normal(size=(8, 32)).astype(np.float32)
        )
        query_features = cascade_query_features(v1, dino, intensity)
        cascade_scores = v1 + 0.1 * dino + 0.2 * intensity
        weights = np.full((8, 2), [0.1, 0.2], dtype=np.float32)
        safety_features = gate_features(
            {"v1": v1, "dino": dino, "intensity": intensity},
            cascade_scores,
            weights,
        )
        self.assertEqual(query_features.shape, (8, 11))
        self.assertEqual(safety_features.shape, (8, 10))
        self.assertTrue(np.isfinite(query_features).all())
        self.assertTrue(np.isfinite(safety_features).all())
        self.assertTrue(
            np.logical_and(
                normalized_entropy(v1) >= 0,
                normalized_entropy(v1) <= 1,
            ).all()
        )

    def test_online_interfaces_do_not_accept_future_tactile_or_probe(self) -> None:
        forward_parameters = set(
            inspect.signature(FactorizedResidualCascade.forward).parameters
        )
        self.assertEqual(
            forward_parameters,
            {
                "self",
                "v1_scores",
                "dino_scores",
                "intensity_scores",
                "query_features",
            },
        )
        feature_names = cascade_feature_names() + gate_feature_names()
        for forbidden in ("tactile", "touch", "true_probe", "contact_frame"):
            self.assertFalse(
                any(forbidden in name for name in feature_names),
                msg=f"forbidden online feature: {forbidden}",
            )

    def test_gate_split_imbalance_returns_safe_fallback(self) -> None:
        query_rows = [
            {"oof_fold": str(index % 3)}
            for index in range(30)
        ]
        records = np.asarray(
            [f"rec_{index:05d}" for index in range(len(query_rows))]
        )
        targets = np.zeros(len(query_rows), dtype=np.float32)
        splits, reason = build_gate_oof_splits(
            query_rows,
            records,
            ["0", "1", "2"],
            targets,
            {
                "inner_validation_fraction": 0.2,
                "gate_inner_split_seed": 17,
            },
        )
        self.assertEqual(splits, [])
        self.assertIn("too imbalanced", reason)

    def test_strict_triple_win_requires_all_three_metrics(self) -> None:
        base = [
            {
                "query_image_name": "q0",
                "tactile_diff_mae": "0.02",
                "tactile_ssim": "0.70",
                "tactile_mask_iou": "0.20",
            },
            {
                "query_image_name": "q1",
                "tactile_diff_mae": "0.02",
                "tactile_ssim": "0.70",
                "tactile_mask_iou": "0.20",
            },
        ]
        cascade = [
            {
                "query_image_name": "q0",
                "tactile_diff_mae": "0.01",
                "tactile_ssim": "0.75",
                "tactile_mask_iou": "0.25",
            },
            {
                "query_image_name": "q1",
                "tactile_diff_mae": "0.01",
                "tactile_ssim": "0.69",
                "tactile_mask_iou": "0.25",
            },
        ]
        self.assertTrue(
            np.array_equal(
                strict_triple_labels(base, cascade),
                np.asarray([1.0, 0.0], dtype=np.float32),
            )
        )

    def test_cascade_training_smoke(self) -> None:
        rng = np.random.default_rng(4)
        queries, candidates = 24, 8
        raw_v1 = rng.normal(size=(queries, candidates)).astype(np.float32)
        raw_dino = rng.normal(size=(queries, candidates)).astype(np.float32)
        raw_intensity = rng.normal(size=(queries, candidates)).astype(
            np.float32
        )
        arrays = {
            "v1": query_standardize(raw_v1),
            "dino": query_standardize(raw_dino),
            "intensity": query_standardize(raw_intensity),
            "target": np.square(
                raw_v1 - 0.2 * raw_dino - 0.1 * raw_intensity
            ).astype(np.float32),
        }
        features = cascade_query_features(
            arrays["v1"],
            arrays["dino"],
            arrays["intensity"],
        )
        fit = np.arange(18, dtype=np.int32)
        validation = np.arange(18, queries, dtype=np.int32)
        records = np.asarray([f"rec_{index:05d}" for index in range(queries)])
        config = {
            "maximum_dino_weight": 0.5,
            "maximum_intensity_weight": 0.5,
            "initial_weight_logit": -3.0,
            "target_temperature": 0.02,
            "score_temperature": 1.0,
            "residual_regularization": 0.05,
            "batch_size": 8,
            "epochs": 3,
            "early_stopping_patience": 2,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "gradient_clip": 1.0,
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "cascade.pt"
            model, report = train_cascade(
                arrays,
                features,
                fit,
                validation,
                records,
                config,
                torch.device("cpu"),
                checkpoint_path,
                {"scope": "unit-test"},
            )
            scores, weights = predict_cascade(
                model,
                arrays,
                features,
                validation,
                torch.device("cpu"),
                checkpoint_path,
            )
        self.assertEqual(scores.shape, (len(validation), candidates))
        self.assertEqual(weights.shape, (len(validation), 2))
        self.assertGreaterEqual(report["best_epoch"], 1)
        self.assertTrue(np.isfinite(scores).all())


if __name__ == "__main__":
    unittest.main()
