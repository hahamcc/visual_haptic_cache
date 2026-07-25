from __future__ import annotations

import inspect
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.evaluate_phase4h_factorized_intensity_oof import (
    build_factor_candidate_output,
    bootstrap_prediction_error,
    fast_bootstrap_comparison,
    load_compatible_regressor_checkpoint,
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

    def test_fast_retrieval_bootstrap_detects_consistent_improvement(self) -> None:
        base, improved = [], []
        for index in range(20):
            common = {
                "query_image_name": f"image_{index:05d}.png",
                "query_record_id": f"rec_{index:05d}",
                "query_probe": "100",
            }
            base.append(
                {
                    **common,
                    "tactile_diff_mae": "0.020",
                    "tactile_ssim": "0.700",
                    "tactile_mask_iou": "0.200",
                    "ranker_oracle_embedding_rank": "2",
                }
            )
            improved.append(
                {
                    **common,
                    "tactile_diff_mae": "0.010",
                    "tactile_ssim": "0.800",
                    "tactile_mask_iou": "0.300",
                    "ranker_oracle_embedding_rank": "1",
                }
            )
        result = fast_bootstrap_comparison(
            base,
            improved,
            {"bootstrap_iterations": 100, "bootstrap_seed": 7},
        )
        self.assertTrue(result["accepted"])
        self.assertLess(
            result["all"]["bootstrap_95_ci"]["tactile_diff_mae"][1],
            0,
        )
        self.assertGreaterEqual(
            result["far_probe75_100"]["bootstrap_95_ci"]["tactile_mask_iou"][0],
            0,
        )

    def test_factor_candidate_output_exports_only_ranker_inputs_and_labels(
        self,
    ) -> None:
        rows = [
            {
                "record_id": "rec_00000",
                "image_name": "q.png",
                "probe": "75",
            },
            {
                "record_id": "rec_00001",
                "image_name": "c.png",
                "probe": "5",
            },
        ]
        v1_groups = {
            "q.png": [
                {
                    "candidate_rank": "1",
                    "candidate_score": "0.1",
                }
            ],
            "c.png": [
                {
                    "candidate_rank": "1",
                    "candidate_score": "0.2",
                }
            ],
        }
        recipe = {
            ("q.png", "c.png"): {
                "candidate_score": "-0.8",
                "detail_patch_score": "0.8",
                "context_patch_score": "0.7",
                "wide_patch_score": "",
                "position_aware_match_score": "",
                "candidate_tactile_embedding_distance": "0.2",
                "candidate_tactile_ssim": "0.8",
                "candidate_tactile_mask_iou": "0.3",
            },
            ("c.png", "q.png"): {
                "candidate_score": "-0.6",
                "detail_patch_score": "0.6",
                "context_patch_score": "0.5",
                "wide_patch_score": "",
                "position_aware_match_score": "",
                "candidate_tactile_embedding_distance": "0.3",
                "candidate_tactile_ssim": "0.7",
                "candidate_tactile_mask_iou": "0.2",
            },
        }
        distances = {
            key: np.asarray([[value], [value + 0.1]], dtype=np.float32)
            for key, value in {
                "global_median": 0.4,
                "motion_only": 0.3,
                "dino_only": 0.2,
                "dino_motion": 0.1,
            }.items()
        }
        output = build_factor_candidate_output(
            rows,
            np.asarray([[1], [0]], dtype=np.int32),
            v1_groups,
            recipe,
            np.asarray([[1], [1]], dtype=np.int32),
            np.asarray([[2], [2]], dtype=np.int32),
            distances,
            {"q.png": "0", "c.png": "1"},
        )
        self.assertEqual(len(output), 2)
        self.assertEqual(output[0]["candidate_image_name"], "c.png")
        self.assertEqual(
            output[0]["predicted_intensity_distance_dino_motion"],
            "0.100000001",
        )
        self.assertNotIn("query_intensity_prediction_mae", output[0])

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
            checkpoint_path = Path(directory) / "best.pt"
            metadata = {
                "scope": "unit-test",
                "predictor": "dino_only",
                "fold": "0",
                "seed": 3,
                "primary_recipe": "unit",
                "feature_mean": np.zeros(5, dtype=np.float32),
                "feature_std": np.ones(5, dtype=np.float32),
                "intensity_mean": np.zeros(INTENSITY_DIM, dtype=np.float32),
                "intensity_std": np.ones(INTENSITY_DIM, dtype=np.float32),
            }
            model, report = train_regressor(
                features,
                targets,
                fit,
                validation,
                rows,
                config,
                torch.device("cpu"),
                checkpoint_path,
                metadata,
            )
            prediction = predict_regressor(
                model,
                features,
                validation,
                batch_size=2,
                device=torch.device("cpu"),
            )
            reused = load_compatible_regressor_checkpoint(
                checkpoint_path,
                features.shape[1],
                config,
                metadata,
                torch.device("cpu"),
            )
        self.assertEqual(prediction.shape, (4, INTENSITY_DIM))
        self.assertGreaterEqual(report["best_epoch"], 1)
        self.assertTrue(np.isfinite(prediction).all())
        self.assertIsNotNone(reused)


if __name__ == "__main__":
    unittest.main()
