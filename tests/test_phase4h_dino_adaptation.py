from __future__ import annotations

import inspect
import gc
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.build_phase4h_dino_ablation import encode_rows
from src.phase4h_dino_adaptation import (
    TACTILE_LATENT_DIM,
    TactileLatentProjector,
    assert_development_only,
    assert_candidate_identity,
    canonical_rotation_degrees,
    combine_layer_tokens,
    contact_crop_reflect,
    deployable_motion_feature,
    position_aware_soft_similarity,
    tactile_latent,
)
from src.train_phase4h_dino_tactile_alignment import model_loss


class Phase4HDinoAdaptationTests(unittest.TestCase):
    def base_row(self) -> dict[str, str]:
        return {
            "split": "0",
            "record_id": "rec_00001",
            "frame_id": "20",
            "image_width": "32",
            "image_height": "24",
            "tip_x": "10",
            "tip_y": "12",
            "base_x": "10",
            "base_y": "4",
            "direction_x": "0",
            "direction_y": "1",
            "probe": "100",
            "contact_frame_detected": "120",
        }

    def test_tactile_latent_has_fixed_shape_and_finite_values(self) -> None:
        diff = np.zeros((96, 96, 3), dtype=np.float32)
        diff[40:56, 44:60] = 0.2
        latent = tactile_latent(diff, threshold=0.04)
        self.assertEqual(latent.shape, (TACTILE_LATENT_DIM,))
        self.assertTrue(np.isfinite(latent).all())
        self.assertGreater(latent[70], 0.0)  # active mask area

    def test_reflection_crop_has_no_black_border_artifact(self) -> None:
        image = np.full((12, 16, 3), 127, dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "constant.png"
            Image.fromarray(image).save(path)
            crop, padding = contact_crop_reflect(path, 0.0, 0.0, 10, rotation_degrees=35.0)
        self.assertEqual(crop.shape, (10, 10, 3))
        self.assertGreater(padding, 0.0)
        self.assertTrue(np.allclose(crop, 127.0 / 255.0, atol=2.0 / 255.0))

    def test_motion_axis_falls_back_to_sensor_axis(self) -> None:
        angle, used = canonical_rotation_degrees(self.base_row(), "motion_axis", {})
        self.assertEqual(used, "sensor_axis_fallback")
        self.assertAlmostEqual(angle, 0.0, places=5)

    def test_intermediate_layer_mean_is_normalized_before_average(self) -> None:
        layers = {
            8: torch.ones(2, 4, 3),
            10: torch.ones(2, 4, 3) * 2,
            12: torch.ones(2, 4, 3) * 3,
        }
        result = combine_layer_tokens(layers, "mean_8_10_12")
        expected = torch.full_like(result, 1.0 / np.sqrt(3.0))
        self.assertTrue(torch.allclose(result, expected))

    def test_position_aware_similarity_prefers_aligned_tokens(self) -> None:
        torch.manual_seed(7)
        query = torch.randn(1, 16, 8)
        aligned = query[:, None].clone()
        shuffled = query[:, torch.randperm(16)][:, None]
        cache = torch.cat((aligned, shuffled), dim=1)
        score = position_aware_soft_similarity(query, cache, radius=2)
        self.assertGreater(float(score[0, 0]), float(score[0, 1]))

    def test_projector_has_no_tactile_forward_argument(self) -> None:
        parameters = set(inspect.signature(TactileLatentProjector.forward).parameters)
        self.assertEqual(parameters, {"self", "visual_motion"})
        model = TactileLatentProjector(12)
        self.assertEqual(model(torch.randn(3, 12)).shape, (3, TACTILE_LATENT_DIM))

    def test_alignment_loss_is_finite_and_backpropagates(self) -> None:
        torch.manual_seed(3)
        model = TactileLatentProjector(12)
        features = torch.randn(4, 12)
        query_latents = torch.randn(4, TACTILE_LATENT_DIM)
        candidate_latents = torch.randn(4, 32, TACTILE_LATENT_DIM)
        dino_ranks = torch.arange(1, 33)[None].expand(4, -1)
        loss, parts = model_loss(
            model,
            features,
            query_latents,
            candidate_latents,
            dino_ranks,
            {
                "target_temperature": 0.02,
                "score_temperature": 0.1,
                "hard_negative_margin": 0.1,
                "listwise_loss_weight": 1.0,
                "latent_loss_weight": 0.1,
                "hard_negative_loss_weight": 0.5,
            },
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(np.isfinite(value) for value in parts.values()))
        loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_deployable_features_ignore_true_probe_and_contact_frame(self) -> None:
        row = self.base_row()
        other = dict(row)
        other["probe"] = "5"
        other["contact_frame_detected"] = "25"
        trajectory = np.zeros((16, 17), dtype=np.float32)
        mask = np.ones(16, dtype=np.float32)
        quality = {
            "real_point_count": 16.0,
            "history_span_frames": 15.0,
            "padding_ratio": 0.0,
            "max_frame_gap": 1.0,
            "cumulative_displacement": 5.0,
        }
        ttc = {"predicted_ttc": "31.0"}
        left = deployable_motion_feature(row, 16.0, 12.0, trajectory, mask, quality, ttc)
        right = deployable_motion_feature(other, 16.0, 12.0, trajectory, mask, quality, ttc)
        self.assertTrue(np.array_equal(left, right))

    def test_candidate_identity_ignores_rank_order_but_not_membership(self) -> None:
        first = {
            "q": [
                {"candidate_record_id": "a", "candidate_image_name": "a1"},
                {"candidate_record_id": "b", "candidate_image_name": "b1"},
            ]
        }
        second = {"q": list(reversed(first["q"]))}
        assert_candidate_identity(first, second)
        second["q"][0] = {"candidate_record_id": "c", "candidate_image_name": "c1"}
        with self.assertRaises(RuntimeError):
            assert_candidate_identity(first, second)

    def test_development_guard_requires_partition_manifest(self) -> None:
        with self.assertRaises(FileNotFoundError):
            assert_development_only(
                [{"split": "0", "record_id": "rec_00001"}],
                Path("definitely_missing_phase4h_partition.csv"),
            )

    def test_token_cache_is_resumable_and_recipe_bound(self) -> None:
        class FakeBackbone:
            calls = 0

            def forward_layers(self, images, layers):
                self.calls += 1
                batch = images.shape[0]
                token = torch.arange(4, dtype=torch.float32).view(1, 1, 4)
                return {
                    layer: token.expand(batch, 256, 4).clone()
                    for layer in layers
                }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "vision.png"
            Image.fromarray(np.full((24, 32, 3), 127, dtype=np.uint8)).save(image_path)
            row = self.base_row()
            row.update({"image_name": "rec_00001/frames/20.png", "vision_path": str(image_path)})
            backbone = FakeBackbone()
            arguments = dict(
                backbone=backbone,
                rows=[row],
                coordinates=[(10.0, 12.0)],
                size=48,
                padding_mode="black",
                canonicalization="raw",
                layer_recipe="layer12",
                tracks={},
                device=torch.device("cpu"),
                batch_size=1,
                center_sigma=0.35,
                label="unit-test",
                cache_prefix=root / "tokens",
            )
            first = encode_rows(**arguments)
            second = encode_rows(**arguments)
            self.assertEqual(backbone.calls, 1)
            self.assertEqual(first[0].shape, (1, 256, 4))
            self.assertTrue(np.array_equal(first[0], second[0]))
            del first, second
            gc.collect()
            arguments["center_sigma"] = 0.4
            with self.assertRaises(RuntimeError):
                encode_rows(**arguments)


if __name__ == "__main__":
    unittest.main()
