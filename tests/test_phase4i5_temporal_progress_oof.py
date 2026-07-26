from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.build_phase4i5_temporal_dino_features import (
    corridor_geometry,
    history_index,
    validate_frame_offsets,
)
from src.train_phase4i5_temporal_progress_oof import (
    PROGRESS_NAMES,
    TemporalProgressModel,
    confusion_report,
    load_temporal_cache,
    model_loss,
    probabilities_for,
    progress_targets,
    train_model,
    training_statistics,
    validate_temporal_cache_metadata,
)


def sample(probe: int = 5) -> dict[str, str]:
    return {
        "image_width": "768",
        "image_height": "512",
        "query_probe": str(probe),
    }


def track(frame: int = 10) -> dict[str, str]:
    return {
        "split": "0",
        "record_id": "rec_00001",
        "frame_id": str(frame),
        "tip_x": "100",
        "tip_y": "200",
    }


class Phase4I5TemporalProgressTests(unittest.TestCase):
    def test_offsets_reject_future_or_unsorted_frames(self) -> None:
        self.assertEqual(
            validate_frame_offsets([-15, -10, -5, 0]),
            (-15, -10, -5, 0),
        )
        with self.assertRaisesRegex(ValueError, "after the query"):
            validate_frame_offsets([-5, 0, 1])
        with self.assertRaisesRegex(ValueError, "sorted"):
            validate_frame_offsets([-5, -10, 0])

    def test_history_index_rejects_duplicate_frames(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "Duplicate"):
            history_index([track(), track()])

    def test_corridor_geometry_is_probe_independent(self) -> None:
        first = corridor_geometry(
            sample(5),
            track(),
            (300.0, 400.0),
            -5,
            -15,
        )
        second = corridor_geometry(
            sample(100),
            track(),
            (300.0, 400.0),
            -5,
            -15,
        )
        self.assertEqual(first[0], (200.0, 300.0))
        self.assertTrue(np.array_equal(first[1], second[1]))
        self.assertEqual(first[1].shape, (8,))

    def test_progress_targets_are_offline_labels(self) -> None:
        rows = [
            {"query_probe": str(value)}
            for value in (5, 10, 20, 30, 50, 75, 100)
        ]
        progress, ttc = progress_targets(
            rows,
            20,
            50,
            [5, 10, 20, 30, 50, 75, 100],
        )
        self.assertTrue(
            np.array_equal(progress, np.asarray([0, 0, 0, 1, 1, 2, 2]))
        )
        self.assertTrue(np.array_equal(ttc, np.arange(7)))

    def test_model_forward_has_only_online_inputs(self) -> None:
        parameters = set(
            inspect.signature(TemporalProgressModel.forward).parameters
        )
        self.assertEqual(
            parameters,
            {"self", "visual", "geometry", "valid_mask", "online"},
        )
        for forbidden in ("probe", "tactile", "touch", "contact_frame"):
            self.assertFalse(any(forbidden in name for name in parameters))
        model = TemporalProgressModel(6, 8, 10, 12, 7, 7, 0.0)
        output = model(
            torch.randn(4, 4, 6),
            torch.randn(4, 4, 8),
            torch.ones(4, 4),
            torch.randn(4, 10),
        )
        self.assertEqual(output["progress_logits"].shape, (4, 3))
        self.assertEqual(output["ordinal_logits"].shape, (4, 2))
        self.assertEqual(output["ttc_logits"].shape, (4, 7))

    def test_multitask_loss_is_finite_and_differentiable(self) -> None:
        model = TemporalProgressModel(6, 8, 10, 12, 7, 7, 0.0)
        output = model(
            torch.randn(6, 4, 6),
            torch.randn(6, 4, 8),
            torch.ones(6, 4),
            torch.randn(6, 10),
        )
        loss = model_loss(
            output,
            torch.tensor([0, 0, 1, 1, 2, 2]),
            torch.tensor([0, 1, 3, 4, 5, 6]),
            torch.ones(6),
            torch.ones(3),
            torch.ones(7),
            0.5,
            0.25,
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(
            any(parameter.grad is not None for parameter in model.parameters())
        )

    def test_temporal_training_smoke(self) -> None:
        rng = np.random.default_rng(3)
        count = 28
        visual = rng.normal(size=(count, 4, 6)).astype(np.float32)
        geometry = rng.normal(size=(count, 4, 8)).astype(np.float32)
        valid = np.ones((count, 4), dtype=np.float32)
        online = rng.normal(size=(count, 10)).astype(np.float32)
        ttc = np.tile(np.arange(7, dtype=np.int64), 4)
        progress = np.where(ttc <= 2, 0, np.where(ttc <= 4, 1, 2))
        records = np.asarray([f"rec_{index:03d}" for index in range(count)])
        fit = np.arange(21, dtype=np.int32)
        early = np.arange(21, 28, dtype=np.int32)
        cfg = {
            "ttc_values": [5, 10, 20, 30, 50, 75, 100],
            "projection_dim": 12,
            "hidden_dim": 7,
            "dropout": 0.0,
            "learning_rate": 0.001,
            "weight_decay": 0.001,
            "epochs": 2,
            "early_stopping_patience": 2,
            "train_batch_size": 8,
            "gradient_clip": 1.0,
            "ttc_loss_weight": 0.5,
            "ordinal_loss_weight": 0.25,
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            model, report = train_model(
                visual,
                geometry,
                valid,
                online,
                progress,
                ttc,
                records,
                fit,
                early,
                cfg,
                torch.device("cpu"),
                checkpoint,
                {"scope": "unit_test"},
            )
            output = probabilities_for(
                model,
                visual,
                geometry,
                valid,
                online,
                early,
                torch.device("cpu"),
                checkpoint,
            )
        self.assertEqual(output[0].shape, (7, 3))
        self.assertEqual(output[1].shape, (7, 2))
        self.assertEqual(output[2].shape, (7, 7))
        self.assertGreaterEqual(report["best_epoch"], 1)

    def test_statistics_ignore_invalid_frames(self) -> None:
        visual = np.ones((3, 4, 2), dtype=np.float32)
        geometry = np.ones((3, 4, 3), dtype=np.float32)
        visual[0, 0] = 1000
        geometry[0, 0] = 1000
        valid = np.ones((3, 4), dtype=np.float32)
        valid[0, 0] = 0
        online = np.arange(12, dtype=np.float32).reshape(3, 4)
        statistics = training_statistics(
            visual,
            geometry,
            valid,
            online,
            np.arange(3),
        )
        self.assertTrue(np.allclose(statistics["visual_mean"], 1.0))
        self.assertTrue(np.allclose(statistics["geometry_mean"], 1.0))

    def test_feature_cache_reorders_by_query_name(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / "temporal"
            np.save(
                Path(f"{prefix}.names.npy"),
                np.asarray(["b", "a"]),
            )
            np.save(
                Path(f"{prefix}.features.npy"),
                np.asarray([[[2.0]], [[1.0]]], dtype=np.float32),
            )
            np.save(
                Path(f"{prefix}.geometry.npy"),
                np.zeros((2, 1, 8), dtype=np.float32),
            )
            np.save(
                Path(f"{prefix}.valid.npy"),
                np.ones((2, 1), dtype=np.float32),
            )
            np.save(
                Path(f"{prefix}.padding.npy"),
                np.zeros((2, 1), dtype=np.float32),
            )
            visual, _, _, _ = load_temporal_cache(prefix, ["a", "b"])
        self.assertEqual(float(visual[0, 0, 0]), 1.0)
        self.assertEqual(float(visual[1, 0, 0]), 2.0)

    def test_feature_metadata_enforces_online_contract(self) -> None:
        cfg = {
            "frame_offsets": [-15, -10, -5, 0],
            "crop_size": 192,
            "dino_model": "dinov2_vits14",
            "dino_image_size": 224,
            "dino_layer": 12,
            "center_sigma": 0.35,
        }
        metadata = {
            **cfg,
            "query_true_probe_feature_used": False,
            "query_tactile_input": False,
            "future_visual_frames_used": False,
        }
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / "temporal"
            Path(f"{prefix}.json").write_text(
                json.dumps(metadata),
                encoding="utf-8",
            )
            loaded = validate_temporal_cache_metadata(prefix, cfg)
            self.assertFalse(loaded["future_visual_frames_used"])
            metadata["future_visual_frames_used"] = True
            Path(f"{prefix}.json").write_text(
                json.dumps(metadata),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "metadata mismatch"):
                validate_temporal_cache_metadata(prefix, cfg)

    def test_confusion_report_uses_true_rows(self) -> None:
        output = confusion_report(
            np.asarray([0, 1, 2, 2]),
            np.asarray([0, 2, 2, 1]),
            np.asarray([0, 0, 1, 1], dtype=np.float32),
            np.asarray([False, True, True, False]),
        )
        self.assertEqual(
            output["progress_confusion_matrix_true_rows"],
            [[1, 0, 0], [0, 0, 1], [0, 1, 1]],
        )
        self.assertEqual(output["far_recall"], 0.5)
        self.assertEqual(output["near_mid_retention"], 0.5)
        self.assertEqual(tuple(output["progress_recall"]), PROGRESS_NAMES)


if __name__ == "__main__":
    unittest.main()
