from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.build_phase4i7_augmented_ttc_features import tip_geometry
from src.train_phase4i7_augmented_ttc_oof import (
    TemporalMotionProgressModel,
    auxiliary_pairs,
    metric_summary,
    nested_split,
    probabilities_for,
    targets_for,
    train_model,
)


def sample_row(
    record: str,
    probe: int,
    fold: int,
    source: str = "original",
    pair: str = "",
) -> dict[str, str]:
    return {
        "split": "3" if source == "increment" else "0",
        "record_id": record,
        "probe": str(probe),
        "oof_fold": str(fold),
        "phase4i7_source": source,
        "approved_pair_id": pair,
        "image_width": "100",
        "image_height": "80",
    }


class Phase4I7AugmentedTTCTests(unittest.TestCase):
    def test_tip_geometry_uses_current_track_only(self) -> None:
        row = sample_row("rec_a", 75, 0)
        row.update({"target_tip_x": "999", "target_tip_y": "999"})
        track = {
            "tip_x": "30",
            "tip_y": "20",
            "base_x": "20",
            "base_y": "20",
        }
        center, geometry = tip_geometry(row, track, -5, -15)
        self.assertEqual(center, (30.0, 20.0))
        self.assertEqual(geometry.shape, (8,))
        self.assertAlmostEqual(float(geometry[4]), 1.0)
        self.assertAlmostEqual(float(geometry[5]), 0.0)

    def test_auxiliary_pairs_are_fold_local(self) -> None:
        rows = [
            sample_row("a", 75, 0, "increment", "pair_1"),
            sample_row("a", 100, 0, "increment", "pair_1"),
            sample_row("b", 75, 0, "increment", "pair_1"),
            sample_row("b", 100, 0, "increment", "pair_1"),
            sample_row("c", 75, 1, "increment"),
            sample_row("c", 100, 1, "increment"),
        ]
        approved = [
            {
                "approved": "1",
                "record_a": "a",
                "record_b": "b",
            }
        ]
        monotonic, consistency = auxiliary_pairs(
            rows,
            np.asarray([0, 1, 2, 3], dtype=np.int32),
            approved,
        )
        self.assertEqual(len(monotonic), 2)
        self.assertEqual(len(consistency), 2)

    def test_nested_split_keeps_pair_group_together(self) -> None:
        rows = []
        for record_index in range(60):
            fold = record_index % 3
            pair = "pair_1" if record_index in {0, 1} else ""
            for probe in (5, 10, 20, 30, 50, 75, 100):
                rows.append(
                    sample_row(
                        f"rec_{record_index:03d}",
                        probe,
                        fold,
                        "increment" if record_index < 6 else "original",
                        pair,
                    )
                )
        cfg = {
            "early_stop_fraction": 0.2,
            "early_stop_seed": 17,
        }
        fit, early, held = nested_split(rows, "2", True, cfg)
        self.assertTrue(len(fit))
        self.assertTrue(len(early))
        self.assertTrue(len(held))
        partitions = {}
        for name, indices in (("fit", fit), ("early", early), ("held", held)):
            for index in indices:
                row = rows[int(index)]
                partitions.setdefault(row["record_id"], name)
                self.assertEqual(partitions[row["record_id"]], name)

    def test_visual_and_full_motion_features_are_supported(self) -> None:
        model = TemporalMotionProgressModel(
            visual_dim=12,
            geometry_dim=8,
            motion_dim=17,
            projection_dim=16,
            visual_hidden_dim=8,
            motion_hidden_dim=6,
            ttc_classes=7,
            dropout=0.0,
        )
        output = model(
            torch.zeros(2, 4, 12),
            torch.zeros(2, 4, 8),
            torch.ones(2, 4),
            torch.zeros(2, 32, 17),
            torch.ones(2, 32),
        )
        self.assertEqual(tuple(output["ttc_logits"].shape), (2, 7))

    def test_metric_summary_reports_ttc_and_far(self) -> None:
        progress = np.asarray(
            [[0.9, 0.05, 0.05], [0.05, 0.05, 0.9]],
            dtype=np.float32,
        )
        ttc = np.zeros((2, 7), dtype=np.float32)
        ttc[0, 0] = 1
        ttc[1, 6] = 1
        report = metric_summary(
            progress,
            ttc,
            np.asarray([0, 2]),
            np.asarray([0, 6]),
            np.asarray([5, 10, 20, 30, 50, 75, 100], dtype=np.float32),
        )
        self.assertEqual(report["ttc_mae_frames"], 0.0)
        self.assertEqual(report["far_recall"], 1.0)

    def test_augmented_training_smoke(self) -> None:
        rows = []
        probes = [5, 10, 20, 30, 50, 75, 100]
        for record_index in range(5):
            for probe in probes:
                rows.append(
                    sample_row(
                        f"rec_{record_index}",
                        probe,
                        record_index % 3,
                    )
                )
        progress, ttc = targets_for(rows, probes, 20, 50)
        rng = np.random.default_rng(7)
        visual = rng.normal(size=(len(rows), 2, 6)).astype(np.float32)
        geometry = rng.normal(size=(len(rows), 2, 8)).astype(np.float32)
        valid = np.ones((len(rows), 2), dtype=np.float32)
        motion = rng.normal(size=(len(rows), 32, 17)).astype(np.float32)
        motion_valid = np.ones((len(rows), 32), dtype=np.float32)
        records = np.asarray([row["record_id"] for row in rows])
        fit = np.arange(0, 28, dtype=np.int32)
        early = np.arange(28, 35, dtype=np.int32)
        cfg = {
            "projection_dim": 8,
            "hidden_dim": 6,
            "motion_hidden_dim": 5,
            "ttc_values": probes,
            "dropout": 0.0,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "ttc_loss_weight": 0.5,
            "ordinal_loss_weight": 0.25,
            "monotonic_loss_weight": 0.25,
            "monotonic_minimum_gap_frames": 10.0,
            "pair_consistency_loss_weight": 0.25,
            "pair_batch_size": 8,
            "gradient_clip": 1.0,
            "train_batch_size": 16,
            "epochs": 2,
            "early_stopping_patience": 2,
        }
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "model.pt"
            model, report = train_model(
                visual,
                geometry,
                valid,
                motion,
                motion_valid,
                progress,
                ttc,
                records,
                rows,
                fit,
                early,
                [],
                True,
                cfg,
                torch.device("cpu"),
                checkpoint,
                {"scope": "unit_test"},
            )
            progress_probability, ttc_probability = probabilities_for(
                model,
                visual,
                geometry,
                valid,
                motion,
                motion_valid,
                early,
                checkpoint,
                batch_size=16,
                device=torch.device("cpu"),
            )
        self.assertTrue(np.isfinite(report["best_validation_loss"]))
        self.assertEqual(progress_probability.shape, (7, 3))
        self.assertEqual(ttc_probability.shape, (7, 7))


if __name__ == "__main__":
    unittest.main()
