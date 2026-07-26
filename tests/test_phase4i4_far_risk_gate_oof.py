from __future__ import annotations

import inspect
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.train_phase4i4_far_risk_gate_oof import (
    THRESHOLD_FIELDS,
    FarRiskClassifier,
    assert_csv_schema,
    choose_recall_threshold,
    far_labels,
    feature_names,
    online_features,
    probabilities_for,
    record_balanced_weights,
    select_by_risk,
    threshold_metrics,
    threshold_csv_row,
    train_model,
)
from src.train_phase4i_factorized_residual_cascade import (
    ONLINE_PROGRESS_FIELDS,
)


def progress_row(index: int, probe: int = 5) -> dict[str, str]:
    row = {
        field: str((index + 1) / (field_index + 2))
        for field_index, field in enumerate(ONLINE_PROGRESS_FIELDS)
    }
    row["predicted_ttc"] = str(10.0 + index * 10.0)
    row["ttc_entropy"] = str(0.1 + index * 0.02)
    row["query_probe"] = str(probe)
    return row


def retrieval(name: str, mae: float) -> dict[str, str]:
    return {
        "query_image_name": name,
        "selected_cache_image_name": f"cache_{name}",
        "ranker_oracle_embedding_rank": "1",
        "tactile_diff_mae": str(mae),
        "tactile_ssim": "0.8",
        "tactile_mask_iou": "0.3",
    }


class Phase4I4FarRiskGateTests(unittest.TestCase):
    def test_online_feature_contract_excludes_true_probe_and_tactile(self) -> None:
        rows = [progress_row(index) for index in range(4)]
        values = online_features(rows)
        self.assertEqual(values.shape, (4, len(feature_names())))
        self.assertTrue(np.isfinite(values).all())
        for forbidden in ("probe", "tactile", "touch", "contact_frame"):
            self.assertFalse(
                any(forbidden in name for name in feature_names())
            )
        self.assertEqual(
            set(inspect.signature(FarRiskClassifier.forward).parameters),
            {"self", "features"},
        )

    def test_true_probe_is_offline_label_only(self) -> None:
        rows = [
            progress_row(0, 5),
            progress_row(1, 50),
            progress_row(2, 75),
            progress_row(3, 100),
        ]
        self.assertTrue(
            np.array_equal(
                far_labels(rows, 75),
                np.asarray([0, 0, 1, 1], dtype=np.float32),
            )
        )

    def test_record_weights_balance_records(self) -> None:
        records = np.asarray(["a", "a", "a", "b", "c", "c"])
        indices = np.arange(len(records), dtype=np.int32)
        weights = record_balanced_weights(indices, records)
        totals = {
            record: float(weights[records == record].sum())
            for record in np.unique(records)
        }
        self.assertTrue(
            np.allclose(list(totals.values()), next(iter(totals.values())))
        )
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=6)

    def test_threshold_selection_is_recall_first_then_low_fpr(self) -> None:
        probabilities = np.asarray([0.9, 0.8, 0.2, 0.1], dtype=np.float32)
        targets = np.asarray([1, 0, 1, 0], dtype=np.float32)
        selected, options = choose_recall_threshold(
            probabilities,
            targets,
            0.9,
        )
        self.assertEqual(selected["recall"], 1.0)
        self.assertEqual(selected["false_positive_rate"], 0.5)
        self.assertAlmostEqual(selected["threshold"], 0.2)
        self.assertEqual(len(options), 4)

    def test_risk_selection_reverts_only_flagged_queries(self) -> None:
        v1 = [retrieval("a", 0.02), retrieval("b", 0.02)]
        current = [retrieval("a", 0.01), retrieval("b", 0.01)]
        selected = select_by_risk(
            v1,
            current,
            np.asarray([True, False]),
        )
        self.assertIs(selected[0], v1[0])
        self.assertIs(selected[1], current[1])

    def test_confusion_metrics(self) -> None:
        values = threshold_metrics(
            np.asarray([0.9, 0.8, 0.2, 0.1]),
            np.asarray([1, 0, 1, 0]),
            0.5,
        )
        self.assertEqual(values["true_positives"], 1)
        self.assertEqual(values["false_positives"], 1)
        self.assertEqual(values["recall"], 0.5)

    def test_threshold_csv_schema_is_exact(self) -> None:
        row = {
            "held_out_fold": "0",
            **threshold_metrics(
                np.asarray([0.9, 0.8, 0.2, 0.1]),
                np.asarray([1, 0, 1, 0]),
                0.5,
            ),
            "selected": True,
        }
        output = threshold_csv_row(row)
        self.assertEqual(set(output), set(THRESHOLD_FIELDS))

    def test_csv_schema_check_rejects_missing_and_extra_fields(self) -> None:
        assert_csv_schema([{"a": "1", "b": "2"}], ["a", "b"], "unit")
        with self.assertRaisesRegex(RuntimeError, "missing=.*b"):
            assert_csv_schema([{"a": "1"}], ["a", "b"], "unit")
        with self.assertRaisesRegex(RuntimeError, "extra=.*c"):
            assert_csv_schema(
                [{"a": "1", "b": "2", "c": "3"}],
                ["a", "b"],
                "unit",
            )

    def test_linear_training_smoke(self) -> None:
        rows = [
            progress_row(index, 100 if index % 3 == 0 else 20)
            for index in range(30)
        ]
        features = online_features(rows)
        targets = far_labels(rows, 75)
        records = np.asarray([f"rec_{index:03d}" for index in range(30)])
        fit = np.arange(20, dtype=np.int32)
        early = np.arange(20, 30, dtype=np.int32)
        cfg = {
            "learning_rate": 0.01,
            "weight_decay": 0.001,
            "batch_size": 8,
            "epochs": 3,
            "early_stopping_patience": 3,
            "gradient_clip": 1.0,
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            model, report = train_model(
                features,
                targets,
                records,
                fit,
                early,
                cfg,
                torch.device("cpu"),
                checkpoint,
                {"scope": "unit_test"},
            )
            probabilities = probabilities_for(
                model,
                features,
                early,
                torch.device("cpu"),
                checkpoint,
            )
        self.assertEqual(probabilities.shape, (10,))
        self.assertTrue(
            np.logical_and(probabilities >= 0, probabilities <= 1).all()
        )
        self.assertGreaterEqual(report["best_epoch"], 1)


if __name__ == "__main__":
    unittest.main()
