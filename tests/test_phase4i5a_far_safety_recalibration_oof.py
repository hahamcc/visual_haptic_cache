from __future__ import annotations

import unittest

import numpy as np

from src.recalibrate_phase4i5a_far_threshold_oof import (
    QUERY_FIELDS,
    classification_for_indices,
    keyed_retrieval_rows,
    parsed_threshold_row,
    select_safety_threshold,
)


def threshold(
    value: float,
    recall: float,
    false_positive_rate: float,
    fold: str = "0",
) -> dict[str, str]:
    return {
        "held_out_fold": fold,
        "threshold": str(value),
        "recall": str(recall),
        "false_positive_rate": str(false_positive_rate),
        "precision": "0.8",
        "risk_rate": "0.3",
        "true_positives": "10",
        "false_positives": "2",
        "true_negatives": "20",
        "false_negatives": "1",
        "selected": "0",
    }


def phase4i3_row(name: str) -> dict[str, str]:
    return {
        "query_record_id": f"record_{name}",
        "query_image_name": name,
        "query_probe": "75",
        "oof_fold": "0",
        "v1_selected_cache_image_name": f"v1_{name}",
        "v1_ranker_oracle_embedding_rank": "2",
        "v1_tactile_diff_mae": "0.02",
        "v1_tactile_ssim": "0.7",
        "v1_tactile_mask_iou": "0.2",
        "phase4i3_selected_cache_image_name": f"phase4i3_{name}",
        "phase4i3_ranker_oracle_embedding_rank": "1",
        "phase4i3_tactile_diff_mae": "0.01",
        "phase4i3_tactile_ssim": "0.8",
        "phase4i3_tactile_mask_iou": "0.3",
    }


class Phase4I5aSafetyRecalibrationTests(unittest.TestCase):
    def test_selects_highest_safe_threshold_with_lowest_fpr(self) -> None:
        rows = [
            threshold(0.5, 0.90, 0.05),
            threshold(0.4, 0.97, 0.08),
            threshold(0.3, 1.00, 0.20),
        ]
        selected = select_safety_threshold(rows, 0.97)
        self.assertEqual(selected["threshold"], 0.4)
        self.assertEqual(selected["recall"], 0.97)

    def test_threshold_parser_preserves_fold_and_counts(self) -> None:
        output = parsed_threshold_row(threshold(0.4, 0.97, 0.08, "2"))
        self.assertEqual(output["held_out_fold"], "2")
        self.assertEqual(output["true_positives"], 10)
        self.assertAlmostEqual(float(output["threshold"]), 0.4)

    def test_rejects_threshold_options_from_multiple_folds(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "mix OOF folds"):
            select_safety_threshold(
                [
                    threshold(0.4, 0.97, 0.08, "0"),
                    threshold(0.3, 1.0, 0.2, "1"),
                ],
                0.97,
            )

    def test_keyed_retrieval_rows_follow_requested_order(self) -> None:
        rows = [phase4i3_row("b"), phase4i3_row("a")]
        output = keyed_retrieval_rows(rows, "v1", ["a", "b"])
        self.assertEqual(
            [row["query_image_name"] for row in output],
            ["a", "b"],
        )
        self.assertEqual(output[0]["selected_cache_image_name"], "v1_a")

    def test_classification_uses_frozen_progress_and_new_risk(self) -> None:
        rows = [
            {
                "true_progress_class": "near",
                "predicted_progress_class": "near",
                "true_far_label": "0",
                "predicted_ttc": "10",
                "true_ttc": "5",
            },
            {
                "true_progress_class": "far",
                "predicted_progress_class": "mid",
                "true_far_label": "1",
                "predicted_ttc": "70",
                "true_ttc": "75",
            },
        ]
        output = classification_for_indices(
            rows,
            np.asarray([False, True]),
            np.asarray([0, 1], dtype=np.int32),
        )
        self.assertEqual(output["far_recall"], 1.0)
        self.assertEqual(output["near_mid_retention"], 1.0)
        self.assertEqual(output["ttc_mae_frames"], 5.0)

    def test_query_schema_has_no_duplicates(self) -> None:
        self.assertEqual(len(QUERY_FIELDS), len(set(QUERY_FIELDS)))


if __name__ == "__main__":
    unittest.main()
