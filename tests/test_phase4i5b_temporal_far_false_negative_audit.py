from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.audit_phase4i5b_temporal_far_false_negatives import (
    QUERY_FIELDS,
    RECORD_FIELDS,
    SLICE_FIELDS,
    build_slices,
    cosine_distance,
    load_feature_metadata,
    numeric_bin,
    record_rows,
    retrieval_outcome,
    safe_correlation,
    temporal_visual_diagnostics,
)


def audit_row(
    name: str,
    record: str,
    outcome: str,
    changed: int = 1,
) -> dict[str, str]:
    harmful = int(
        outcome
        in ("spatial_only_gain", "mae_or_ssim_harm", "triple_harm")
    )
    return {
        "query_record_id": record,
        "query_image_name": name,
        "query_probe": "100",
        "oof_fold": "0",
        "far_risk_probability": "0.1",
        "far_miss_margin": "0.2",
        "miss_margin_bin": "gt_0.15",
        "temporal_ttc_absolute_error": "15",
        "base_ttc_absolute_error": "25",
        "temporal_minus_base_ttc_absolute_error": "-10",
        "temporal_ttc_error_bin": "10_to_20",
        "visual_first_last_cosine_distance": "0.03",
        "visual_change_quartile": "q1",
        "corridor_approach_reduction": "0.02",
        "approach_direction": "approaching",
        "phase4i3_changed_cache": str(changed),
        "mae_delta": "0.001" if harmful else "-0.001",
        "ssim_delta": "-0.01" if harmful else "0.01",
        "iou_delta": "-0.02" if outcome == "triple_harm" else "0.02",
        "retrieval_outcome": outcome,
        "strict_triple_win": str(int(outcome == "strict_triple_win")),
        "spatial_only_gain": str(int(outcome == "spatial_only_gain")),
        "mae_or_ssim_harm": str(harmful),
        "triple_harm": str(int(outcome == "triple_harm")),
        "high_confidence_false_negative": "1",
    }


class Phase4I5bFalseNegativeAuditTests(unittest.TestCase):
    def test_retrieval_outcomes_are_mutually_exclusive(self) -> None:
        epsilon = 1e-12
        self.assertEqual(
            retrieval_outcome(-0.1, 0.1, 0.1, epsilon),
            "strict_triple_win",
        )
        self.assertEqual(
            retrieval_outcome(0.1, -0.1, -0.1, epsilon),
            "triple_harm",
        )
        self.assertEqual(
            retrieval_outcome(0.1, 0.1, 0.1, epsilon),
            "spatial_only_gain",
        )
        self.assertEqual(
            retrieval_outcome(0.1, 0.1, -0.1, epsilon),
            "mae_or_ssim_harm",
        )
        self.assertEqual(
            retrieval_outcome(0.0, 0.0, 0.0, epsilon),
            "mixed_or_neutral",
        )

    def test_temporal_visual_diagnostics_use_valid_frames(self) -> None:
        visual = np.asarray(
            [
                [1.0, 0.0],
                [100.0, 100.0],
                [0.0, 1.0],
                [-1.0, 0.0],
            ],
            dtype=np.float32,
        )
        geometry = np.zeros((4, 8), dtype=np.float32)
        geometry[:, 6] = np.asarray([0.5, 0.4, 0.3, 0.2])
        valid = np.asarray([1, 0, 1, 1], dtype=np.float32)
        padding = np.asarray([0.0, 1.0, 0.2, 0.4], dtype=np.float32)
        output = temporal_visual_diagnostics(
            visual,
            geometry,
            valid,
            padding,
        )
        self.assertAlmostEqual(
            output["visual_first_last_cosine_distance"],
            2.0,
        )
        self.assertAlmostEqual(
            output["corridor_approach_reduction"],
            0.3,
        )
        self.assertAlmostEqual(output["temporal_padding_ratio"], 0.2)

    def test_cosine_and_bins(self) -> None:
        self.assertEqual(
            cosine_distance(
                np.asarray([1.0, 0.0]),
                np.asarray([1.0, 0.0]),
            ),
            0.0,
        )
        self.assertEqual(numeric_bin(0.05, 0.05, 0.15), "le_0.05")
        self.assertEqual(numeric_bin(0.1, 0.05, 0.15), "0.05_to_0.15")
        self.assertEqual(numeric_bin(0.2, 0.05, 0.15), "gt_0.15")

    def test_slice_and_record_aggregation(self) -> None:
        rows = [
            audit_row("a", "rec_a", "triple_harm"),
            audit_row("b", "rec_a", "strict_triple_win"),
            audit_row("c", "rec_b", "mixed_or_neutral", changed=0),
        ]
        slices = build_slices(rows)
        self.assertEqual(slices[0]["queries"], "3")
        records = record_rows(rows, 10)
        self.assertEqual(records[0]["query_record_id"], "rec_a")
        self.assertEqual(records[0]["triple_harm_queries"], "1")
        self.assertEqual(records[0]["queries"], "2")

    def test_feature_metadata_rejects_future_input(self) -> None:
        metadata = {
            "query_true_probe_feature_used": False,
            "query_tactile_input": False,
            "future_visual_frames_used": False,
        }
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / "features"
            Path(f"{prefix}.json").write_text(
                json.dumps(metadata),
                encoding="utf-8",
            )
            loaded = load_feature_metadata(prefix)
            self.assertFalse(loaded["future_visual_frames_used"])
            metadata["future_visual_frames_used"] = True
            Path(f"{prefix}.json").write_text(
                json.dumps(metadata),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "online contract"):
                load_feature_metadata(prefix)

    def test_safe_correlation_handles_constant_input(self) -> None:
        self.assertIsNone(
            safe_correlation(np.ones(3), np.asarray([1.0, 2.0, 3.0]))
        )
        self.assertAlmostEqual(
            safe_correlation(
                np.asarray([1.0, 2.0, 3.0]),
                np.asarray([2.0, 4.0, 6.0]),
            ),
            1.0,
        )

    def test_output_schemas_have_no_duplicates(self) -> None:
        for fields in (QUERY_FIELDS, SLICE_FIELDS, RECORD_FIELDS):
            self.assertEqual(len(fields), len(set(fields)))


if __name__ == "__main__":
    unittest.main()
