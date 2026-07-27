from __future__ import annotations

import unittest
from collections import Counter

import numpy as np

from src.build_phase4i6f_ttc_increment import (
    approved_pair_records,
    assign_grouped_folds,
    select_diverse_fill,
)


def row(record_id: str) -> dict[str, str]:
    return {"record_id": record_id}


class Phase4I6fTTCIncrementTests(unittest.TestCase):
    def test_approved_pairs_are_disjoint(self) -> None:
        pair_rows = [
            {
                "planned_pair_id": "pair_001",
                "record_a": "a",
                "record_b": "b",
            },
            {
                "planned_pair_id": "pair_002",
                "record_a": "c",
                "record_b": "d",
            },
        ]
        selected, pairs = approved_pair_records(
            pair_rows,
            ["pair_002"],
        )
        self.assertEqual(selected[0]["planned_pair_id"], "pair_002")
        self.assertEqual(pairs, {"pair_002": ("c", "d")})
        pair_rows[1]["record_a"] = "a"
        with self.assertRaisesRegex(RuntimeError, "reuse"):
            approved_pair_records(
                pair_rows,
                ["pair_001", "pair_002"],
            )

    def test_diverse_fill_preserves_seed_and_target(self) -> None:
        rows = [row(chr(ord("a") + index)) for index in range(6)]
        assignments = {
            "speed_bin": np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)
        }
        standardized = np.arange(6, dtype=np.float32)[:, None]
        selected = select_diverse_fill(
            rows,
            {"a", "b"},
            target_records=4,
            assignments=assignments,
            standardized=standardized,
            quantile_bins=3,
            distance_weight=0.2,
        )
        record_ids = {rows[index]["record_id"] for index in selected}
        self.assertEqual(len(record_ids), 4)
        self.assertTrue({"a", "b"}.issubset(record_ids))

    def test_pair_grouped_folds_are_balanced(self) -> None:
        rows = [row(chr(ord("a") + index)) for index in range(12)]
        group_by_record = {
            "a": "pair_1",
            "b": "pair_1",
            "c": "pair_2",
            "d": "pair_2",
            **{
                chr(ord("a") + index): f"single_{index}"
                for index in range(4, 12)
            },
        }
        assignments = {
            "speed_bin": np.asarray(
                [0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1],
                dtype=np.int64,
            )
        }
        folds = assign_grouped_folds(
            rows,
            group_by_record,
            assignments,
            fold_count=3,
        )
        self.assertEqual(folds["a"], folds["b"])
        self.assertEqual(folds["c"], folds["d"])
        counts = Counter(folds.values())
        self.assertEqual(sorted(counts.values()), [4, 4, 4])


if __name__ == "__main__":
    unittest.main()
