from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.reserve_phase4i6_raw_candidate_pool import (
    select_candidate_records,
)


def make_record(root: Path, modality: str, split: str, record: str) -> None:
    (root / modality / split / record).mkdir(parents=True, exist_ok=True)


class Phase4I6RawCandidatePoolTests(unittest.TestCase):
    def test_selects_complete_disjoint_range(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index in range(5):
                record = f"rec_{3000 + index:05d}"
                make_record(root, "vision", "3", record)
                make_record(root, "touch", "3", record)
            selected = select_candidate_records(
                root,
                "vision",
                "touch",
                "3",
                1,
                3,
                set(),
                set(),
            )
        self.assertEqual(
            selected,
            ["rec_03001", "rec_03002", "rec_03003"],
        )

    def test_rejects_missing_touch_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_record(root, "vision", "3", "rec_03000")
            (root / "touch" / "3").mkdir(parents=True)
            with self.assertRaisesRegex(RuntimeError, "miss touch"):
                select_candidate_records(
                    root,
                    "vision",
                    "touch",
                    "3",
                    0,
                    1,
                    set(),
                    set(),
                )

    def test_rejects_frozen_record_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_record(root, "vision", "3", "rec_03000")
            make_record(root, "touch", "3", "rec_03000")
            with self.assertRaisesRegex(RuntimeError, "frozen record"):
                select_candidate_records(
                    root,
                    "vision",
                    "touch",
                    "3",
                    0,
                    1,
                    {("3", "rec_03000")},
                    set(),
                )


if __name__ == "__main__":
    unittest.main()
