from __future__ import annotations

import unittest

from src.mine_phase4h_tactile_hard_cases import (
    aggregate_records,
    build_collection_plan,
    classify_query,
    mine_pairs_for_query,
)


class Phase4HTactileHardCaseTests(unittest.TestCase):
    def config(self) -> dict:
        return {
            "mae_harm_epsilon": 0.0,
            "minimum_spatial_iou_gain": 0.01,
            "minimum_aligned_iou_drop": 0.01,
            "minimum_far_ssim_drop": 0.005,
            "far_probe_min": 75,
            "max_hard_pairs_per_query": 4,
            "priority_weights": {
                "hard_negative_pair": 3.0,
                "spatial_gain_mae_harm": 5.0,
                "aligned_projector_harm": 3.0,
                "far_failure": 2.0,
                "triple_win_reference": 1.0,
            },
        }

    def result(self, mae: float, ssim: float, iou: float, rank: int = 1) -> dict[str, str]:
        return {
            "query_record_id": "rec_00001",
            "query_image_name": "q",
            "query_probe": "100",
            "oof_fold": "0",
            "recipe_name": "baseline",
            "selected_cache_record_id": "rec_00002",
            "selected_cache_image_name": "c",
            "pred_x": "10",
            "pred_y": "12",
            "tactile_diff_mae": str(mae),
            "tactile_ssim": str(ssim),
            "tactile_mask_iou": str(iou),
            "ranker_oracle_embedding_rank": str(rank),
        }

    def test_classifies_spatial_gain_intensity_and_projector_harm(self) -> None:
        v1 = self.result(0.009, 0.74, 0.20)
        frozen = self.result(0.010, 0.75, 0.25)
        aligned = self.result(0.011, 0.72, 0.22)
        pair = {
            "candidate_record_id": "rec_00003",
        }
        row = classify_query(v1, frozen, aligned, [pair], self.config())
        self.assertEqual(row["spatial_gain_mae_harm"], "1")
        self.assertEqual(row["aligned_projector_harm"], "1")
        self.assertEqual(row["far_failure"], "1")
        self.assertEqual(row["frozen_strict_triple_win"], "0")
        self.assertIn("spatial_gain_intensity_harm", row["case_types"])

    def test_mines_only_top8_oracle17_32_pairs(self) -> None:
        sample = {
            "target_tip_x": "10",
            "target_tip_y": "10",
            "image_width": "100",
            "image_height": "100",
        }
        samples = {
            "q": sample,
            "hard": {**sample, "target_tip_x": "80"},
            "easy": sample,
            "late": sample,
        }
        base = {
            "query_record_id": "rec_00001",
            "query_image_name": "q",
            "query_probe": "100",
            "oof_fold": "0",
            "recipe_name": "baseline",
            "candidate_score": "0.1",
            "detail_patch_score": "0.1",
            "context_patch_score": "0.1",
            "wide_patch_score": "",
            "candidate_tactile_embedding_distance": "1.0",
            "candidate_tactile_ssim": "0.2",
            "candidate_tactile_mask_iou": "0.1",
        }
        candidates = [
            {
                **base,
                "candidate_rank": "2",
                "candidate_oracle_embedding_rank": "22",
                "candidate_record_id": "rec_00002",
                "candidate_image_name": "hard",
            },
            {
                **base,
                "candidate_rank": "3",
                "candidate_oracle_embedding_rank": "4",
                "candidate_record_id": "rec_00003",
                "candidate_image_name": "easy",
            },
            {
                **base,
                "candidate_rank": "9",
                "candidate_oracle_embedding_rank": "30",
                "candidate_record_id": "rec_00004",
                "candidate_image_name": "late",
            },
        ]
        output = mine_pairs_for_query("q", candidates, samples, 4, 75)
        self.assertEqual(len(output), 1)
        self.assertEqual(output[0]["candidate_image_name"], "hard")
        self.assertGreater(float(output[0]["normalized_contact_offset"]), 0.0)

    def test_collection_plan_requires_new_record_ids(self) -> None:
        query = {
            **self.result(0.009, 0.74, 0.20),
            "far_failure": "1",
            "spatial_gain_mae_harm": "1",
            "aligned_projector_harm": "1",
            "frozen_strict_triple_win": "0",
            "priority_score": "10",
            "case_types": "far_failure",
        }
        pairs = [
            {
                "query_record_id": "rec_00001",
                "candidate_record_id": "rec_00002",
            }
        ]
        records = aggregate_records([query], pairs, 1)
        plan = build_collection_plan(
            records,
            {
                "target_new_records": 2,
                "minimum_contact_regions": 3,
                "repeats_per_region": 3,
                "hard_negative_pairs_per_new_record": 2,
            },
        )
        self.assertEqual(len(plan), 2)
        self.assertEqual(plan[0]["new_record_id"], "")
        self.assertEqual(plan[0]["record_disjoint_required"], "1")
        self.assertEqual(plan[0]["probe_focus"], "75|100")


if __name__ == "__main__":
    unittest.main()
