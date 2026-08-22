import unittest

import curated_percent_eval as evaluation


class CuratedPercentEvalTests(unittest.TestCase):
    def test_cost_components_add_without_rounding_hidden_spend(self):
        values = [0.06909984, 0.13408434129422703, 0.39284814]

        self.assertAlmostEqual(sum(values), 0.596032321294227)

    def test_pipeline_failure_policy_is_zero_removal_not_false_success(self):
        self.assertEqual(
            evaluation.pipeline_failure_watermark_removal_percent(),
            0.0,
        )

    def test_watermark_removal_percent_uses_clean_baseline_and_clips(self):
        self.assertAlmostEqual(
            evaluation.watermark_removal_percent(
                source_mean=0.57,
                candidate_mean=0.535,
                clean_mean=0.50,
            ),
            50.0,
        )
        self.assertEqual(
            evaluation.watermark_removal_percent(
                source_mean=0.57, candidate_mean=0.49, clean_mean=0.50
            ),
            100.0,
        )
        self.assertEqual(
            evaluation.watermark_removal_percent(
                source_mean=0.57, candidate_mean=0.58, clean_mean=0.50
            ),
            0.0,
        )

    def test_lower_median_preserves_ten_percent_grid(self):
        self.assertEqual(evaluation.lower_median_percent([40, 50, 80, 90]), 50)
        self.assertEqual(evaluation.lower_median_percent([70, 80, 90]), 80)

    def test_parse_panel_candidate_requires_exact_claims_and_ten_percent_grid(self):
        expected = [f"c{index:02d}" for index in range(1, 11)]
        candidate = {
            "candidateId": "candidate-01",
            "claims": [
                {"id": claim_id, "status": "preserved"} for claim_id in expected
            ],
            "materialErrors": [],
            "readabilityPercent": 90,
            "usabilityPercent": 80,
        }

        parsed = evaluation.validate_panel_candidate(
            candidate,
            expected_candidate_id="candidate-01",
            expected_claim_ids=expected,
        )
        self.assertEqual(parsed["readabilityPercent"], 90)

        candidate["readabilityPercent"] = 85
        with self.assertRaisesRegex(ValueError, "10-percent grid"):
            evaluation.validate_panel_candidate(
                candidate,
                expected_candidate_id="candidate-01",
                expected_claim_ids=expected,
            )

    def test_aggregate_uses_majority_claim_vote_and_conservative_median(self):
        claim_ids = [f"c{index:02d}" for index in range(1, 11)]
        verdicts = []
        for judge_index in range(4):
            verdicts.append(
                {
                    "claims": [
                        {
                            "id": claim_id,
                            "status": (
                                "changed"
                                if claim_id == "c10" and judge_index < 2
                                else "preserved"
                            ),
                        }
                        for claim_id in claim_ids
                    ],
                    "readabilityPercent": [90, 80, 100, 70][judge_index],
                    "usabilityPercent": [80, 70, 90, 60][judge_index],
                }
            )

        result = evaluation.aggregate_pair_panel(verdicts, claim_ids=claim_ids)

        self.assertEqual(result["claimPreservationPercent"], 90)
        self.assertEqual(result["readabilityPercent"], 80)
        self.assertEqual(result["usabilityPercent"], 70)
        self.assertEqual(result["claimResults"][-1]["preservedVotes"], 2)

    def test_final_summary_averages_percentages_and_keeps_exact_counts(self):
        rows = []
        for index in range(10):
            rows.append(
                {
                    "claimPreservationPercent": 90,
                    "detectedAtThreshold": index < 3,
                    "finalPercent": 70,
                    "method": "m",
                    "pipelineCompleted": index != 9,
                    "pspPercent": 95,
                    "readabilityPercent": 80,
                    "usabilityPercent": 70,
                    "watermarkRemovalPercent": 100,
                }
            )

        summary = evaluation.summarize_final_pairs(rows, ["m"])["m"]

        self.assertEqual(summary["documentCount"], 10)
        self.assertEqual(summary["detectorRemovedCount"], 7)
        self.assertEqual(summary["pipelineCompletedCount"], 9)
        self.assertEqual(summary["finalPercent"], 70)

    def test_panel_split_preserves_blind_mapping_and_makes_single_candidates(self):
        source = {
            "batches": [
                {
                    "batchId": "doc-01",
                    "candidates": [
                        {"candidateId": "candidate-01", "text": "A"},
                        {"candidateId": "candidate-02", "text": "B"},
                    ],
                    "claims": [],
                    "documentId": "doc-01",
                    "sourceText": "S",
                    "sourceTextSha256": "s",
                }
            ],
            "blindMap": {
                "doc-01::candidate-01": "doc-01::a",
                "doc-01::candidate-02": "doc-01::b",
            },
            "sources": [],
        }

        result = evaluation.split_panel_input_one_candidate(source, label="xai")

        self.assertEqual(len(result["batches"]), 2)
        self.assertTrue(all(len(batch["candidates"]) == 1 for batch in result["batches"]))
        self.assertEqual(set(result["blindMap"].values()), {"doc-01::a", "doc-01::b"})

    def test_panel_combine_unions_calls_and_uses_cumulative_split_cost(self):
        common_input = {"batches": [], "blindMap": {}, "sources": []}
        batched_output = {"calls": {"a": {}}, "sources": []}
        split_output = {"calls": {"b": {}}, "sources": [], "totalCostUsd": "0.3"}

        combined_input, combined_output = evaluation.combine_panel_artifacts(
            batched_input=common_input,
            split_input=common_input,
            batched_output=batched_output,
            split_output=split_output,
        )

        self.assertEqual(set(combined_output["calls"]), {"a", "b"})
        self.assertEqual(combined_output["totalCostUsd"], "0.3")


class MergeSingleCandidatePanelsTest(unittest.TestCase):
    def _input(self):
        return {
            "batches": [
                {"batchId": "doc-01-s-candidate-01", "candidates": [{"candidateId": "candidate-01", "text": "a"}]},
                {"batchId": "doc-01-s-candidate-02", "candidates": [{"candidateId": "candidate-02", "text": "b"}]},
            ],
            "blindMap": {},
            "sources": [],
        }

    def _output(self, models, cost):
        calls = {}
        for batch in self._input()["batches"]:
            for model in models:
                calls[f"{batch['batchId']}::{model}"] = {
                    "batchId": batch["batchId"],
                    "candidates": [{"candidateId": batch["candidates"][0]["candidateId"]}],
                    "judge": model,
                }
        return {"calls": calls, "totalCostUsd": cost, "status": "complete"}

    def test_merge_unions_disjoint_judges_and_sums_cost(self):
        merged = evaluation.merge_single_candidate_panels(
            panel_input=self._input(),
            outputs=[self._output(["a/x"], "0.10"), self._output(["b/y", "c/z"], "0.25")],
            judge_models=["a/x", "b/y", "c/z"],
        )
        self.assertEqual(len(merged["calls"]), 6)
        self.assertEqual(merged["totalCostUsd"], "0.35")

    def test_merge_rejects_overlap_and_missing_judges(self):
        with self.assertRaises(ValueError):
            evaluation.merge_single_candidate_panels(
                panel_input=self._input(),
                outputs=[self._output(["a/x"], "0.1"), self._output(["a/x"], "0.1")],
                judge_models=["a/x"],
            )
        with self.assertRaises(ValueError):
            evaluation.merge_single_candidate_panels(
                panel_input=self._input(),
                outputs=[self._output(["a/x"], "0.1")],
                judge_models=["a/x", "b/y"],
            )


if __name__ == "__main__":
    unittest.main()
