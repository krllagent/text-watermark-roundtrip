import json
import tempfile
import unittest
from pathlib import Path

import run_curated_panel as panel


def _batch(candidate_count=2):
    return {
        "batchId": "doc-01-canary",
        "documentId": "doc-01",
        "sourceText": "Source report.",
        "claims": [
            {"id": f"c{index:02d}", "text": f"Claim {index}."}
            for index in range(1, 11)
        ],
        "candidates": [
            {"candidateId": f"candidate-{index:02d}", "text": f"Candidate {index}."}
            for index in range(1, candidate_count + 1)
        ],
    }


class CuratedPanelTests(unittest.TestCase):
    def test_resume_filter_can_exclude_a_replaced_judge(self):
        allowed = {"vendor/a", "vendor/b"}
        previous = {
            "a": {"candidates": [], "judge": "vendor/a"},
            "old": {"candidates": [], "judge": "vendor/old"},
        }

        kept = {
            key: value
            for key, value in previous.items()
            if "candidates" in value and value["judge"] in allowed
        }

        self.assertEqual(set(kept), {"a"})

    def test_lost_paid_response_is_closed_at_reserved_ceiling(self):
        state = {
            "calls": {},
            "inFlight": {
                "doc-01::vendor/model": {
                    "ceilingUsd": "0.02",
                    "requestSha256": "a" * 64,
                }
            },
            "priorCostUsd": "0.01",
        }

        result = panel.resolve_lost_paid_responses(state)

        self.assertEqual(result["inFlight"], {})
        self.assertEqual(result["status"], "batch_failed")
        self.assertEqual(result["totalCostUsd"], "0.03")
        self.assertTrue(result["calls"]["doc-01::vendor/model"]["costIsUpperBound"])

    def test_prior_ledger_subtracts_only_resumable_valid_calls(self):
        checkpoint = {
            "calls": {
                "valid": {"candidates": [], "costUsd": "0.2", "judge": "keep"},
                "other": {"candidates": [], "costUsd": "0.3", "judge": "replace"},
                "lost": {"costUsd": "0.1", "judge": "keep", "terminalError": {}},
            },
            "sources": [],
            "totalCostUsd": "0.7",
        }

        result = panel.make_prior_ledger(checkpoint, reusable_models=["keep"])

        self.assertEqual(result["reusableCostUsd"], "0.2")
        self.assertEqual(result["totalCostUsd"], "0.5")

    def test_prompt_freezes_ten_percent_grid_and_hides_method_names(self):
        prompt = panel.build_prompt(_batch())

        self.assertIn(
            "0, 10, 20, 30, 40, 50, 60, 70, 80, 90, or 100",
            " ".join(prompt.split()),
        )
        self.assertNotIn("DIPPER", prompt)
        self.assertNotIn("roundtrip", prompt)

    def test_schema_has_exact_dynamic_candidates_and_claims(self):
        schema = panel.response_format(_batch(candidate_count=5))
        candidate = schema["json_schema"]["schema"]["properties"]["candidates"]["items"]

        self.assertEqual(candidate["properties"]["candidateId"]["enum"], [
            "candidate-01", "candidate-02", "candidate-03", "candidate-04", "candidate-05"
        ])
        claims = candidate["properties"]["claims"]
        self.assertEqual(claims["minItems"], 10)
        self.assertEqual(claims["maxItems"], 10)
        self.assertIn("c01", claims["items"]["properties"]["id"]["enum"])
        self.assertNotIn(
            "maxLength",
            candidate["properties"]["materialErrors"]["items"],
        )

    def test_parse_response_rejects_non_grid_percentage(self):
        batch = _batch(candidate_count=1)
        payload = {
            "candidates": [
                {
                    "candidateId": "candidate-01",
                    "claims": [
                        {"id": f"c{index:02d}", "status": "preserved"}
                        for index in range(1, 11)
                    ],
                    "materialErrors": [],
                    "readabilityPercent": 85,
                    "usabilityPercent": 100,
                }
            ]
        }

        with self.assertRaisesRegex(ValueError, "10-percent grid"):
            panel.parse_response(json.dumps(payload), batch)

    def test_canary_contains_identical_and_five_targeted_tampers(self):
        source = (
            "twelve library employees. On May 16th, 2023. $7,500. "
            "from May 16th to June 30th. continue utilizing the digital visitor log"
        )
        claims = [
            {"id": f"c{index:02d}", "text": f"Claim {index}."}
            for index in range(1, 11)
        ]
        batch, expected = panel.build_canary_batch(source=source, claims=claims)

        self.assertEqual(batch["candidates"][0]["text"], source)
        self.assertIn("$75,000", batch["candidates"][1]["text"])
        self.assertEqual(expected["tamperedClaimIds"], ["c03", "c05", "c07", "c08", "c10"])


    def test_single_canary_has_identical_tampered_and_empty_prompts(self):
        source = (
            "twelve library employees. On May 16th, 2023. $7,500. "
            "from May 16th to June 30th. continue utilizing the digital visitor log"
        )
        claims = [
            {"id": f"c{index:02d}", "text": f"Claim {index}."}
            for index in range(1, 11)
        ]
        batches, expected = panel.build_single_canary_batches(source=source, claims=claims)

        self.assertEqual([b["batchId"] for b in batches], [
            "doc-01-canary-identical", "doc-01-canary-tampered", "doc-01-canary-empty",
        ])
        self.assertTrue(all(len(b["candidates"]) == 1 for b in batches))
        self.assertEqual(batches[0]["candidates"][0]["text"], source)
        self.assertIn("$75,000", batches[1]["candidates"][0]["text"])
        self.assertEqual(batches[2]["candidates"][0]["text"], "")
        self.assertEqual(expected["tamperedClaimIds"], ["c03", "c05", "c07", "c08", "c10"])

    def test_single_canary_validation_fails_judge_that_rewards_empty_text(self):
        source = (
            "twelve library employees. On May 16th, 2023. $7,500. "
            "from May 16th to June 30th. continue utilizing the digital visitor log"
        )
        claims = [
            {"id": f"c{index:02d}", "text": f"Claim {index}."}
            for index in range(1, 11)
        ]
        batches, expected = panel.build_single_canary_batches(source=source, claims=claims)
        panel_input = {"batches": batches, "expected": expected}
        judges = [{"model": "good/judge"}, {"model": "bad/judge"}]

        def verdict(statuses, read, usab):
            return {
                "candidateId": "candidate-01",
                "claims": [{"id": f"c{i:02d}", "status": statuses[i - 1]} for i in range(1, 11)],
                "readabilityPercent": read,
                "usabilityPercent": usab,
                "materialErrors": [],
            }

        preserved = ["preserved"] * 10
        tampered = [
            "changed" if f"c{i:02d}" in expected["tamperedClaimIds"] else "preserved"
            for i in range(1, 11)
        ]
        missing = ["missing"] * 10
        calls = {}
        for judge in judges:
            calls[f"doc-01-canary-identical::{judge['model']}"] = {
                "candidates": [verdict(preserved, 100, 100)]
            }
            calls[f"doc-01-canary-tampered::{judge['model']}"] = {
                "candidates": [verdict(tampered, 90, 80)]
            }
        calls["doc-01-canary-empty::good/judge"] = {"candidates": [verdict(missing, 0, 0)]}
        calls["doc-01-canary-empty::bad/judge"] = {"candidates": [verdict(preserved, 100, 100)]}

        result = panel.validate_canary(panel_input, calls, judges)

        by_judge = {row["judge"]: row for row in result["judges"]}
        self.assertTrue(by_judge["good/judge"]["passed"])
        self.assertFalse(by_judge["bad/judge"]["passed"])
        self.assertEqual(by_judge["bad/judge"]["emptyPreservedCount"], 10)
        self.assertFalse(result["passed"])


    def test_judge_reasoning_effort_defaults_to_panel_and_honours_override(self):
        config = {"panel": {"reasoningEffort": "none"}}
        self.assertEqual(panel.judge_reasoning_effort(config, {"model": "a"}), "none")
        self.assertEqual(
            panel.judge_reasoning_effort(config, {"model": "b", "reasoningEffort": "low"}),
            "low",
        )
        self.assertEqual(panel.judge_reasoning_effort({}, {"model": "c"}), "none")


if __name__ == "__main__":
    unittest.main()
