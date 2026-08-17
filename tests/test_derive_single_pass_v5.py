from __future__ import annotations

import copy
import json
from pathlib import Path
import unittest

from corpus_contract import canonical_json_bytes
from derive_single_pass_v5 import (
    build_single_pass_v5_result,
    load_single_pass_v5_config,
)
from unmark import build_v4_draft_request


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "fixtures" / "verified-paraphrase-config-v5-single-pass.json"
RESULT_PATH = ROOT / "results" / "verified-paraphrase-derived-v5-single-pass.json"


class SinglePassV5Tests(unittest.TestCase):
    def test_config_freezes_one_stage_and_marks_holdout_post_hoc(self) -> None:
        config = load_single_pass_v5_config(CONFIG_PATH)

        self.assertEqual(config.method_id, "paraphrase-v5-single-pass")
        self.assertEqual(config.call_graph, ("paraphrase-draft",))
        self.assertTrue(config.post_hoc_simplification)
        self.assertFalse(config.pre_registered_holdout)
        self.assertEqual(config.holdout_interpretation, "descriptive_only")
        self.assertEqual(config.model, "qwen/qwen3.6-35b-a3b")
        self.assertEqual(config.provider_order, ("deepinfra/fp8",))
        self.assertEqual(config.expected_response_providers, ("DeepInfra",))
        self.assertFalse(config.allow_fallbacks)
        self.assertEqual(config.data_collection, "deny")
        self.assertTrue(config.zdr)
        self.assertEqual(config.max_tokens, 4096)

    def test_parity_fixture_is_byte_exact_for_demo_single_stage(self) -> None:
        config = load_single_pass_v5_config(CONFIG_PATH)
        fixture = json.loads(config.parity_fixture_path.read_text())
        request = build_v4_draft_request(fixture["sample"]["sourceMaskedText"])

        self.assertEqual(fixture["request"]["stage"], "paraphrase-draft")
        self.assertEqual(fixture["request"]["messages"], list(request.to_messages()))
        self.assertEqual(fixture["request"]["maxTokens"], 4096)
        self.assertEqual(fixture["request"]["model"], config.model)
        self.assertEqual(
            fixture["request"]["provider"]["order"],
            ["deepinfra/fp8"],
        )
        self.assertFalse(fixture["request"]["provider"]["allowFallbacks"])
        self.assertEqual(fixture["request"]["provider"]["dataCollection"], "deny")
        self.assertTrue(fixture["request"]["provider"]["zdr"])
        self.assertTrue(fixture["request"]["provider"]["requireParameters"])

    def test_derivation_proves_identity_and_recomputes_all_metrics(self) -> None:
        config = load_single_pass_v5_config(CONFIG_PATH)

        result = build_single_pass_v5_result(config)

        self.assertEqual(result["documentCount"], 20)
        proof = result["derivationProof"]
        self.assertEqual(proof["documentCount"], 20)
        self.assertTrue(proof["allDraftCallsAccepted"])
        self.assertTrue(proof["allDraftsByteIdenticalToV4FinalMaskedText"])
        self.assertTrue(proof["allRestoredOutputsByteIdenticalToV4OutputText"])
        self.assertTrue(proof["allBlindAuditCandidatesByteIdentical"])
        self.assertEqual(proof["transferredBlindReviewCount"], 20)
        self.assertTrue(
            all(
                document["identityProof"]["draftEqualsV4FinalMaskedText"]
                for document in result["documents"]
            )
        )
        self.assertTrue(
            all(
                document["identityProof"]["restoredOutputEqualsV4OutputText"]
                for document in result["documents"]
            )
        )

        development = result["cohorts"]["development"]
        holdout = result["cohorts"]["holdoutDescriptive"]
        self.assertEqual(development["documentIds"], ["doc-01"])
        self.assertEqual(holdout["documentCount"], 19)
        self.assertEqual(development["detector"]["status"], "insufficient_evidence")
        self.assertEqual(holdout["detector"]["status"], "not_detected")
        self.assertAlmostEqual(
            development["meanNormalizedWordDistance"],
            0.2911392405063291,
        )
        self.assertAlmostEqual(
            holdout["meanNormalizedWordDistance"],
            0.43142208115768443,
        )
        for cohort in (development, holdout):
            self.assertEqual(cohort["protectedTokenFailureCount"], 0)
            self.assertEqual(cohort["pipelineFailureCount"], 0)
            self.assertEqual(cohort["semanticFidelityFailureCount"], 0)
            self.assertEqual(cohort["totalFailureCount"], 0)

        usage = result["draftOnlyActualUsage"]
        self.assertEqual(usage["callCount"], 20)
        self.assertEqual(usage["providerCostCredits"], "0.01754920")
        self.assertEqual(usage["providerCostCreditsPer1000Documents"], "0.87746000")
        self.assertEqual(usage["latencyMsMedian"], 9361.68)
        self.assertEqual(usage["latencyMsP95NearestRank"], 16009.373)

        gate = result["decisionGate"]
        self.assertEqual(gate["status"], "pass_exploratory_post_hoc")
        self.assertTrue(gate["passed"])
        self.assertFalse(gate["confirmatoryHoldoutClaimAllowed"])

    def test_derivation_rejects_any_v4_draft_final_divergence(self) -> None:
        config = load_single_pass_v5_config(CONFIG_PATH)
        raw = json.loads(config.v4_result_path.read_text())
        broken = copy.deepcopy(raw)
        broken["methods"][0]["documents"][0]["transformationOutcome"][
            "rawFinalMaskedText"
        ] += " changed"

        with self.assertRaisesRegex(ValueError, "draft/final identity"):
            build_single_pass_v5_result(config, v4_result=broken)

    def test_derivation_rejects_audit_not_bound_to_exact_v4_candidate(self) -> None:
        config = load_single_pass_v5_config(CONFIG_PATH)
        audit = json.loads(config.audit_result_path.read_text())
        broken = copy.deepcopy(audit)
        broken["sourceArtifactSha256"] = "0" * 64

        with self.assertRaisesRegex(ValueError, "audit source artifact"):
            build_single_pass_v5_result(config, audit_result=broken)

    def test_committed_result_is_canonical_and_byte_reproducible(self) -> None:
        config = load_single_pass_v5_config(CONFIG_PATH)

        rebuilt = canonical_json_bytes(build_single_pass_v5_result(config))

        self.assertEqual(rebuilt, RESULT_PATH.read_bytes())


if __name__ == "__main__":
    unittest.main()
