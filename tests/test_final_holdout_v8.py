from __future__ import annotations

import copy
import hashlib
import inspect
import json
from pathlib import Path
import unittest

from corpus_contract import canonical_json_bytes
from freeze_final_holdout_v8 import (
    ARTIFACT_PATH,
    MARKED_DIRECTORY,
    PLAN_PATH,
    build_final_holdout_package,
    validate_review_bound_allowlist,
)
from watermark_toy import inspect_positions, load_lexicon, score_corpus


ROOT = Path(__file__).resolve().parents[1]


class FinalHoldoutV8Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.package = build_final_holdout_package(ROOT / PLAN_PATH, root=ROOT)

    def test_review_classifies_all_411_contexts_before_keyed_encoding(self) -> None:
        inventory = self.package.inventory
        review = self.package.review
        allowlist = self.package.allowlist

        self.assertEqual(inventory["eligibleOccurrences"], 411)
        self.assertEqual(review["reviewedOccurrences"], 411)
        self.assertEqual(review["approvedOccurrences"], 275)
        self.assertEqual(review["rejectedOccurrences"], 136)
        self.assertEqual(allowlist["approvedOccurrences"], 275)
        self.assertEqual(allowlist["rejectedOccurrences"], 136)
        self.assertEqual(len(review["documents"]), 20)
        for document in review["documents"]:
            decisions = document["decisions"]
            self.assertEqual(len(decisions), document["reviewedOccurrences"])
            self.assertEqual(
                {decision["decision"] for decision in decisions},
                {"approved", "rejected"}
                if document["rejectedOccurrences"]
                else {"approved"},
            )
            for decision in decisions:
                if decision["decision"] == "rejected":
                    self.assertIn(
                        decision["criterion"],
                        {
                            "grammar_or_part_of_speech",
                            "meaning_or_reference",
                            "named_label_integrity",
                            "register_or_collocation",
                            "technical_term_integrity",
                        },
                    )
                    self.assertTrue(decision["reason"].strip())

        serialized = canonical_json_bytes(inventory)
        for forbidden in (
            b'"active"',
            b'"densityBps"',
            b'"favoredVariant"',
            b'"keySha256"',
        ):
            self.assertNotIn(forbidden, serialized)

    def test_mask_is_bound_and_tampering_fails(self) -> None:
        validate_review_bound_allowlist(
            self.package.plan,
            self.package.inventory,
            self.package.review,
            self.package.allowlist,
        )

        tampered = copy.deepcopy(self.package.allowlist)
        tampered["documents"][0]["approvedOccurrenceFingerprints"][0] = "0" * 64
        with self.assertRaisesRegex(ValueError, "allowlist|fingerprint"):
            validate_review_bound_allowlist(
                self.package.plan,
                self.package.inventory,
                self.package.review,
                tampered,
            )

        tampered = copy.deepcopy(self.package.allowlist)
        tampered["reviewSha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "reviewSha256"):
            validate_review_bound_allowlist(
                self.package.plan,
                self.package.inventory,
                self.package.review,
                tampered,
            )

    def test_encoder_skips_every_rejected_context_and_full_scorer_passes(self) -> None:
        artifact = self.package.artifact
        gate = artifact["prepaidGate"]
        encoding = artifact["encoding"]

        self.assertEqual(encoding["approvedActiveOpportunities"], 18)
        self.assertEqual(encoding["approvedActiveFavored"], 18)
        self.assertEqual(encoding["rejectedActiveSkipped"], 16)
        self.assertEqual(encoding["physicalChanges"], 7)
        self.assertEqual(encoding["rejectedPhysicalChanges"], 0)
        self.assertTrue(
            all(not row["changed"] for row in encoding["rejectedPositions"])
        )

        self.assertEqual(gate["status"], "passed")
        self.assertEqual(gate["unmarked"]["status"], "not_detected")
        self.assertEqual(gate["unmarked"]["activePositions"], 34)
        self.assertEqual(gate["unmarked"]["hits"], 18)
        self.assertEqual(gate["marked"]["status"], "detected")
        self.assertEqual(gate["marked"]["activePositions"], 34)
        self.assertEqual(gate["marked"]["hits"], 25)
        self.assertEqual(
            gate["marked"]["pValueExact"],
            {"numerator": 9707899, "denominator": 2147483648},
        )

        # The public detector remains byte-for-byte unaware of the corpus mask.
        self.assertNotIn("allowlist", inspect.signature(score_corpus).parameters)
        detector = self.package.plan["detectorImplementation"]
        self.assertFalse(detector["allowlistAware"])
        self.assertEqual(
            hashlib.sha256((ROOT / detector["path"]).read_bytes()).hexdigest(),
            detector["sha256"],
        )

        marker = json.loads((ROOT / "fixtures/experiment-config-v1.json").read_text())[
            "marker"
        ]
        key = bytes.fromhex(marker["keyHex"])
        lexicon = load_lexicon(ROOT / "fixtures/synonym_pairs-v1.json")
        review_by_id = {
            row["documentId"]: row for row in self.package.review["documents"]
        }
        source_manifest = json.loads(
            (ROOT / "corpus/holdout-v6/manifest-v6-holdout.json").read_text()
        )
        for source in source_manifest["documents"]:
            document_id = source["documentId"]
            original_text = (ROOT / source["path"]).read_text()
            marked_text = (ROOT / MARKED_DIRECTORY / f"{document_id}.md").read_text()
            original_positions = inspect_positions(
                original_text,
                key=key,
                document_id=document_id,
                density_bps=1000,
                lexicon=lexicon,
                context_width=4,
            )
            marked_positions = inspect_positions(
                marked_text,
                key=key,
                document_id=document_id,
                density_bps=1000,
                lexicon=lexicon,
                context_width=4,
            )
            self.assertEqual(len(original_positions), len(marked_positions))
            decisions = review_by_id[document_id]["decisions"]
            for decision, before, after in zip(
                decisions, original_positions, marked_positions, strict=True
            ):
                if decision["decision"] == "rejected":
                    self.assertEqual(before.token, after.token)

    def test_1000_wrong_keys_and_every_frozen_output_have_parity(self) -> None:
        artifact = self.package.artifact
        controls = artifact["wrongKeyControlsOnMarked"]
        self.assertEqual(controls["count"], 1000)
        self.assertEqual(
            controls["sufficientCount"] + controls["insufficientCount"],
            1000,
        )
        self.assertEqual(artifact["providerCalls"], 0)
        self.assertEqual(
            artifact["providerExecution"]["status"],
            "blocked_pending_committed_v7_winner_binding",
        )
        for relative_path, expected_bytes in self.package.files.items():
            self.assertEqual((ROOT / relative_path).read_bytes(), expected_bytes)
        self.assertEqual(
            json.loads((ROOT / ARTIFACT_PATH).read_text(encoding="utf-8")),
            artifact,
        )

    def test_all_maintained_json_artifacts_have_evidence_contract(self) -> None:
        for relative_path in self.package.files:
            if not relative_path.endswith(".json"):
                continue
            value = json.loads(self.package.files[relative_path])
            with self.subTest(path=relative_path):
                self.assertEqual(value["schemaVersion"], 1)
                self.assertTrue(value["verifiedAt"])
                self.assertTrue(value["methodology"])
                self.assertTrue(value["sources"])


if __name__ == "__main__":
    unittest.main()
