from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from corpus_contract import (
    build_context_inventory,
    build_manifest,
    canonical_json_bytes,
    inspect_corpus,
    load_corpus_plan,
    validate_context_reviews,
)
from watermark_toy import SynonymLexicon


class CorpusContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.lexicon = SynonymLexicon.from_pairs(
            [["big", "large"], ["small", "little"], ["begin", "start"]]
        )

    def make_plan(self) -> dict[str, object]:
        return {
            "corpusVersion": "test-corpus-v1",
            "documentContract": {
                "count": 2,
                "maxWords": 100,
                "minEligiblePositionsPerDocument": 2,
                "minWords": 10,
            },
            "documents": [
                {
                    "documentId": "doc-a",
                    "genre": "test note",
                    "path": "corpus/original/doc-a.md",
                },
                {
                    "documentId": "doc-b",
                    "genre": "test memo",
                    "path": "corpus/original/doc-b.md",
                },
            ],
            "methodology": "Original short test documents for contract tests.",
            "schemaVersion": 1,
            "sources": [{"title": "Test source", "url": "https://example.com/test"}],
            "verifiedAt": "2026-08-16",
        }

    def write_fixture(
        self,
        root: Path,
        *,
        left: str | None = None,
        right: str | None = None,
    ) -> Path:
        source_root = root / "corpus" / "original"
        source_root.mkdir(parents=True)
        (source_root / "doc-a.md").write_text(
            left
            or "# First\n\nA big plan can start with a small test and stay clear.\n",
            encoding="utf-8",
            newline="\n",
        )
        (source_root / "doc-b.md").write_text(
            right
            or (
                "# Second\n\nA large plan may begin with a little check at "
                "https://example.com/demo.\n"
            ),
            encoding="utf-8",
            newline="\n",
        )
        plan_path = root / "plan.json"
        plan_path.write_bytes(canonical_json_bytes(self.make_plan()))
        return plan_path

    def test_valid_corpus_builds_deterministic_manifest_and_key_free_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan_path = self.write_fixture(root)
            plan = load_corpus_plan(plan_path)
            documents = inspect_corpus(root, plan=plan, lexicon=self.lexicon)
            manifest = build_manifest(
                plan=plan,
                plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
                lexicon=self.lexicon,
                documents=documents,
            )
            inventory = build_context_inventory(
                plan=plan,
                lexicon=self.lexicon,
                documents=documents,
            )

        self.assertEqual(len(documents), 2)
        self.assertEqual(manifest["documentCount"], 2)
        self.assertGreaterEqual(manifest["eligiblePositions"], 4)
        self.assertEqual(
            inventory["documents"][0]["reviewStatus"],
            "pending_manual_context_review",
        )
        self.assertEqual(
            canonical_json_bytes(manifest),
            canonical_json_bytes(json.loads(canonical_json_bytes(manifest))),
        )
        self.assertNotIn(b'"keySha256"', canonical_json_bytes(inventory))

    def test_plan_rejects_count_mismatch_and_unsafe_path(self) -> None:
        for mutate, expected in (
            (
                lambda plan: plan["documentContract"].update({"count": 3}),
                "document count",
            ),
            (
                lambda plan: plan["documents"][0].update({"path": "../escape.md"}),
                "invalid corpus document path",
            ),
        ):
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as temporary:
                plan = self.make_plan()
                mutate(plan)
                path = Path(temporary) / "plan.json"
                path.write_bytes(canonical_json_bytes(plan))
                with self.assertRaisesRegex(ValueError, expected):
                    load_corpus_plan(path)

    def test_surface_rejects_non_reserved_email_and_prohibited_dash(self) -> None:
        invalid_documents = (
            (
                "# First\n\nA big plan can start with a small test for person@real.test.\n",
                "non-reserved email domain",
            ),
            (
                "# First\n\nA big plan can start with a small test — and stay clear.\n",
                "prohibited dash",
            ),
        )
        for left, expected in invalid_documents:
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                plan_path = self.write_fixture(root, left=left)
                plan = load_corpus_plan(plan_path)
                with self.assertRaisesRegex(ValueError, expected):
                    inspect_corpus(root, plan=plan, lexicon=self.lexicon)

    def test_context_reviews_are_bound_to_every_exact_inventory_document(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan_path = self.write_fixture(root)
            plan = load_corpus_plan(plan_path)
            documents = inspect_corpus(root, plan=plan, lexicon=self.lexicon)
            inventory = build_context_inventory(
                plan=plan,
                lexicon=self.lexicon,
                documents=documents,
            )
        inventory_sha256 = hashlib.sha256(canonical_json_bytes(inventory)).hexdigest()
        review = {
            "corpusVersion": inventory["corpusVersion"],
            "documents": [
                {
                    "decision": "approved",
                    "documentId": document["documentId"],
                    "documentSha256": document["sha256"],
                    "eligibleOccurrences": document["eligibleOccurrences"],
                    "findings": [],
                    "reviewedOccurrences": document["eligibleOccurrences"],
                }
                for document in inventory["documents"]
            ],
            "inventorySha256": inventory_sha256,
            "lexiconSha256": inventory["lexiconSha256"],
            "methodology": "Reviewed every candidate occurrence in its supplied context.",
            "reviewVersion": "test-review-v1",
            "reviewer": "independent-test-reviewer",
            "schemaVersion": 1,
            "sources": [{"title": "Test source", "url": "https://example.com/test"}],
            "verifiedAt": "2026-08-16",
        }
        approval = validate_context_reviews(inventory=inventory, reviews=[review])
        self.assertEqual(approval["approvedDocumentCount"], 2)

        broken = json.loads(json.dumps(review))
        broken["documents"][0]["reviewedOccurrences"] -= 1
        with self.assertRaisesRegex(ValueError, "incomplete"):
            validate_context_reviews(inventory=inventory, reviews=[broken])

        broken = json.loads(json.dumps(review))
        broken["inventorySha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "inventorySha256"):
            validate_context_reviews(inventory=inventory, reviews=[broken])


if __name__ == "__main__":
    unittest.main()
