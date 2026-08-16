from __future__ import annotations

from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from corpus_contract import canonical_json_bytes
from run_corpus_controls import (
    ARTIFACT_PATH,
    MARKED_MANIFEST_PATH,
    ControlSpec,
    build_corpus_controls,
    check_control_outputs,
    control_spec_from_config,
    load_control_inputs,
    main,
    write_control_outputs,
)
from watermark_toy import SynonymLexicon


class CorpusControlsFixture:
    def __init__(self) -> None:
        self.lexicon = SynonymLexicon.from_pairs(
            [["big", "large"], ["small", "little"], ["begin", "start"]]
        )
        self.key = bytes.fromhex("11" * 32)
        documents = []
        repeated = "big large small little begin start " * 240
        for index in range(2):
            document_id = f"doc-{index + 1:02d}"
            text = f"# Test Document {index + 1}\n\n{repeated.strip()}.\n"
            raw = text.encode("utf-8")
            documents.append(
                SimpleNamespace(
                    document_id=document_id,
                    eligible_positions=1_440,
                    genre="test",
                    path=f"corpus/original/{document_id}.md",
                    protected_span_count=0,
                    sha256=hashlib.sha256(raw).hexdigest(),
                    text=text,
                    title=f"Test Document {index + 1}",
                    word_count=1_443,
                )
            )

        self.config = SimpleNamespace(
            context_width=4,
            density_bps=1_000,
            experiment_version="test-experiment-v1",
            key=self.key,
            min_active_positions=7,
            raw={
                "marker": {
                    "contextWidth": 4,
                    "densitiesBps": [500, 1_000, 2_000],
                    "keyHex": self.key.hex(),
                    "mainDensityBps": 1_000,
                    "minActivePositions": 7,
                    "wrongKeyCount": 3,
                    "wrongKeySeedHex": (b"tiny-test-wrong-key-seed").hex(),
                }
            },
            sha256="a" * 64,
            sources=(
                {"title": "Test source", "url": "https://example.com/source"},
            ),
            verified_at="2026-08-16",
        )
        inventory_sha256 = "c" * 64
        self.corpus = SimpleNamespace(
            config=self.config,
            documents=tuple(documents),
            inventory_sha256=inventory_sha256,
            lexicon=self.lexicon,
            lexicon_file_sha256="d" * 64,
            manifest={"corpusVersion": "test-corpus-v1"},
            manifest_sha256="b" * 64,
            review_approval={
                "approvedDocumentCount": len(documents),
                "corpusVersion": "test-corpus-v1",
                "inventorySha256": inventory_sha256,
                "lexiconSha256": self.lexicon.sha256,
                "reviewVersions": ["test-review-a", "test-review-b"],
                "schemaVersion": 1,
            },
            review_sha256s=("e" * 64, "f" * 64),
        )
        self.spec = ControlSpec(
            densities_bps=(500, 1_000, 2_000),
            main_density_bps=1_000,
            wrong_key_count=3,
            wrong_key_seed=b"tiny-test-wrong-key-seed",
        )


class RunCorpusControlsTests(unittest.TestCase):
    def setUp(self) -> None:
        fixture = CorpusControlsFixture()
        self.config = fixture.config
        self.corpus = fixture.corpus
        self.spec = fixture.spec
        self.key = fixture.key

    def test_controls_are_deterministic_self_detecting_and_key_safe(self) -> None:
        first = build_corpus_controls(self.config, self.corpus, spec=self.spec)
        second = build_corpus_controls(self.config, self.corpus, spec=self.spec)

        self.assertEqual(first.files, second.files)
        self.assertEqual(first.artifact, second.artifact)
        self.assertEqual(
            first.artifact["keySha256"], hashlib.sha256(self.key).hexdigest()
        )
        all_bytes = b"".join(first.files[path] for path in sorted(first.files))
        self.assertNotIn(self.key.hex().encode("ascii"), all_bytes)
        self.assertNotIn(b'"keyHex"', first.files[ARTIFACT_PATH])
        self.assertEqual(first.files[ARTIFACT_PATH].count(b'"keySha256"'), 1)

        results = first.artifact["results"]
        self.assertEqual([result["densityBps"] for result in results], [500, 1_000, 2_000])
        for result in results:
            marked = result["trueKey"]["marked"]
            unmarked = result["trueKey"]["unmarked"]
            marking = result["marking"]
            self.assertEqual(marked["status"], "detected")
            self.assertEqual(unmarked["status"], "not_detected")
            self.assertEqual(marked["hits"], marked["activePositions"])
            self.assertEqual(marked["activePositions"], marking["activePositions"])
            self.assertGreater(marking["changedPositions"], 0)
            self.assertLessEqual(
                marking["changedPositions"], marking["activePositions"]
            )
            self.assertEqual(
                marking["coverage"]["activePerEligible"]["numerator"],
                marking["activePositions"],
            )
            wrong_keys = result["wrongKeysOnMarked"]
            self.assertEqual(wrong_keys["count"], 3)
            self.assertEqual(len(wrong_keys["scores"]), 3)
            self.assertTrue(all("status" in score for score in wrong_keys["scores"]))
        self.assertTrue(first.artifact["acceptance"]["passed"])

        manifest = json.loads(first.files[MARKED_MANIFEST_PATH])
        self.assertEqual(manifest["densityBps"], 1_000)
        self.assertEqual(manifest["documentCount"], 2)
        self.assertEqual(len(manifest["documents"]), 2)
        self.assertIn("markedSha256", manifest["documents"][0])
        self.assertIn("sourceSha256", manifest["documents"][0])

    def test_complete_review_approval_is_required(self) -> None:
        state = vars(self.corpus).copy()
        state["review_sha256s"] = ()
        without_reviews = SimpleNamespace(**state)
        with self.assertRaisesRegex(ValueError, "review"):
            build_corpus_controls(self.config, without_reviews, spec=self.spec)

        state = vars(self.corpus).copy()
        state["review_approval"] = {
            **self.corpus.review_approval,
            "approvedDocumentCount": 1,
        }
        incomplete = SimpleNamespace(**state)
        with self.assertRaisesRegex(ValueError, "approvedDocumentCount"):
            build_corpus_controls(self.config, incomplete, spec=self.spec)

    def test_loader_revalidates_both_exact_review_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inventory = {
                "corpusVersion": "test-corpus-v1",
                "documents": [
                    {
                        "documentId": document.document_id,
                        "eligibleOccurrences": document.eligible_positions,
                        "occurrences": [],
                        "reviewStatus": "pending_manual_context_review",
                        "sha256": document.sha256,
                    }
                    for document in self.corpus.documents
                ],
                "lexiconSha256": self.corpus.lexicon.sha256,
                "methodology": "Tiny review-validation fixture.",
                "schemaVersion": 1,
                "sources": [
                    {"title": "Test source", "url": "https://example.com/source"}
                ],
                "verifiedAt": "2026-08-16",
            }
            inventory_bytes = canonical_json_bytes(inventory)
            inventory_path = root / "context-inventory.json"
            inventory_path.write_bytes(inventory_bytes)
            inventory_sha256 = hashlib.sha256(inventory_bytes).hexdigest()

            review_bindings = []
            review_sha256s = []
            for index, document in enumerate(self.corpus.documents):
                review = {
                    "corpusVersion": "test-corpus-v1",
                    "documents": [
                        {
                            "decision": "approved",
                            "documentId": document.document_id,
                            "documentSha256": document.sha256,
                            "eligibleOccurrences": document.eligible_positions,
                            "findings": [],
                            "reviewedOccurrences": document.eligible_positions,
                        }
                    ],
                    "inventorySha256": inventory_sha256,
                    "lexiconSha256": self.corpus.lexicon.sha256,
                    "methodology": "Reviewed every supplied occurrence.",
                    "reviewVersion": f"test-review-{index + 1}",
                    "reviewer": f"reviewer-{index + 1}",
                    "schemaVersion": 1,
                    "sources": [
                        {
                            "title": "Test source",
                            "url": "https://example.com/source",
                        }
                    ],
                    "verifiedAt": "2026-08-16",
                }
                path = root / f"review-{index + 1}.json"
                raw = canonical_json_bytes(review)
                path.write_bytes(raw)
                digest = hashlib.sha256(raw).hexdigest()
                review_bindings.append((path, digest))
                review_sha256s.append(digest)

            config_state = vars(self.config).copy()
            config_state.update(
                inventory_expected_sha256=inventory_sha256,
                inventory_path=inventory_path,
                review_bindings=tuple(review_bindings),
            )
            config = SimpleNamespace(**config_state)
            corpus_state = vars(self.corpus).copy()
            corpus_state.update(
                inventory_sha256=inventory_sha256,
                review_approval={
                    "approvedDocumentCount": 2,
                    "corpusVersion": "test-corpus-v1",
                    "inventorySha256": inventory_sha256,
                    "lexiconSha256": self.corpus.lexicon.sha256,
                    "reviewVersions": ["test-review-1", "test-review-2"],
                    "schemaVersion": 1,
                },
                review_sha256s=tuple(review_sha256s),
            )
            corpus = SimpleNamespace(**corpus_state)

            with (
                patch("run_experiment.load_experiment_config", return_value=config),
                patch("run_experiment.load_reviewed_corpus", return_value=corpus),
            ):
                loaded_config, loaded_corpus = load_control_inputs(
                    root / "config.json", root=root
                )
            self.assertIs(loaded_config, config)
            self.assertIs(loaded_corpus, corpus)

            review_bindings[1][0].write_bytes(
                review_bindings[1][0].read_bytes() + b" "
            )
            with (
                patch("run_experiment.load_experiment_config", return_value=config),
                patch("run_experiment.load_reviewed_corpus", return_value=corpus),
                self.assertRaisesRegex(ValueError, "hash differs"),
            ):
                load_control_inputs(root / "config.json", root=root)

            config_state["review_bindings"] = tuple(review_bindings[:1])
            one_review_config = SimpleNamespace(**config_state)
            with (
                patch(
                    "run_experiment.load_experiment_config",
                    return_value=one_review_config,
                ),
                patch("run_experiment.load_reviewed_corpus", return_value=corpus),
                self.assertRaisesRegex(ValueError, "missing document"),
            ):
                load_control_inputs(root / "config.json", root=root)

    def test_output_freshness_is_byte_exact(self) -> None:
        outputs = build_corpus_controls(self.config, self.corpus, spec=self.spec)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            missing = check_control_outputs(root, outputs)
            self.assertFalse(missing["passed"])

            write_control_outputs(root, outputs)
            fresh = check_control_outputs(root, outputs)
            self.assertTrue(fresh["passed"])
            self.assertTrue(all(fresh["files"].values()))

            artifact_path = root / ARTIFACT_PATH
            artifact_path.write_bytes(artifact_path.read_bytes() + b" ")
            stale = check_control_outputs(root, outputs)
            self.assertFalse(stale["passed"])
            self.assertFalse(stale["files"][ARTIFACT_PATH])

            write_control_outputs(root, outputs)
            marked_path = root / "corpus/marked-1000/doc-01.md"
            marked_path.write_bytes(marked_path.read_bytes() + b"drift")
            stale = check_control_outputs(root, outputs)
            self.assertFalse(stale["passed"])
            self.assertFalse(stale["files"]["corpus/marked-1000/doc-01.md"])

            write_control_outputs(root, outputs)
            unexpected_path = root / "corpus/marked-1000/doc-old.md"
            unexpected_path.write_text("stale\n", encoding="utf-8")
            stale = check_control_outputs(root, outputs)
            self.assertFalse(stale["passed"])
            self.assertEqual(
                stale["unexpectedFiles"], ["corpus/marked-1000/doc-old.md"]
            )

    def test_production_spec_requires_configured_one_thousand_wrong_keys(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly 1000"):
            control_spec_from_config(self.config)
        parsed = control_spec_from_config(
            self.config,
            require_production_wrong_key_count=False,
        )
        self.assertEqual(parsed, self.spec)

        mismatched = ControlSpec(
            densities_bps=self.spec.densities_bps,
            main_density_bps=self.spec.main_density_bps,
            wrong_key_count=2,
            wrong_key_seed=b"different-wrong-key-seed",
        )
        with self.assertRaisesRegex(ValueError, "frozen experiment config"):
            build_corpus_controls(self.config, self.corpus, spec=mismatched)

    def test_cli_check_never_writes_and_uses_byte_freshness(self) -> None:
        outputs = build_corpus_controls(self.config, self.corpus, spec=self.spec)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            arguments = ["--root", str(root), "--check"]
            with (
                patch(
                    "run_corpus_controls.load_control_inputs",
                    return_value=(self.config, self.corpus),
                ),
                patch(
                    "run_corpus_controls.control_spec_from_config",
                    return_value=self.spec,
                ),
                patch("run_corpus_controls.build_corpus_controls", return_value=outputs),
                patch("run_corpus_controls.write_control_outputs") as writer,
                redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(main(arguments), 1)
                writer.assert_not_called()

            write_control_outputs(root, outputs)
            with (
                patch(
                    "run_corpus_controls.load_control_inputs",
                    return_value=(self.config, self.corpus),
                ),
                patch(
                    "run_corpus_controls.control_spec_from_config",
                    return_value=self.spec,
                ),
                patch("run_corpus_controls.build_corpus_controls", return_value=outputs),
                patch("run_corpus_controls.write_control_outputs") as writer,
                redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(main(arguments), 0)
                writer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
