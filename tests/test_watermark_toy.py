from __future__ import annotations

import json
import math
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

from watermark_toy import (
    Document,
    SynonymLexicon,
    binomial_tail_probability,
    compare_active_fingerprints,
    encode_text,
    inspect_positions,
    load_lexicon,
    run_wrong_key_controls,
    score_corpus,
    score_text,
)


ROOT = Path(__file__).resolve().parents[1]
FIXED_KEY = bytes.fromhex(
    "00112233445566778899aabbccddeeff"
    "102132435465768798a9babbdcddedef"
)


def alpha_label(number: int) -> str:
    """Return a lowercase alphabetic label that the tokenizer sees as one word."""
    chars: list[str] = []
    value = number
    while True:
        value, remainder = divmod(value, 26)
        chars.append(chr(ord("a") + remainder))
        if value == 0:
            return "".join(reversed(chars))
        value -= 1


def synthetic_document(document_id: str, repetitions: int = 12) -> Document:
    source_words = ["big", "small", "start", "help", "method", "result"]
    fragments: list[str] = []
    for index in range(repetitions):
        for offset, word in enumerate(source_words):
            label = alpha_label(index * len(source_words) + offset)
            fragments.append(f"context{label} {word}.")
    return Document(document_id=document_id, text=" ".join(fragments))


class BinomialTests(unittest.TestCase):
    def test_exact_tail_probability_and_one_percent_boundary(self) -> None:
        self.assertEqual(binomial_tail_probability(16, 20), Fraction(1549, 262144))
        self.assertEqual(binomial_tail_probability(15, 20), Fraction(5425, 262144))
        self.assertLessEqual(binomial_tail_probability(16, 20), Fraction(1, 100))
        self.assertGreater(binomial_tail_probability(15, 20), Fraction(1, 100))

    def test_invalid_binomial_counts_are_rejected(self) -> None:
        for hits, trials in [(-1, 10), (11, 10), (1, -1)]:
            with self.subTest(hits=hits, trials=trials):
                with self.assertRaises(ValueError):
                    binomial_tail_probability(hits, trials)


class LexiconTests(unittest.TestCase):
    def test_lexicon_rejects_duplicate_and_non_word_members(self) -> None:
        with self.assertRaises(ValueError):
            SynonymLexicon.from_pairs([["big", "large"], ["large", "huge"]])
        with self.assertRaises(ValueError):
            SynonymLexicon.from_pairs([["two words", "phrase"]])
        with self.assertRaises(ValueError):
            SynonymLexicon.from_pairs(["ab"])

    def test_fixture_lexicon_is_canonical_and_non_overlapping(self) -> None:
        raw = json.loads((ROOT / "fixtures" / "synonym_pairs-v1.json").read_text())
        lexicon = SynonymLexicon.from_pairs(raw["pairs"])
        self.assertGreaterEqual(len(lexicon.pairs), 30)
        self.assertEqual(len(lexicon.token_to_pair), len(lexicon.pairs) * 2)

    def test_file_lexicon_requires_evidence_and_manual_review_contract(self) -> None:
        incomplete = {"schemaVersion": 1, "pairs": [["big", "large"]]}
        with tempfile.NamedTemporaryFile("w", encoding="utf-8") as fixture:
            json.dump(incomplete, fixture)
            fixture.flush()
            with self.assertRaisesRegex(ValueError, "lexiconVersion"):
                load_lexicon(fixture.name)


class MarkerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.lexicon = SynonymLexicon.from_pairs(
            [
                ["big", "large"],
                ["small", "little"],
                ["begin", "start"],
                ["assist", "help"],
                ["approach", "method"],
                ["outcome", "result"],
            ]
        )

    def test_encode_is_deterministic_and_self_detects(self) -> None:
        document = synthetic_document("deterministic", repetitions=6)
        first = encode_text(
            document.text,
            key=FIXED_KEY,
            document_id=document.document_id,
            density_bps=10_000,
            lexicon=self.lexicon,
        )
        second = encode_text(
            document.text,
            key=FIXED_KEY,
            document_id=document.document_id,
            density_bps=10_000,
            lexicon=self.lexicon,
        )
        self.assertEqual(first, second)
        self.assertEqual(first.active_positions, first.eligible_positions)
        self.assertGreater(first.changed_positions, 0)

        score = score_text(
            first.text,
            key=FIXED_KEY,
            document_id=document.document_id,
            density_bps=10_000,
            lexicon=self.lexicon,
        )
        self.assertEqual(score.active_positions, first.active_positions)
        self.assertEqual(score.hits, score.active_positions)
        self.assertEqual(score.status, "detected")
        self.assertAlmostEqual(score.z_score or 0.0, math.sqrt(score.active_positions))

    def test_golden_vector_and_partner_normalization_are_stable(self) -> None:
        lexicon = SynonymLexicon.from_pairs(
            [
                ["big", "large"],
                ["small", "little"],
                ["begin", "start"],
                ["assist", "help"],
            ]
        )
        text = "Big plans start small, and helpers help big ideas."
        before = inspect_positions(
            text,
            key=FIXED_KEY,
            document_id="golden-1",
            density_bps=10_000,
            lexicon=lexicon,
        )
        encoded = encode_text(
            text,
            key=FIXED_KEY,
            document_id="golden-1",
            density_bps=10_000,
            lexicon=lexicon,
        )
        after = inspect_positions(
            encoded.text,
            key=FIXED_KEY,
            document_id="golden-1",
            density_bps=10_000,
            lexicon=lexicon,
        )

        self.assertEqual(
            lexicon.sha256,
            "76dd360d0b6c3c7ec917742cdfbee02e4f77143096bc61dec77b950606adceee",
        )
        self.assertEqual(
            encoded.text,
            "Big plans start little, and helpers assist big ideas.",
        )
        self.assertEqual(
            [position.fingerprint for position in before],
            [position.fingerprint for position in after],
        )
        self.assertEqual(
            before[0].fingerprint,
            "3ced4be2547c597ca48d0546d2f5650792f2083fead7aa444bcc80c70e3045d1",
        )
        self.assertEqual(
            before[-1].fingerprint,
            "4a0fc3c9bcdef001a74dc7e605497e5daa23c06d8d584d13fd27882bb0a0492f",
        )
        self.assertEqual(
            encode_text(
                encoded.text,
                key=FIXED_KEY,
                document_id="golden-1",
                density_bps=10_000,
                lexicon=lexicon,
            ).text,
            encoded.text,
        )

    def test_active_set_is_monotonic_across_density(self) -> None:
        document = synthetic_document("density", repetitions=20)
        active_sets: list[set[int]] = []
        for density_bps in (500, 1_000, 2_000):
            positions = inspect_positions(
                document.text,
                key=FIXED_KEY,
                document_id=document.document_id,
                density_bps=density_bps,
                lexicon=self.lexicon,
            )
            active_sets.append({position.start for position in positions if position.active})

        self.assertLess(active_sets[0], active_sets[1])
        self.assertLess(active_sets[1], active_sets[2])

    def test_main_density_has_a_fixed_activation_vector(self) -> None:
        document = synthetic_document("activation-golden", repetitions=12)
        positions = inspect_positions(
            document.text,
            key=FIXED_KEY,
            document_id=document.document_id,
            density_bps=1_000,
            lexicon=self.lexicon,
        )
        self.assertEqual(
            [index for index, position in enumerate(positions) if position.active],
            [8, 16, 20, 21, 27, 32, 40, 41, 43, 48, 64, 66, 69, 70],
        )

    def test_repeated_fingerprint_gets_stable_occurrence_rank(self) -> None:
        positions = inspect_positions(
            "big big big big big big big big big big",
            key=FIXED_KEY,
            document_id="repeated",
            density_bps=10_000,
            lexicon=self.lexicon,
        )
        repeated = [
            position
            for position in positions
            if position.context == ("<big|large>",) * 4
        ]
        self.assertEqual(
            [position.occurrence_rank for position in repeated],
            list(range(len(repeated))),
        )

    def test_protected_spans_are_not_eligible_or_modified(self) -> None:
        text = (
            'big "big start" https://big.example/help user@example.com '
            "#start @help 50% small"
        )
        encoded = encode_text(
            text,
            key=FIXED_KEY,
            document_id="protected",
            density_bps=10_000,
            lexicon=self.lexicon,
        )
        self.assertEqual(encoded.eligible_positions, 2)
        for protected in (
            '"big start"',
            "https://big.example/help",
            "user@example.com",
            "#start",
            "@help",
            "50%",
        ):
            self.assertIn(protected, encoded.text)

    def test_words_around_a_url_inside_quotes_remain_protected(self) -> None:
        text = 'outside "big https://example.com/help start" outside'
        encoded = encode_text(
            text,
            key=FIXED_KEY,
            document_id="nested-protection",
            density_bps=10_000,
            lexicon=self.lexicon,
        )
        self.assertEqual(encoded.text, text)
        self.assertEqual(encoded.eligible_positions, 0)

    def test_case_shape_is_preserved(self) -> None:
        encoded = encode_text(
            "Big BIG big Small SMALL small",
            key=FIXED_KEY,
            document_id="case",
            density_bps=10_000,
            lexicon=self.lexicon,
        )
        words = encoded.text.split()
        self.assertTrue(words[0].istitle())
        self.assertTrue(words[1].isupper())
        self.assertTrue(words[2].islower())
        self.assertTrue(words[3].istitle())
        self.assertTrue(words[4].isupper())
        self.assertTrue(words[5].islower())

    def test_short_scoring_unit_is_insufficient_not_clean(self) -> None:
        result = score_text(
            "big small start help",
            key=FIXED_KEY,
            document_id="short",
            density_bps=10_000,
            lexicon=self.lexicon,
            min_active_positions=20,
        )
        self.assertEqual(result.status, "insufficient_evidence")
        self.assertIsNone(result.p_value)
        self.assertIsNone(result.z_score)

    def test_document_id_domain_separates_identical_texts(self) -> None:
        text = synthetic_document("ignored", repetitions=20).text
        left = inspect_positions(
            text,
            key=FIXED_KEY,
            document_id="left",
            density_bps=1_000,
            lexicon=self.lexicon,
        )
        right = inspect_positions(
            text,
            key=FIXED_KEY,
            document_id="right",
            density_bps=1_000,
            lexicon=self.lexicon,
        )
        left_active = {position.start for position in left if position.active}
        right_active = {position.start for position in right if position.active}
        self.assertNotEqual(left_active, right_active)

    def test_invalid_key_density_and_document_id_fail_explicitly(self) -> None:
        cases = [
            {"key": b"short", "document_id": "valid", "density_bps": 1_000},
            {"key": FIXED_KEY, "document_id": "", "density_bps": 1_000},
            {"key": FIXED_KEY, "document_id": "valid", "density_bps": 0},
            {"key": FIXED_KEY, "document_id": "valid", "density_bps": 10_001},
        ]
        for case in cases:
            with self.subTest(case=case):
                with self.assertRaises((TypeError, ValueError)):
                    inspect_positions("big", lexicon=self.lexicon, **case)

    def test_corpus_score_pools_positions_but_keeps_document_diagnostics(self) -> None:
        originals = [synthetic_document(f"doc-{index}", repetitions=8) for index in range(4)]
        marked = [
            Document(
                document_id=document.document_id,
                text=encode_text(
                    document.text,
                    key=FIXED_KEY,
                    document_id=document.document_id,
                    density_bps=2_000,
                    lexicon=self.lexicon,
                ).text,
            )
            for document in originals
        ]
        score = score_corpus(
            marked,
            key=FIXED_KEY,
            density_bps=2_000,
            lexicon=self.lexicon,
        )
        self.assertEqual(score.document_count, 4)
        self.assertEqual(len(score.documents), 4)
        self.assertEqual(score.hits, score.active_positions)
        self.assertEqual(score.status, "detected")

    def test_fingerprint_comparison_separates_lost_and_new_positions(self) -> None:
        document = synthetic_document("fingerprints", repetitions=8)
        marked_text = encode_text(
            document.text,
            key=FIXED_KEY,
            document_id=document.document_id,
            density_bps=2_000,
            lexicon=self.lexicon,
        ).text
        baseline = score_text(
            marked_text,
            key=FIXED_KEY,
            document_id=document.document_id,
            density_bps=2_000,
            lexicon=self.lexicon,
        )
        unchanged = compare_active_fingerprints(baseline, baseline)
        self.assertEqual(unchanged.lost_active, 0)
        self.assertEqual(unchanged.new_active, 0)

        rewritten = score_text(
            "This rewritten passage contains no words from the toy lexicon.",
            key=FIXED_KEY,
            document_id=document.document_id,
            density_bps=2_000,
            lexicon=self.lexicon,
        )
        comparison = compare_active_fingerprints(baseline, rewritten)
        self.assertEqual(comparison.lost_active, baseline.active_positions)
        self.assertEqual(comparison.surviving_active, 0)

        different_document = score_text(
            marked_text,
            key=FIXED_KEY,
            document_id="different-document",
            density_bps=2_000,
            lexicon=self.lexicon,
        )
        with self.assertRaisesRegex(ValueError, "same document ID"):
            compare_active_fingerprints(baseline, different_document)

        different_density = score_text(
            marked_text,
            key=FIXED_KEY,
            document_id=document.document_id,
            density_bps=1_000,
            lexicon=self.lexicon,
        )
        with self.assertRaisesRegex(ValueError, "density_bps"):
            compare_active_fingerprints(baseline, different_density)

    def test_wrong_key_controls_do_not_read_true_key_signal(self) -> None:
        originals = [synthetic_document(f"control-{index}", repetitions=10) for index in range(10)]
        marked = [
            Document(
                document_id=document.document_id,
                text=encode_text(
                    document.text,
                    key=FIXED_KEY,
                    document_id=document.document_id,
                    density_bps=1_000,
                    lexicon=self.lexicon,
                ).text,
            )
            for document in originals
        ]
        true_score = score_corpus(
            marked,
            key=FIXED_KEY,
            density_bps=1_000,
            lexicon=self.lexicon,
        )
        controls = run_wrong_key_controls(
            marked,
            density_bps=1_000,
            lexicon=self.lexicon,
            count=200,
            seed=b"unit-test-wrong-keys",
        )
        self.assertEqual(true_score.status, "detected")
        self.assertEqual(controls.count, 200)
        self.assertLessEqual(controls.detected_rate, 0.04)
        self.assertLess(controls.median_z_score, true_score.z_score or 0.0)

    def test_wrong_key_rate_is_undefined_when_all_scores_are_insufficient(self) -> None:
        controls = run_wrong_key_controls(
            [Document(document_id="tiny", text="big")],
            density_bps=1_000,
            lexicon=self.lexicon,
            count=10,
            seed=b"unit-test-insufficient-wrong-keys",
        )
        self.assertEqual(controls.insufficient_count, 10)
        self.assertIsNone(controls.detected_rate)


if __name__ == "__main__":
    unittest.main()
