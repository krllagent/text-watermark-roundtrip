from __future__ import annotations

import unittest

from text_contract import PROTECTED_SENTINEL, analyze_text, find_protected_spans


class TextContractTests(unittest.TestCase):
    def test_quote_and_nested_url_form_one_exclusion_union(self) -> None:
        text = 'Read "big https://example.com/help?email=user@example.com start" now.'
        spans = find_protected_spans(text)
        self.assertEqual(len(spans), 1)
        self.assertEqual(
            text[spans[0].start : spans[0].end],
            '"big https://example.com/help?email=user@example.com start"',
        )

    def test_analysis_keeps_one_context_sentinel_per_protected_span(self) -> None:
        analysis = analyze_text('alpha "hidden words" beta user@example.com gamma')
        self.assertEqual(analysis.all_word_count, 8)
        self.assertEqual(analysis.scorable_word_count, 3)
        self.assertEqual(
            [token.normalized for token in analysis.context_tokens],
            ["alpha", PROTECTED_SENTINEL, "beta", PROTECTED_SENTINEL, "gamma"],
        )

    def test_contract_protects_expected_surface_exactly(self) -> None:
        protected = [
            "https://example.com/a?b=1",
            "person@example.com",
            "@person",
            "#topic",
            "$1,250.50",
            "19%",
            '"quoted words stay"',
            "“curly quoted words stay”",
            "'single quoted words stay'",
            "‘curly single quoted words stay’",
            "`inline code`",
        ]
        text = " outside ".join(protected)
        spans = find_protected_spans(text)
        extracted = [text[span.start : span.end] for span in spans]
        self.assertEqual(extracted, protected)

    def test_apostrophes_inside_words_do_not_start_protected_spans(self) -> None:
        analysis = analyze_text("don't stop because you're ready")
        self.assertEqual(analysis.protected_spans, ())

    def test_url_trailing_prose_punctuation_stays_outside_the_span(self) -> None:
        text = "See https://example.com/path), then www.example.org/demo."
        spans = find_protected_spans(text)
        self.assertEqual(
            [text[span.start : span.end] for span in spans],
            ["https://example.com/path", "www.example.org/demo"],
        )


if __name__ == "__main__":
    unittest.main()
