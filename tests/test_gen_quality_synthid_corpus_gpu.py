import unittest

import gen_quality_synthid_corpus_gpu as generator


def _good_report() -> str:
    paragraphs = []
    for index in range(8):
        sentences = []
        for sentence in range(4):
            sentences.append(
                "The library team recorded ordinary visitor feedback and reviewed "
                f"the tablet procedure during phase {index + 1}, observation {sentence + 1}."
            )
        paragraphs.append(" ".join(sentences))
    paragraphs[-1] += (
        " The trial began on 12 March 2026 and cost $4,250. "
        "The report recommends continuing the service after the final review."
    )
    return "\n\n".join(paragraphs)


class QualityGateTests(unittest.TestCase):
    def test_accepts_finished_coherent_english_report(self):
        issues = generator.quality_issues(
            _good_report(),
            token_count=760,
            max_new_tokens=1_100,
            stopped_on_eos=True,
        )

        self.assertEqual(issues, [])

    def test_rejects_foreign_script_drift(self):
        text = _good_report() + "\n\n" + ("随机文本" * 80) + "."

        issues = generator.quality_issues(
            text,
            token_count=850,
            max_new_tokens=1_100,
            stopped_on_eos=True,
        )

        self.assertIn("non_latin_script", issues)

    def test_rejects_output_cut_off_at_token_limit(self):
        issues = generator.quality_issues(
            _good_report(),
            token_count=1_100,
            max_new_tokens=1_100,
            stopped_on_eos=False,
        )

        self.assertIn("no_natural_stop", issues)

    def test_rejects_headings_and_missing_required_report_elements(self):
        text = _good_report().replace("$4,250", "the agreed amount")
        text = "# Summary\n\n" + text.replace("The report recommends", "The team considered")

        issues = generator.quality_issues(
            text,
            token_count=760,
            max_new_tokens=1_100,
            stopped_on_eos=True,
        )

        self.assertIn("heading_or_list", issues)
        self.assertIn("missing_currency", issues)
        self.assertIn("missing_recommendation", issues)

    def test_retry_prompt_targets_only_observed_failures(self):
        prompt = generator.retry_prompt(
            "Write the report.",
            ["missing_currency", "word_count"],
        )

        self.assertIn("$4,250", prompt)
        self.assertIn("between 520 and 620 words", prompt)
        self.assertNotIn("seven or eight prose paragraphs", prompt)


if __name__ == "__main__":
    unittest.main()
