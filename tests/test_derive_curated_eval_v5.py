import unittest

import derive_curated_eval_v5 as derivation


class DeriveCuratedEvalV5Tests(unittest.TestCase):
    def _source(self):
        return {
            "budgetsUsd": {"dipperGpu": "0.35", "panel": "0.65", "totalAdditional": "1.25", "transformations": "0.25"},
            "panel": {"canaryCalls": 4, "fullCalls": 200},
            "judges": [
                {"vendor": "OpenAI", "model": "openai/a"},
                {"vendor": "Anthropic", "model": "anthropic/claude-haiku-4.5", "promptUsdPerToken": "1", "completionUsdPerToken": "2"},
                {"vendor": "Google", "model": "google/g"},
                {"vendor": "xAI", "model": "x-ai/grok-4.20", "promptUsdPerToken": "1", "completionUsdPerToken": "2"},
            ],
        }

    def test_replaces_two_judges_and_raises_panel_budget(self):
        result = derivation.derive(self._source(), "c" * 64)

        models = [judge["model"] for judge in result["judges"]]
        self.assertEqual(models, ["openai/a", "anthropic/claude-sonnet-5", "google/g", "x-ai/grok-4.6"])
        self.assertEqual(result["judges"][3]["reasoningEffort"], "low")
        self.assertNotIn("reasoningEffort", result["judges"][1])
        self.assertEqual(result["budgetsUsd"]["panel"], "1.60")
        self.assertEqual(result["budgetsUsd"]["totalAdditional"], "2.20")
        self.assertEqual(result["panel"]["candidatesPerPrompt"], 1)
        self.assertEqual(result["derivedFrom"]["sha256"], "c" * 64)

    def test_rejects_unexpected_source_judges(self):
        source = self._source()
        source["judges"][1]["model"] = "anthropic/other"
        with self.assertRaises(ValueError):
            derivation.derive(source, "c" * 64)


if __name__ == "__main__":
    unittest.main()
