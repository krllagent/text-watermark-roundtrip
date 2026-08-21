import unittest

import derive_curated_eval_v4 as derivation


class DeriveCuratedEvalV4Tests(unittest.TestCase):
    def test_only_xai_judge_and_provenance_change(self):
        source = {
            "judges": [
                {"vendor": "OpenAI", "model": "openai/a"},
                {
                    "vendor": "xAI",
                    "model": "x-ai/grok-4.20",
                    "promptUsdPerToken": "1",
                    "completionUsdPerToken": "2",
                },
            ]
        }

        result = derivation.derive(source, "b" * 64)

        self.assertEqual(result["judges"][0], source["judges"][0])
        self.assertEqual(result["judges"][1]["model"], "x-ai/grok-4.6")
        self.assertEqual(result["derivedFrom"]["sha256"], "b" * 64)


if __name__ == "__main__":
    unittest.main()
