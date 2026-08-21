import unittest

import derive_curated_eval_v2 as derivation


class DeriveCuratedEvalV2Tests(unittest.TestCase):
    def test_only_google_judge_and_provenance_change(self):
        source = {
            "judges": [
                {"vendor": "OpenAI", "model": "openai/a"},
                {
                    "vendor": "Google",
                    "model": "google/gemini-3.7-flash",
                    "promptUsdPerToken": "1",
                    "completionUsdPerToken": "2",
                },
            ],
            "verifiedAt": "old",
        }

        result = derivation.derive(source, source_sha256="a" * 64)

        self.assertEqual(result["judges"][0], source["judges"][0])
        self.assertEqual(result["judges"][1]["model"], "google/gemma-4-31b-it")
        self.assertEqual(result["derivedFrom"]["sha256"], "a" * 64)
        self.assertNotEqual(result["verifiedAt"], "old")


if __name__ == "__main__":
    unittest.main()
