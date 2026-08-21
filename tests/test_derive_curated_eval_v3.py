import unittest

import derive_curated_eval_v3 as derivation


class DeriveCuratedEvalV3Tests(unittest.TestCase):
    def test_only_panel_token_cap_and_provenance_change(self):
        source = {"panel": {"maxCompletionTokensPerCall": 900}, "verifiedAt": "old"}

        result = derivation.derive(source, "a" * 64)

        self.assertEqual(result["panel"]["maxCompletionTokensPerCall"], 1400)
        self.assertEqual(result["derivedFrom"]["sha256"], "a" * 64)
        self.assertNotEqual(result["verifiedAt"], "old")


if __name__ == "__main__":
    unittest.main()
