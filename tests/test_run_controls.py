from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

from run_controls import canonical_json_bytes, run_preflight
from watermark_toy import load_lexicon


ROOT = Path(__file__).resolve().parents[1]
FIXED_KEY = bytes.fromhex(
    "00112233445566778899aabbccddeeff"
    "102132435465768798a9babbdcddedef"
)


class RunControlsTests(unittest.TestCase):
    def test_preflight_is_deterministic_and_contains_no_plaintext_key(self) -> None:
        fixture_path = ROOT / "fixtures" / "synthetic-preflight-v1.json"
        fixture_bytes = fixture_path.read_bytes()
        fixture = json.loads(fixture_bytes)
        fixture["generator"]["documentCount"] = 4
        fixture["generator"]["repetitionsPerDocument"] = 8
        lexicon = load_lexicon(ROOT / "fixtures" / "synonym_pairs-v1.json")
        arguments = {
            "fixture": fixture,
            "fixture_sha256": hashlib.sha256(canonical_json_bytes(fixture)).hexdigest(),
            "lexicon": lexicon,
            "key": FIXED_KEY,
            "density_bps": 10_000,
            "wrong_key_count": 20,
            "wrong_key_seed": b"unit-test-preflight-seed",
        }
        first = run_preflight(**arguments)
        second = run_preflight(**arguments)

        self.assertEqual(canonical_json_bytes(first), canonical_json_bytes(second))
        self.assertNotIn(FIXED_KEY.hex().encode("ascii"), canonical_json_bytes(first))
        self.assertTrue(first["acceptance"]["passed"])
        self.assertEqual(first["results"]["wrongKeysOnMarked"]["count"], 20)

    def test_preflight_rejects_generator_words_missing_from_lexicon(self) -> None:
        fixture_path = ROOT / "fixtures" / "synthetic-preflight-v1.json"
        fixture_bytes = fixture_path.read_bytes()
        fixture = json.loads(fixture_bytes)
        fixture["generator"]["eligibleWords"] = ["missingword"]
        lexicon = load_lexicon(ROOT / "fixtures" / "synonym_pairs-v1.json")

        with self.assertRaisesRegex(ValueError, "missingword"):
            run_preflight(
                fixture=fixture,
                fixture_sha256=hashlib.sha256(canonical_json_bytes(fixture)).hexdigest(),
                lexicon=lexicon,
                key=FIXED_KEY,
                density_bps=1_000,
                wrong_key_count=1,
            )


if __name__ == "__main__":
    unittest.main()
