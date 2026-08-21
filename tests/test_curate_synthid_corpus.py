import unittest

import curate_synthid_corpus as curator


class CurateSynthIDCorpusTests(unittest.TestCase):
    def test_exact_edits_apply_once_and_record_ids(self):
        text, edit_ids = curator.apply_exact_edits(
            "The total was seventeen plots.",
            [
                {
                    "editId": "count",
                    "old": "seventeen plots",
                    "new": "seven plots",
                }
            ],
        )

        self.assertEqual(text, "The total was seven plots.")
        self.assertEqual(edit_ids, ["count"])

    def test_missing_or_repeated_old_text_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "exactly once"):
            curator.apply_exact_edits(
                "No matching phrase.",
                [{"editId": "missing", "old": "old", "new": "new"}],
            )
        with self.assertRaisesRegex(ValueError, "exactly once"):
            curator.apply_exact_edits(
                "old and old",
                [{"editId": "duplicate", "old": "old", "new": "new"}],
            )


if __name__ == "__main__":
    unittest.main()
