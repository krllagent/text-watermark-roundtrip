import unittest

import compute_curated_psp as psp


class ComputeCuratedPspTests(unittest.TestCase):
    def test_empty_pipeline_failure_has_zero_similarity(self):
        self.assertEqual(psp.score_or_failure(None, "source", ""), 0.0)

    def test_summary_reports_percentage_mean_by_method(self):
        rows = [
            {"method": "a", "psp": 0.8},
            {"method": "a", "psp": 1.0},
            {"method": "b", "psp": 0.5},
        ]

        result = psp.summarize(rows)

        self.assertEqual(result["a"]["pairCount"], 2)
        self.assertAlmostEqual(result["a"]["meanPspPercent"], 90.0)
        self.assertAlmostEqual(result["b"]["meanPspPercent"], 50.0)


if __name__ == "__main__":
    unittest.main()
