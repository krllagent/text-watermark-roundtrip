import unittest

from analyze_synthid_ngram_retention import aggregate_counts, fit_line


class AnalyzeSynthIDNgramRetentionTests(unittest.TestCase):
    def test_fit_line_reports_exact_linear_relationship(self):
        result = fit_line([(0.0, 0.5), (0.5, 0.6), (1.0, 0.7)])

        self.assertAlmostEqual(result["intercept"], 0.5)
        self.assertAlmostEqual(result["slope"], 0.2)
        self.assertAlmostEqual(result["pearsonR"], 1.0)
        self.assertAlmostEqual(result["rSquared"], 1.0)

    def test_aggregate_counts_keeps_exact_counts_authoritative(self):
        result = aggregate_counts(
            [
                {
                    "validPositions": 3,
                    "reusedPositions": 2,
                    "novelPositions": 1,
                    "gValueCount": 6,
                    "gOneCount": 4,
                    "reusedGValueCount": 4,
                    "reusedGOneCount": 3,
                    "novelGValueCount": 2,
                    "novelGOneCount": 1,
                },
                {
                    "validPositions": 2,
                    "reusedPositions": 1,
                    "novelPositions": 1,
                    "gValueCount": 4,
                    "gOneCount": 2,
                    "reusedGValueCount": 2,
                    "reusedGOneCount": 2,
                    "novelGValueCount": 2,
                    "novelGOneCount": 0,
                },
            ]
        )

        self.assertEqual(result["validPositions"], 5)
        self.assertEqual(result["reusedPositions"], 3)
        self.assertEqual(result["novelPositions"], 2)
        self.assertEqual(result["gValueCount"], 10)
        self.assertEqual(result["gOneCount"], 6)
        self.assertEqual(result["reusedGValueCount"], 6)
        self.assertEqual(result["reusedGOneCount"], 5)
        self.assertEqual(result["novelGValueCount"], 4)
        self.assertEqual(result["novelGOneCount"], 1)
        self.assertAlmostEqual(result["exactNgramReuseFraction"], 0.6)
        self.assertAlmostEqual(result["meanG"], 0.6)
        self.assertAlmostEqual(result["reusedMeanG"], 5 / 6)
        self.assertAlmostEqual(result["novelMeanG"], 0.25)


if __name__ == "__main__":
    unittest.main()
