import unittest

import numpy as np

import calibrate_synthid_threshold as calibration


class SynthIDCalibrationTests(unittest.TestCase):
    def test_nearest_rank_quantile_uses_conservative_upper_order_statistic(self):
        values = np.asarray([0.1, 0.2, 0.3, 0.4])

        self.assertEqual(calibration.nearest_rank(values, 0.75), 0.3)
        self.assertEqual(calibration.nearest_rank(values, 0.99), 0.4)

    def test_weighted_bernoulli_null_is_deterministic_and_centered_at_half(self):
        frequencies = {1: 2_000, 2: 100, 3: 20}
        first = calibration.simulate_weighted_bernoulli_null(
            frequencies,
            total_weight=2_260,
            replicates=20_000,
            seed=7,
        )
        second = calibration.simulate_weighted_bernoulli_null(
            frequencies,
            total_weight=2_260,
            replicates=20_000,
            seed=7,
        )

        np.testing.assert_array_equal(first, second)
        self.assertAlmostEqual(float(first.mean()), 0.5, places=3)
        self.assertGreater(calibration.nearest_rank(first, 0.99), 0.5)


if __name__ == "__main__":
    unittest.main()
