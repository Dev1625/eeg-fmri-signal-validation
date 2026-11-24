import unittest

import numpy as np

from signal_validation.benchmark import blocked_folds, ridge_predict, run
from signal_validation.evaluation import compute_cv_r2, generate_time_shifted_features


class BenchmarkTests(unittest.TestCase):
    def test_folds_cover_all_volumes_once_and_leave_a_gap(self):
        folds = list(blocked_folds(240))
        np.testing.assert_array_equal(np.concatenate([test for _, test in folds]), np.arange(240))
        for train, test in folds:
            self.assertGreaterEqual(np.abs(train[:, None] - test[None, :]).min(), 4)

    def test_train_mean_is_the_zero_score_baseline(self):
        y = np.array([2.0, 3.0, 4.0])
        self.assertEqual(compute_cv_r2(y, np.full(3, 1.0), 1.0), 0.0)

    def test_ridge_handles_constant_feature_and_recovers_slope(self):
        x = np.column_stack([np.arange(8), np.ones(8)])
        y = 3 * x[:, 0] + 5
        pred = ridge_predict(x[:6], y[:6], x[6:], penalty=0.001)
        np.testing.assert_allclose(pred, y[6:], atol=0.003)

    def test_shift_preserves_order_under_rotation(self):
        np.testing.assert_array_equal(generate_time_shifted_features(np.arange(5), 2), [3, 4, 0, 1, 2])

    def test_synthetic_benchmark_exposes_overcleaning(self):
        report = run(n_shifts=16)
        raw, template, notch = report['methods']
        self.assertEqual(raw['artifact_removed_pct'], 0.0)
        self.assertGreater(notch['artifact_removed_pct'], template['artifact_removed_pct'])
        self.assertGreater(template['alpha_vs_clean_pct'], notch['alpha_vs_clean_pct'])
        self.assertGreater(template['cv_r2'], notch['cv_r2'])
        self.assertGreater(template['cv_r2'], template['shift_null_95th_r2'])


if __name__ == '__main__':
    unittest.main()
