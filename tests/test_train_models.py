import unittest
import numpy as np
import pandas as pd
from pathlib import Path

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import scripts.train_models as tm


class TestTrainModels(unittest.TestCase):
    def test_make_training_matrix_shape(self):
        idx = pd.date_range('2020-01-01', periods=500, freq='10min')
        series = pd.Series(np.arange(500.0), index=idx)
        X, y = tm.make_training_matrix(series)
        self.assertGreater(X.shape[0], 0)
        self.assertEqual(X.shape[0], y.shape[0])

    def test_standardize_fit_transform(self):
        X = np.random.RandomState(0).randn(100, 10)
        mean, scale = tm.standardize_fit(X)
        Xs = tm.standardize_transform(X, mean, scale)
        # mean about 0, std about 1
        self.assertTrue(np.allclose(Xs.mean(axis=0), 0, atol=1e-7))
        self.assertTrue(np.allclose(Xs.std(axis=0), 1, atol=1e-7))

    def test_fit_ridge_basic(self):
        rng = np.random.RandomState(1)
        X = rng.normal(size=(200, 5))
        true_beta = np.array([0.5, -1.2, 0.0, 2.0, -0.7])
        y = X @ true_beta + rng.normal(scale=0.01, size=200)
        coef = tm.fit_ridge(X, y, alpha=0.1)
        # coef includes intercept first
        est = coef[1:]
        self.assertTrue(np.allclose(est, true_beta, atol=0.2))


if __name__ == '__main__':
    unittest.main()
