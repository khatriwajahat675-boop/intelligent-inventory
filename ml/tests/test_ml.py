"""Split, SMOTE, metric and model-selection tests (TRD 18.1 'Model tests')."""
import unittest

import numpy as np
import pandas as pd

from ml.evaluation import selection as sel
from ml.evaluation.metrics import mae, rmse, smape, wape
from ml.models import baselines as bl
from ml.pipelines import data_prep as dp
from ml.pipelines import splits as sp
from ml.pipelines.smote import smote


def frame(n=1000, days=100):
    rng = np.random.default_rng(0)
    d = pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, days, n), unit="D") + pd.to_timedelta(rng.integers(0, 86400, n), unit="s")
    return pd.DataFrame({"id": np.arange(n), "date": d, "cat": rng.choice(list("ABC"), n), "y": rng.integers(0, 2, n)})


class Splits(unittest.TestCase):
    def test_proportions_and_disjoint(self):
        s = sp.chronological_split(frame(5000), "date")
        n = sum(s.sizes().values())
        self.assertEqual(n, 5000)
        self.assertAlmostEqual(len(s.train) / n, 0.70, delta=0.02)
        self.assertAlmostEqual(len(s.val) / n, 0.15, delta=0.02)
        sp.assert_no_overlap(s, "id")

    def test_strictly_chronological_no_shared_day(self):
        s = sp.chronological_split(frame(), "date")
        sp.assert_chronological(s, "date")
        self.assertTrue(s.train.date.dt.normalize().max() < s.val.date.dt.normalize().min())

    def test_input_order_irrelevant(self):
        f = frame()
        a = sp.chronological_split(f, "date")
        b = sp.chronological_split(f.sample(frac=1, random_state=1), "date")
        self.assertEqual(set(a.test.id), set(b.test.id))

    def test_bad_fractions_and_empty(self):
        with self.assertRaises(ValueError):
            sp.chronological_split(frame(), "date", 0.5, 0.2, 0.2)
        with self.assertRaises(ValueError):
            sp.chronological_split(frame().iloc[0:0], "date")

    def test_stratified_reproducible_and_balanced(self):
        f = frame(2000)
        a, b = sp.stratified_split(f, ["cat", "y"], seed=7), sp.stratified_split(f, ["cat", "y"], seed=7)
        self.assertEqual(list(a.test.id), list(b.test.id))
        sp.assert_no_overlap(a, "id")
        for part in (a.train, a.val, a.test):
            self.assertAlmostEqual(part.y.mean(), f.y.mean(), delta=0.05)

    def test_leakage_detector_fires(self):
        s = sp.chronological_split(frame(), "date")
        bad = sp.SplitResult(s.train, s.train, s.test, "x")
        with self.assertRaises(AssertionError):
            sp.assert_no_overlap(bad, "id")

    def test_psi_zero_for_identical(self):
        x = pd.Series(np.random.default_rng(1).normal(size=2000))
        self.assertLess(sp.population_stability_index(x, x), 1e-6)
        self.assertGreater(sp.population_stability_index(x, x + 3), 0.25)


class Smote(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.X = np.vstack([rng.normal(0, 1, (200, 3)), rng.normal(5, 1, (20, 3))])
        self.y = np.array([0] * 200 + [1] * 20)

    def test_balances_and_keeps_originals(self):
        X2, y2 = smote(self.X, self.y)
        self.assertEqual((y2 == 0).sum(), (y2 == 1).sum())
        np.testing.assert_array_equal(X2[:220], self.X)

    def test_synthetic_inside_minority_hull_and_deterministic(self):
        X2, y2 = smote(self.X, self.y, seed=3)
        syn = X2[220:]
        lo, hi = self.X[self.y == 1].min(0), self.X[self.y == 1].max(0)
        self.assertTrue((syn >= lo - 1e-9).all() and (syn <= hi + 1e-9).all())
        np.testing.assert_array_equal(X2, smote(self.X, self.y, seed=3)[0])

    def test_tiny_minority_and_strategy(self):
        X = np.vstack([self.X[:50], self.X[-1:]]); y = np.array([0] * 50 + [1])
        self.assertEqual((smote(X, y)[1] == 1).sum(), 50)
        self.assertEqual((smote(self.X, self.y, sampling_strategy=0.5)[1] == 1).sum(), 100)

    def test_validation(self):
        with self.assertRaises(ValueError):
            smote(self.X, self.y[:-1])
        with self.assertRaises(ValueError):
            smote(self.X, self.y, sampling_strategy=0)


class Metrics(unittest.TestCase):
    def test_values(self):
        self.assertEqual(mae([1, 2], [2, 4]), 1.5)
        self.assertAlmostEqual(rmse([0, 0], [3, 4]), (12.5) ** 0.5)
        self.assertAlmostEqual(wape([10, 10], [8, 12]), 0.2)
        self.assertEqual(wape([0, 0], [1, 1]), float("inf"))
        self.assertEqual(smape([0, 0], [0, 0]), 0.0)


def weekly_series(n=140, uplift=3.0, seed=0):
    idx = pd.date_range("2024-01-01", periods=n)
    rng = np.random.default_rng(seed)
    base = 10 + np.where(idx.dayofweek >= 5, uplift * 3, 0) + rng.normal(0, 0.5, n)
    return pd.Series(base, index=idx)


class Forecasting(unittest.TestCase):
    def test_baselines_shapes(self):
        y = weekly_series()
        for name in bl.BASELINES:
            self.assertEqual(len(bl.build(name).fit(y).predict(14)), 14)

    def test_weekend_uplift_learned(self):                         # EC-25
        y = weekly_series()
        s = sel.select_on_validation(y[:100], y[100:])
        self.assertEqual(s.status, "approved")
        self.assertIn(s.model, ("dow_mean", "seasonal_naive"))

    def test_no_seasonality_falls_back_to_baseline_family(self):    # EC-28
        rng = np.random.default_rng(1)
        y = pd.Series(rng.poisson(10, 140).astype(float), index=pd.date_range("2024-01-01", periods=140))
        s = sel.select_on_validation(y[:100], y[100:])
        self.assertIn(s.status, ("approved", "baseline"))

    def test_cold_start_and_intermittent(self):                     # EC-19 / EC-52
        y = weekly_series(10)
        self.assertEqual(sel.select_on_validation(y[:6], y[6:]).status, "cold_start")
        z = pd.Series(([0.0] * 9 + [5.0]) * 12, index=pd.date_range("2024-01-01", periods=120))
        self.assertEqual(sel.select_on_validation(z[:80], z[80:]).status, "intermittent")

    def test_missing_library_is_isolated(self):                     # EC-22/23
        y = weekly_series()
        s = sel.select_on_validation(y[:100], y[100:], candidates=("dow_mean", "not_a_model"))
        self.assertIn("not_a_model", s.failures)
        self.assertIn(s.status, ("approved", "baseline"))

    def test_negative_clamp_and_gaps_reindexed(self):               # EC-20 / EC-21
        np.testing.assert_array_equal(sel.clamp_non_negative([-1, 2]), [0, 2])
        y = weekly_series(20).drop(pd.date_range("2024-01-05", periods=3))
        self.assertEqual(len(sel.reindex_daily(y)), 20)

    def test_no_lookahead_in_test_eval(self):
        y = weekly_series(140)
        s = sel.select_on_validation(y[:100], y[100:120])
        res = sel.evaluate_on_test(s, y[:120], y[120:])
        self.assertIn("naive", res)


class Loaders(unittest.TestCase):
    def test_eu_number_parsing(self):
        out = dp.parse_eu_number(pd.Series(["$2.084,25", "70,68%", "1.377", "5,81", "-7,14%", "x"]))
        self.assertEqual(list(out[:5]), [2084.25, 70.68, 1377.0, 5.81, -7.14])
        self.assertTrue(np.isnan(out.iloc[5]))


if __name__ == "__main__":
    unittest.main()
