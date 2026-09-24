"""Baseline and optional advanced forecasters behind one interface.

fit(y: pd.Series with DatetimeIndex, daily freq) -> self
predict(horizon: int) -> np.ndarray   (raw, may be negative for ARIMA/Prophet)

ARIMA (statsmodels), Prophet and LSTM (torch) are imported lazily so the
baseline path works in a minimal environment and a missing library becomes a
handled failure (EC-22 / EC-23) instead of an import-time crash.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class Forecaster:
    name = "base"

    def fit(self, y: pd.Series) -> "Forecaster":
        raise NotImplementedError

    def predict(self, horizon: int) -> np.ndarray:
        raise NotImplementedError


class Naive(Forecaster):
    name = "naive"

    def fit(self, y):
        self.last = float(y.iloc[-1])
        return self

    def predict(self, horizon):
        return np.full(horizon, self.last)


class MovingAverage(Forecaster):
    name = "moving_average"

    def __init__(self, window: int = 14):
        self.window = window

    def fit(self, y):
        self.mean = float(y.iloc[-self.window:].mean())
        return self

    def predict(self, horizon):
        return np.full(horizon, self.mean)


class SeasonalNaive(Forecaster):
    """Repeat the last full season (default weekly) - captures weekday uplift (EC-25)."""
    name = "seasonal_naive"

    def __init__(self, season: int = 7):
        self.season = season

    def fit(self, y):
        if len(y) < self.season:
            raise ValueError("history shorter than one season")
        self.tail = y.iloc[-self.season:].to_numpy(dtype=float)
        return self

    def predict(self, horizon):
        reps = int(np.ceil(horizon / self.season))
        return np.tile(self.tail, reps)[:horizon]


class DayOfWeekMean(Forecaster):
    """Mean by weekday over the last `weeks` weeks (smoother than SeasonalNaive)."""
    name = "dow_mean"

    def __init__(self, weeks: int = 8):
        self.weeks = weeks

    def fit(self, y):
        y = y.iloc[-7 * self.weeks:]
        self.by_dow = y.groupby(y.index.dayofweek).mean().reindex(range(7)).fillna(y.mean())
        self.next_dow = (y.index[-1] + pd.Timedelta(days=1)).dayofweek
        return self

    def predict(self, horizon):
        return np.array([self.by_dow[(self.next_dow + i) % 7] for i in range(horizon)])


class Arima(Forecaster):
    name = "arima"

    def __init__(self, order=(1, 0, 1)):
        self.order = order

    def fit(self, y):
        from statsmodels.tsa.arima.model import ARIMA  # lazy
        self.res = ARIMA(y.asfreq("D").astype(float), order=self.order).fit()
        return self

    def predict(self, horizon):
        return np.asarray(self.res.forecast(horizon), dtype=float)


class ProphetModel(Forecaster):
    name = "prophet"

    def fit(self, y):
        from prophet import Prophet  # lazy
        m = Prophet(weekly_seasonality=True, yearly_seasonality=len(y) > 365, daily_seasonality=False)
        m.fit(pd.DataFrame({"ds": y.index, "y": y.to_numpy(dtype=float)}))
        self.m, self.last = m, y.index[-1]
        return self

    def predict(self, horizon):
        fut = pd.DataFrame({"ds": pd.date_range(self.last + pd.Timedelta(days=1), periods=horizon)})
        return self.m.predict(fut)["yhat"].to_numpy()


BASELINES = {"naive": Naive, "moving_average": MovingAverage,
             "seasonal_naive": SeasonalNaive, "dow_mean": DayOfWeekMean}
ADVANCED = {"arima": Arima, "prophet": ProphetModel}


def build(name: str) -> Forecaster:
    reg = {**BASELINES, **ADVANCED}
    if name not in reg:
        raise KeyError(f"unknown model '{name}'")
    return reg[name]()
