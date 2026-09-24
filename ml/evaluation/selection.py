"""Backtesting and model-selection policy (PRD 6.2, TRD 13.1-13.4).

* Chronological data only; models are fitted on the past and scored on the future.
* Model choice uses the VALIDATION window; the TEST window is scored once at the end.
* A candidate is promoted only if it beats the naive-family baseline by `min_gain`
  (EC-28: never promote a worse/no-better complex model).
* Failures of one candidate never fail the SKU (EC-23) - it is recorded and skipped.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ml.evaluation.metrics import all_metrics, wape
from ml.models import baselines as bl

MIN_HISTORY_DAYS = 28          # below this -> cold start (EC-19)
ZERO_SHARE_INTERMITTENT = 0.5  # >50% zero days -> intermittent (EC-52)


@dataclass
class Selection:
    model: str
    status: str                       # approved | baseline | cold_start | intermittent
    metrics: dict = field(default_factory=dict)
    candidates: dict = field(default_factory=dict)
    failures: dict = field(default_factory=dict)
    quality_flag: str = "ok"          # ok | low_confidence | fallback


def clamp_non_negative(x) -> np.ndarray:
    """EC-21: forecasts feeding the reorder engine can never be negative."""
    return np.clip(np.asarray(x, dtype=float), 0.0, None)


def reindex_daily(y: pd.Series, treat_missing_as_zero: bool = True) -> pd.Series:
    """EC-20: make missing calendar days explicit before training."""
    full = pd.date_range(y.index.min(), y.index.max(), freq="D")
    y = y.reindex(full)
    return y.fillna(0.0) if treat_missing_as_zero else y.interpolate()


def rolling_origin_wape(model_name: str, y: pd.Series, horizon: int, n_folds: int = 3) -> float:
    """Average WAPE over `n_folds` chronological folds ending at the end of `y`."""
    scores = []
    for k in range(n_folds, 0, -1):
        cut = len(y) - k * horizon
        if cut < MIN_HISTORY_DAYS:
            continue
        m = bl.build(model_name).fit(y.iloc[:cut])
        pred = clamp_non_negative(m.predict(horizon))
        scores.append(wape(y.iloc[cut:cut + horizon], pred))
    if not scores:
        raise ValueError("not enough history for backtest")
    return float(np.mean(scores))


def select_model(y_train_val: pd.Series, horizon: int = 14, candidates=("dow_mean", "seasonal_naive", "moving_average",
                 "arima", "prophet"), min_gain: float = 0.02) -> Selection:
    y = reindex_daily(y_train_val)
    if len(y) < MIN_HISTORY_DAYS:
        return Selection("moving_average", "cold_start", quality_flag="low_confidence")
    if (y == 0).mean() > ZERO_SHARE_INTERMITTENT:
        return Selection("moving_average", "intermittent", quality_flag="low_confidence")
    scores, failures = {}, {}
    for name in ("naive", *candidates):
        try:
            scores[name] = rolling_origin_wape(name, y, horizon)
        except Exception as exc:  # EC-22/23: isolate model failures
            failures[name] = f"{type(exc).__name__}: {exc}"
    if "naive" not in scores:
        return Selection("moving_average", "baseline", failures=failures, quality_flag="fallback")
    base = scores["naive"]
    best = min((n for n in scores if n != "naive"), key=lambda n: scores[n], default=None)
    if best is not None and scores[best] <= base * (1 - min_gain):
        return Selection(best, "approved", {"wape": scores[best], "baseline_wape": base}, scores, failures)
    fallback = best if best is not None and scores[best] < base else "naive"
    return Selection("naive" if fallback == "naive" else fallback, "baseline",
                     {"wape": scores.get(fallback, base), "baseline_wape": base}, scores, failures)


def evaluate_on_test(selection: Selection, y_trainval: pd.Series, y_test: pd.Series) -> dict:
    """One-shot final evaluation on the untouched TEST window vs the naive baseline."""
    y_trainval = reindex_daily(y_trainval)
    h = len(y_test)
    out = {}
    for name in {selection.model, "naive"}:
        pred = clamp_non_negative(bl.build(name).fit(y_trainval).predict(h))
        out[name] = all_metrics(y_test.reindex(pd.date_range(y_test.index.min(), periods=h)).fillna(0), pred)
    return out


def select_on_validation(y_train: pd.Series, y_val: pd.Series, candidates=("dow_mean", "seasonal_naive",
                         "moving_average", "arima", "prophet"), min_gain: float = 0.02) -> Selection:
    """Fit on TRAIN, score each candidate on the VALIDATION window, apply the baseline gate."""
    ytr = reindex_daily(y_train)
    h = len(pd.date_range(y_val.index.min(), y_val.index.max(), freq="D"))
    yv = reindex_daily(y_val).to_numpy()
    if len(ytr) < MIN_HISTORY_DAYS:
        return Selection("moving_average", "cold_start", quality_flag="low_confidence")
    if (ytr == 0).mean() > ZERO_SHARE_INTERMITTENT:
        return Selection("moving_average", "intermittent", quality_flag="low_confidence")
    scores, failures = {}, {}
    for name in ("naive", *candidates):
        try:
            scores[name] = wape(yv, clamp_non_negative(bl.build(name).fit(ytr).predict(h)))
        except Exception as exc:  # EC-22/23
            failures[name] = f"{type(exc).__name__}: {exc}"
    base = scores.get("naive")
    others = {k: v for k, v in scores.items() if k != "naive"}
    if base is None or not others:
        return Selection("moving_average", "baseline", failures=failures, quality_flag="fallback")
    best = min(others, key=others.get)
    if others[best] <= base * (1 - min_gain):
        return Selection(best, "approved", {"val_wape": others[best], "baseline_val_wape": base}, scores, failures)
    return Selection("naive", "baseline", {"val_wape": base, "baseline_val_wape": base}, scores, failures)
