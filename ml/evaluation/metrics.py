"""Forecast error metrics (PRD 2.2 / 6.2): MAE, RMSE, WAPE, sMAPE."""
from __future__ import annotations

import numpy as np


def _a(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


def mae(y, yhat) -> float:
    return float(np.mean(np.abs(_a(y) - _a(yhat))))


def rmse(y, yhat) -> float:
    return float(np.sqrt(np.mean((_a(y) - _a(yhat)) ** 2)))


def wape(y, yhat) -> float:
    """Weighted absolute percentage error; safe with zeros (returns inf if sum(y)=0)."""
    y, yhat = _a(y), _a(yhat)
    denom = np.abs(y).sum()
    return float(np.abs(y - yhat).sum() / denom) if denom > 0 else float("inf")


def smape(y, yhat) -> float:
    y, yhat = _a(y), _a(yhat)
    denom = (np.abs(y) + np.abs(yhat)) / 2
    mask = denom > 0
    return float(np.mean(np.abs(y - yhat)[mask] / denom[mask])) if mask.any() else 0.0


def all_metrics(y, yhat) -> dict:
    return {"wape": wape(y, yhat), "smape": smape(y, yhat), "mae": mae(y, yhat), "rmse": rmse(y, yhat)}
