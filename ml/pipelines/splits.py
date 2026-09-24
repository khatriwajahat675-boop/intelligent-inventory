"""Train / validation / test splitting (70 / 15 / 15).

PRD 6.2 + TRD 13.4: time-series data must be split chronologically and never
shuffled.  Cross-sectional data (a single inventory snapshot) has no time axis,
so it is split with a seeded, stratified shuffle instead.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRAIN, VAL, TEST = 0.70, 0.15, 0.15


@dataclass(frozen=True)
class SplitResult:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    method: str

    def sizes(self) -> dict[str, int]:
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}


def _check_fractions(train: float, val: float, test: float) -> None:
    if min(train, val, test) <= 0 or abs(train + val + test - 1.0) > 1e-9:
        raise ValueError("fractions must be positive and sum to 1.0")


def chronological_split(
    df: pd.DataFrame,
    date_col: str,
    train: float = TRAIN,
    val: float = VAL,
    test: float = TEST,
) -> SplitResult:
    """Oldest 70% -> train, next 15% -> validation, newest 15% -> test.

    Cut points are snapped to calendar-day boundaries so that a single day is
    never shared between two sets (avoids same-day leakage).
    """
    _check_fractions(train, val, test)
    if df.empty:
        raise ValueError("cannot split an empty frame")
    d = df.sort_values(date_col, kind="mergesort").reset_index(drop=True)
    day = d[date_col].dt.normalize()
    per_day = day.value_counts().sort_index()
    cum_share = per_day.cumsum() / len(d)
    train_end = cum_share.index[np.searchsorted(cum_share.values, train - 1e-12)]
    val_end = cum_share.index[np.searchsorted(cum_share.values, train + val - 1e-12)]
    tr = d[day <= train_end]
    va = d[(day > train_end) & (day <= val_end)]
    te = d[day > val_end]
    if min(len(tr), len(va), len(te)) == 0:
        raise ValueError("data spans too few days for a 3-way chronological split")
    return SplitResult(tr, va, te, "chronological (day-aligned)")


def stratified_split(
    df: pd.DataFrame,
    strata_cols: list[str],
    train: float = TRAIN,
    val: float = VAL,
    test: float = TEST,
    seed: int = 42,
) -> SplitResult:
    """Seeded stratified 70/15/15 split for cross-sectional data.

    Every stratum contributes to each set in (approximately) the requested
    proportion; strata with fewer than 3 rows go entirely to train.
    """
    _check_fractions(train, val, test)
    rng = np.random.default_rng(seed)
    label = df[strata_cols].astype(str).agg("|".join, axis=1)
    part = pd.Series("train", index=df.index)
    for _, idx in df.groupby(label).groups.items():
        idx = np.array(idx)
        n = len(idx)
        if n < 3:
            continue
        idx = rng.permutation(idx)
        n_val = max(1, int(round(n * val)))
        n_test = max(1, int(round(n * test)))
        part.loc[idx[:n_val]] = "val"
        part.loc[idx[n_val : n_val + n_test]] = "test"
    return SplitResult(
        df[part == "train"], df[part == "val"], df[part == "test"],
        f"stratified by {strata_cols} (seed={seed})",
    )


def assert_no_overlap(split: SplitResult, key: str) -> None:
    """Row identity must be disjoint across sets."""
    a, b, c = (set(x[key]) for x in (split.train, split.val, split.test))
    if a & b or a & c or b & c:
        raise AssertionError(f"key '{key}' appears in more than one split")


def assert_chronological(split: SplitResult, date_col: str) -> None:
    """max(train) < min(val) and max(val) < min(test)."""
    if not (split.train[date_col].max() < split.val[date_col].min()
            and split.val[date_col].max() < split.test[date_col].min()):
        raise AssertionError("splits overlap in time (leakage)")


def population_stability_index(a: pd.Series, b: pd.Series, bins: int = 10) -> float:
    """PSI of `b` against reference `a` (<0.1 stable, 0.1-0.25 moderate, >0.25 shift)."""
    edges = np.unique(np.quantile(a.dropna(), np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    pa = np.histogram(a.dropna(), edges)[0] / max(a.notna().sum(), 1)
    pb = np.histogram(b.dropna(), edges)[0] / max(b.notna().sum(), 1)
    pa, pb = np.clip(pa, 1e-6, None), np.clip(pb, 1e-6, None)
    return float(np.sum((pb - pa) * np.log(pb / pa)))
