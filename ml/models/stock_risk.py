"""Stock-out risk classifier trained with SMOTE (train split only).

Pipeline: fit encoder/scaler on TRAIN -> SMOTE on TRAIN -> fit -> pick decision
threshold on VALIDATION -> report once on TEST.  Target-defining columns
(Stock_On_Hand / Reorder_Level for FMCG, on-hand / days-of-inventory for the
grocery file) are deliberately excluded from features to prevent leakage.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler

from ml.pipelines.smote import smote


@dataclass
class Design:
    num_cols: list[str]
    cat_cols: list[str]
    scaler: StandardScaler | None = None
    cat_levels: dict | None = None

    def fit(self, df: pd.DataFrame) -> "Design":
        self.scaler = StandardScaler().fit(df[self.num_cols].astype(float))
        self.cat_levels = {c: sorted(df[c].astype(str).unique()) for c in self.cat_cols}
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        parts = [self.scaler.transform(df[self.num_cols].astype(float))]
        for c in self.cat_cols:
            v = df[c].astype(str).to_numpy()
            parts.append(np.column_stack([(v == lvl).astype(float) for lvl in self.cat_levels[c]]))
        return np.hstack(parts)

    def feature_names(self) -> list[str]:
        names = list(self.num_cols)
        for c in self.cat_cols:
            names += [f"{c}={lvl}" for lvl in self.cat_levels[c]]
        return names


def _metrics(y, p, thr: float) -> dict:
    pred = (p >= thr).astype(int)
    out = {
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y, p)),
        "threshold": float(thr),
        "positives": int(y.sum()), "n": int(len(y)),
    }
    out["roc_auc"] = float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan")
    return out


def _best_threshold(y, p) -> float:
    grid = np.unique(np.quantile(p, np.linspace(0.05, 0.95, 61)))
    scores = [f1_score(y, (p >= t).astype(int), zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


def run_experiment(train, val, test, num_cols, cat_cols, target="stock_risk",
                   use_smote=True, model="logreg", seed=42) -> dict:
    design = Design(num_cols, cat_cols).fit(train)
    Xtr, Xva, Xte = (design.transform(d) for d in (train, val, test))
    ytr, yva, yte = (d[target].to_numpy() for d in (train, val, test))
    n_before = len(ytr)
    if use_smote:
        Xtr, ytr = smote(Xtr, ytr, k_neighbors=5, seed=seed)
    clf = (LogisticRegression(max_iter=2000, random_state=seed) if model == "logreg"
           else RandomForestClassifier(n_estimators=200, max_depth=8, random_state=seed, n_jobs=-1))
    clf.fit(Xtr, ytr)
    pva, pte = clf.predict_proba(Xva)[:, 1], clf.predict_proba(Xte)[:, 1]
    thr = _best_threshold(yva, pva)
    return {
        "model": model, "smote": use_smote,
        "train_rows_before": n_before, "train_rows_fit": int(len(ytr)),
        "train_positive_rate_fit": float(ytr.mean()),
        "validation": _metrics(yva, pva, thr), "test": _metrics(yte, pte, thr),
    }
