"""Full multi-model comparison for the stock-risk / diminishing-stock classifier.

Trains every model in ml.models.model_zoo on the SAME encoded features (fit on
TRAIN only), tunes one decision threshold per model on VALIDATION (max F1),
and reports one held-out TEST scorecard per model: precision, recall, F1,
support (per class), ROC-AUC, PR-AUC, a full confusion matrix (TP/FP/TN/FN),
and a confidence measure (mean probability mass the model puts on whatever
class it predicted - a calibration-style number, not an accuracy).

TRAIN is the retrieval-grounded-augmented + SMOTE-balanced set; VAL/TEST are
always 100% real, untouched data (see docs/AUGMENTATION.md) - otherwise every
number here would measure how well a model recognises its own synthetic
near-duplicates, not genuine generalisation.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve,
                             precision_recall_curve)

from ml.models.model_zoo import MODEL_NAMES, build_model
from ml.models.stock_risk import Design
from ml.pipelines.smote import smote


def best_threshold(y: np.ndarray, p: np.ndarray) -> float:
    grid = np.unique(np.quantile(p, np.linspace(0.02, 0.98, 97)))
    scores = [f1_score(y, (p >= t).astype(int), zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


def full_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    confidence = float(np.mean(np.where(pred == 1, p, 1 - p)))       # mean prob mass on the predicted class
    return {
        "threshold": round(float(threshold), 4),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y, p)), 4) if 0 < y.sum() < len(y) else None,
        "pr_auc": round(float(average_precision_score(y, p)), 4),
        "confidence_mean": round(confidence, 4),
        "support_negative": int((y == 0).sum()), "support_positive": int((y == 1).sum()),
        "confusion_matrix": {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)},
    }


def curves(y: np.ndarray, p: np.ndarray) -> dict:
    fpr, tpr, _ = roc_curve(y, p)
    prec, rec, _ = precision_recall_curve(y, p)
    step = max(1, len(fpr) // 60)                       # thin points for compact JSON/plots
    return {"roc": {"fpr": fpr[::step].round(4).tolist(), "tpr": tpr[::step].round(4).tolist()},
            "pr": {"recall": rec[::step].round(4).tolist(), "precision": prec[::step].round(4).tolist()}}


def compare_all(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, num_cols: list[str],
                cat_cols: list[str], target: str = "stock_risk", use_smote: bool = True,
                max_train_rows: int | None = None, seed: int = 42) -> dict:
    if max_train_rows and len(train) > max_train_rows:
        parts = []
        for _, g in train.groupby(target):
            n = max(1, int(round(max_train_rows * len(g) / len(train))))
            parts.append(g.sample(n=min(n, len(g)), random_state=seed))
        train = pd.concat(parts, ignore_index=True)
    design = Design(num_cols, cat_cols).fit(train)
    Xtr, Xva, Xte = (design.transform(d) for d in (train, val, test))
    ytr, yva, yte = (d[target].to_numpy() for d in (train, val, test))
    n_before = len(ytr)
    if use_smote:
        Xtr, ytr = smote(Xtr, ytr, k_neighbors=5, seed=seed)

    results, fitted = [], {}
    for name in MODEL_NAMES:
        t0 = time.perf_counter()
        clf = build_model(name, seed=seed)
        clf.fit(Xtr, ytr)
        fit_s = time.perf_counter() - t0
        pva = clf.predict_proba(Xva)[:, 1]
        pte = clf.predict_proba(Xte)[:, 1]
        thr = best_threshold(yva, pva)
        row = {"model": name, "fit_seconds": round(fit_s, 3), "train_rows_used": int(len(ytr)),
              "train_positive_rate": round(float(ytr.mean()), 4),
              "validation": full_metrics(yva, pva, thr), "test": full_metrics(yte, pte, thr),
              "test_curves": curves(yte, pte)}
        results.append(row)
        fitted[name] = (clf, pte)
    return {"design": design, "features": design.feature_names(), "n_train_before_smote": n_before,
            "use_smote": use_smote, "max_train_rows": max_train_rows, "results": results, "fitted": fitted,
            "X_test": Xte, "y_test": yte, "X_val": Xva, "y_val": yva}
