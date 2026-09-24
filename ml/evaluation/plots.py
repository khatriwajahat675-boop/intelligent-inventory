"""Comparison charts for the model zoo - F1/recall/precision/PR-AUC/ROC-AUC,
confusion matrices, ROC/PR curve overlays, confidence & support, feature
importance, and augmentation "progress" (volume growth + a learning curve).
Colors are the validated palette in ml/evaluation/palette.py (dataviz skill).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ml.evaluation.palette import CATEGORICAL, INK, MODEL_COLORS, SEQUENTIAL_BLUE, SURFACE, apply_style


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def plot_metric_bars(results: list[dict], title: str, out: Path) -> None:
    metrics = ["precision", "recall", "f1", "pr_auc", "roc_auc"]
    labels = ["Precision", "Recall", "F1", "PR-AUC", "ROC-AUC"]
    models = [r["model"] for r in results]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(metrics))
    width = 0.8 / len(models)
    for i, r in enumerate(results):
        vals = [r["test"][m] or 0 for m in metrics]
        ax.bar(x + i * width, vals, width, label=r["model"], color=MODEL_COLORS.get(r["model"], CATEGORICAL[i % 8]),
              edgecolor=SURFACE, linewidth=0.5, zorder=3)
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score (test set)")
    ax.set_title(title)
    apply_style(ax)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    _save(fig, out)


def plot_confusion_matrices(results: list[dict], title: str, out: Path) -> None:
    n = len(results)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.6 * rows))
    axes = np.atleast_1d(axes).ravel()
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq_blue", SEQUENTIAL_BLUE)
    for i, r in enumerate(results):
        ax = axes[i]
        cm = r["test"]["confusion_matrix"]
        mat = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
        ax.imshow(mat, cmap=cmap)
        for (yy, xx), v in np.ndenumerate(mat):
            ax.text(xx, yy, f"{v:,}", ha="center", va="center",
                   color=INK["primary"] if v < mat.max() * 0.6 else "white", fontsize=11, fontweight="bold")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["pred: OK", "pred: at-risk"], fontsize=8)
        ax.set_yticks([0, 1]); ax.set_yticklabels(["true: OK", "true: at-risk"], fontsize=8)
        ax.set_title(r["model"], fontsize=10, color=INK["primary"])
        for s in ax.spines.values():
            s.set_visible(False)
    for j in range(len(results), len(axes)):
        axes[j].axis("off")
    fig.suptitle(title, color=INK["primary"])
    fig.tight_layout()
    _save(fig, out)


def plot_curve_overlay(results: list[dict], kind: str, title: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for r in results:
        c = r["test_curves"][kind]
        xkey, ykey = ("fpr", "tpr") if kind == "roc" else ("recall", "precision")
        ax.plot(c[xkey], c[ykey], color=MODEL_COLORS.get(r["model"], CATEGORICAL[0]), linewidth=2,
               label=f"{r['model']} (AUC={r['test']['roc_auc' if kind == 'roc' else 'pr_auc']})")
    if kind == "roc":
        ax.plot([0, 1], [0, 1], "--", color=INK["muted"], linewidth=1)
        ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
    else:
        ax.set_xlabel("recall"); ax.set_ylabel("precision")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_title(title)
    apply_style(ax)
    ax.legend(frameon=False, fontsize=8, loc="lower left" if kind == "roc" else "lower left")
    _save(fig, out)


def plot_confidence_support(results: list[dict], title: str, out: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    models = [r["model"] for r in results]
    conf = [r["test"]["confidence_mean"] for r in results]
    colors = [MODEL_COLORS.get(m, CATEGORICAL[0]) for m in models]
    ax1.bar(models, conf, color=colors, edgecolor=SURFACE, zorder=3)
    ax1.set_ylim(0, 1.05); ax1.set_ylabel("mean confidence on predicted class")
    ax1.set_title("Prediction confidence"); ax1.tick_params(axis="x", rotation=30)
    apply_style(ax1)

    sup_neg = results[0]["test"]["support_negative"]
    sup_pos = results[0]["test"]["support_positive"]
    ax2.bar(["OK (negative)", "at-risk (positive)"], [sup_neg, sup_pos],
           color=[INK["muted"], "#d03b3b"], edgecolor=SURFACE, zorder=3)
    for i, v in enumerate([sup_neg, sup_pos]):
        ax2.text(i, v, f"{v:,}", ha="center", va="bottom", color=INK["primary"], fontsize=10)
    ax2.set_title("Test-set class support (same for every model)")
    apply_style(ax2)
    fig.suptitle(title, color=INK["primary"])
    fig.tight_layout()
    _save(fig, out)


def plot_feature_importance(explanation: dict, model_name: str, out: Path, top_n: int = 12) -> None:
    ranked = explanation["ranked_features"][:top_n][::-1]
    fig, ax = plt.subplots(figsize=(7, 0.4 * len(ranked) + 1.5))
    vals = [r["importance"] for r in ranked]
    names = [r["feature"] for r in ranked]
    ax.barh(names, vals, color=CATEGORICAL[0], zorder=3)
    method_label = "SHAP |value|" if explanation["method"] == "shap" else "permutation importance"
    ax.set_xlabel(f"importance ({method_label})")
    ax.set_title(f"Feature importance - {model_name}")
    apply_style(ax)
    ax.yaxis.grid(False)
    fig.tight_layout()
    _save(fig, out)


def plot_augmentation_volume(report: dict, out: Path) -> None:
    names = [d["dataset"] for d in report["datasets"]]
    real = [d["real_rows"] for d in report["datasets"]]
    synth = [d["synthetic_rows_generated"] for d in report["datasets"]]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(names))
    ax.bar(x, real, color=CATEGORICAL[0], label="real (train split)", zorder=3)
    ax.bar(x, synth, bottom=real, color=CATEGORICAL[1], label="synthetic (retrieval-grounded)", zorder=3)
    for i, (r, s) in enumerate(zip(real, synth)):
        ax.text(i, r + s, f"{r + s:,}", ha="center", va="bottom", fontsize=9, color=INK["primary"])
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("training rows")
    ax.set_title("Augmentation volume: real vs. synthetic (train split only)")
    apply_style(ax)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    _save(fig, out)


def plot_learning_curve(fractions: list[float], f1_scores: list[float], model_name: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(fractions, f1_scores, marker="o", color=CATEGORICAL[0], linewidth=2, markersize=6, zorder=3)
    ax.set_xlabel("fraction of augmented training data used")
    ax.set_ylabel("test F1 score")
    ax.set_title(f"Training-volume progress curve - {model_name}")
    ax.set_ylim(0, 1.05)
    apply_style(ax)
    fig.tight_layout()
    _save(fig, out)
