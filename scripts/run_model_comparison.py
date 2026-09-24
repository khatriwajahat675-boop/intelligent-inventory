"""Train every model in the zoo on the augmented+SMOTE-balanced TRAIN split,
evaluate on 100% real VAL/TEST, run interpretability, and render every
comparison chart. Primary run is the grocery stock-risk classifier (the
dataset with a learnable label); FMCG is run too, for completeness, with its
known "label is per-invoice noise" limitation stated plainly in the report.

    PYTHONPATH=. python scripts/run_model_comparison.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ml.evaluation import compare_models as cm
from ml.evaluation import interpretability as interp
from ml.evaluation import plots

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data/processed"
REP = ROOT / "reports/model_comparison"
PLOTS = REP / "plots"

GRO_NUM = ["Avg_Daily_Sales", "Lead_Time_Days", "Supplier_OnTime_Pct", "Order_Frequency_per_month",
          "Stock_Age_Days", "Unit_Cost_USD", "SKU_Churn_Rate", "Quantity_Reserved", "Returns_Qty"]
GRO_CAT = ["Category", "ABC_Class", "Warehouse_ID", "Supplier_ID"]
FMCG_NUM = ["Units", "Cost_Price", "Selling_Price", "Lead_Time_Days", "month", "dow", "hour"]
FMCG_CAT = ["Category", "Brand", "City", "Store_Format", "Channel"]


def load(dataset: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    d = PROC / dataset
    train = pd.read_csv(d / "train_augmented.csv.gz")
    val, test = pd.read_csv(d / "val.csv"), pd.read_csv(d / "test.csv")
    return train, val, test


def run_dataset(name: str, num_cols: list[str], cat_cols: list[str], max_train_rows: int, limitation: str | None) -> dict:
    train, val, test = load(name)
    out = cm.compare_all(train, val, test, num_cols, cat_cols, max_train_rows=max_train_rows)
    results = out["results"]

    plots.plot_metric_bars(results, f"{name}: model comparison (test set)", PLOTS / f"{name}_metric_bars.png")
    plots.plot_confusion_matrices(results, f"{name}: confusion matrices (test set)", PLOTS / f"{name}_confusion.png")
    plots.plot_curve_overlay(results, "roc", f"{name}: ROC curves", PLOTS / f"{name}_roc.png")
    plots.plot_curve_overlay(results, "pr", f"{name}: precision-recall curves", PLOTS / f"{name}_pr.png")
    plots.plot_confidence_support(results, f"{name}: confidence & support", PLOTS / f"{name}_confidence_support.png")

    best = max(results, key=lambda r: r["test"]["f1"])
    clf, _ = out["fitted"][best["model"]]
    expl = interp.explain(clf, out["X_val"], out["X_test"], out["y_test"], out["features"], best["model"])
    plots.plot_feature_importance(expl, best["model"], PLOTS / f"{name}_feature_importance.png")

    # training-volume "progress" curve for the best model: test F1 as a growing
    # fraction of the augmented+SMOTE-balanced train set is used to fit it
    lc_fractions, lc_f1 = run_learning_curve(train, val, test, num_cols, cat_cols, best["model"], max_train_rows)
    plots.plot_learning_curve(lc_fractions, lc_f1, best["model"], PLOTS / f"{name}_learning_curve.png")

    report = {"dataset": name, "num_cols": num_cols, "cat_cols": cat_cols, "max_train_rows": max_train_rows,
             "n_train_before_smote": out["n_train_before_smote"], "known_limitation": limitation,
             "best_model_by_test_f1": best["model"], "interpretability": expl,
             "learning_curve": {"fractions": lc_fractions, "test_f1": lc_f1},
             "results": [{k: v for k, v in r.items() if k != "test_curves"} for r in results]}
    REP.mkdir(parents=True, exist_ok=True)
    (REP / f"{name}_comparison.json").write_text(json.dumps(report, indent=2, default=str))
    return report


def run_learning_curve(train, val, test, num_cols, cat_cols, model_name, max_train_rows):
    """Fit the best model on growing fractions of TRAIN, score on the fixed real TEST set."""
    from ml.evaluation.compare_models import best_threshold, full_metrics
    from ml.models.model_zoo import build_model
    from ml.models.stock_risk import Design
    from ml.pipelines.smote import smote
    import numpy as np

    if max_train_rows and len(train) > max_train_rows:
        parts = []
        for _, g in train.groupby("stock_risk"):
            n = max(1, int(round(max_train_rows * len(g) / len(train))))
            parts.append(g.sample(n=min(n, len(g)), random_state=42))
        train = pd.concat(parts, ignore_index=True)

    design = Design(num_cols, cat_cols).fit(train)
    Xtr_all, Xva, Xte = design.transform(train), design.transform(val), design.transform(test)
    ytr_all, yva, yte = train["stock_risk"].to_numpy(), val["stock_risk"].to_numpy(), test["stock_risk"].to_numpy()
    rng = np.random.default_rng(0)
    fractions, f1s = [0.05, 0.1, 0.25, 0.5, 1.0], []
    for frac in fractions:
        n = max(50, int(len(Xtr_all) * frac))
        idx = rng.choice(len(Xtr_all), size=n, replace=False)
        Xtr, ytr = Xtr_all[idx], ytr_all[idx]
        if ytr.sum() >= 2 and (len(ytr) - ytr.sum()) >= 2:
            Xtr, ytr = smote(Xtr, ytr, k_neighbors=min(5, int(ytr.sum()) - 1), seed=0)
        clf = build_model(model_name, seed=0)
        clf.fit(Xtr, ytr)
        pva, pte = clf.predict_proba(Xva)[:, 1], clf.predict_proba(Xte)[:, 1]
        thr = best_threshold(yva, pva)
        f1s.append(full_metrics(yte, pte, thr)["f1"])
    return fractions, f1s


def write_markdown() -> None:
    L = ["# Model comparison report", "",
        "All results are on the **held-out, 100% real TEST split** - TRAIN was the retrieval-grounded-"
        "augmented + SMOTE-balanced set (see docs/AUGMENTATION.md). VAL/TEST were never augmented.", ""]
    for f in sorted(REP.glob("*_comparison.json")):
        r = json.loads(f.read_text())
        L += [f"## {r['dataset']}", ""]
        if r["known_limitation"]:
            L += [f"> **Known limitation:** {r['known_limitation']}", ""]
        L += [f"Training rows before SMOTE (post-augmentation, capped at {r['max_train_rows']:,} for tractability): "
             f"{r['n_train_before_smote']:,}. Best model by test F1: **{r['best_model_by_test_f1']}**.", "",
             "| model | precision | recall | f1 | roc-auc | pr-auc | confidence | TP | FP | TN | FN | fit(s) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for res in r["results"]:
            t, cmx = res["test"], res["test"]["confusion_matrix"]
            L.append(f"| {res['model']} | {t['precision']} | {t['recall']} | {t['f1']} | {t['roc_auc']} | "
                     f"{t['pr_auc']} | {t['confidence_mean']} | {cmx['tp']} | {cmx['fp']} | {cmx['tn']} | "
                     f"{cmx['fn']} | {res['fit_seconds']} |")
        L += ["", f"Test-set support: {r['results'][0]['test']['support_negative']:,} negative "
             f"(OK), {r['results'][0]['test']['support_positive']:,} positive (at-risk).", "",
             f"Interpretability method: **{r['interpretability']['method']}**. Top features for "
             f"{r['best_model_by_test_f1']}:", ""]
        for feat in r["interpretability"]["ranked_features"][:8]:
            L.append(f"- `{feat['feature']}`: {feat['importance']}")
        L += ["", f"![metrics](plots/{r['dataset']}_metric_bars.png)", f"![confusion](plots/{r['dataset']}_confusion.png)",
             f"![roc](plots/{r['dataset']}_roc.png)", f"![pr](plots/{r['dataset']}_pr.png)",
             f"![confidence](plots/{r['dataset']}_confidence_support.png)",
             f"![importance](plots/{r['dataset']}_feature_importance.png)",
             f"![learning curve](plots/{r['dataset']}_learning_curve.png)", ""]
    (REP / "model_comparison_report.md").write_text("\n".join(L))


def main() -> None:
    aug = json.loads((ROOT / "reports/augmentation/augmentation_report.json").read_text())
    plots.plot_augmentation_volume(aug, PLOTS / "augmentation_volume.png")

    run_dataset("grocery", GRO_NUM, GRO_CAT, max_train_rows=20_506, limitation=None)
    run_dataset("fmcg", FMCG_NUM, FMCG_CAT, max_train_rows=150_000,
               limitation="Stock_On_Hand/Reorder_Level in the source FMCG data are independent random "
               "draws per invoice line (documented in docs/STATUS.md); the resulting stock_risk label "
               "carries no learnable signal beyond the base rate even after augmentation. Reported here "
               "for completeness and honesty, not as a claim of a working classifier.")
    write_markdown()
    print("done:", [p.name for p in REP.glob("*.json")])


if __name__ == "__main__":
    main()
