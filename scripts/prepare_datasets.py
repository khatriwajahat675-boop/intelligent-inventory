"""Validate both datasets, split 70/15/15, run the SMOTE experiment, write reports.

    PYTHONPATH=. python scripts/prepare_datasets.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.models.stock_risk import run_experiment
from ml.pipelines import data_prep as dp
from ml.pipelines import splits as sp

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT, REP = ROOT / "data/raw", ROOT / "data/processed", ROOT / "reports"
SEED = 42

FMCG_NUM = ["Units", "Cost_Price", "Selling_Price", "Lead_Time_Days", "month", "dow", "hour"]
FMCG_CAT = ["Category", "Brand", "City", "Store_Format", "Channel"]
# Stock_On_Hand / Reorder_Level define the target -> excluded (leakage).
GRO_NUM = ["Avg_Daily_Sales", "Lead_Time_Days", "Supplier_OnTime_Pct", "Order_Frequency_per_month",
           "Stock_Age_Days", "Unit_Cost_USD", "SKU_Churn_Rate", "Quantity_Reserved", "Returns_Qty"]
GRO_CAT = ["Category", "ABC_Class", "Warehouse_ID", "Supplier_ID"]


def drift(split: sp.SplitResult, cols: list[str]) -> dict:
    out = {}
    for c in cols:
        out[c] = {"val_psi": round(sp.population_stability_index(split.train[c], split.val[c]), 4),
                  "test_psi": round(sp.population_stability_index(split.train[c], split.test[c]), 4)}
    return out


def save(split: sp.SplitResult, name: str) -> None:
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    for part in ("train", "val", "test"):
        getattr(split, part).to_csv(d / f"{part}.csv", index=False)


def describe(split: sp.SplitResult, date_col: str | None) -> dict:
    n = sum(split.sizes().values())
    info = {}
    for part in ("train", "val", "test"):
        d = getattr(split, part)
        row = {"rows": len(d), "share": round(len(d) / n, 4), "stock_risk_rate": round(float(d["stock_risk"].mean()), 4)}
        if date_col:
            row["from"], row["to"] = str(d[date_col].min().date()), str(d[date_col].max().date())
        info[part] = row
    return info


def main() -> None:
    REP.mkdir(exist_ok=True)
    report: dict = {}

    # ---------------- FMCG (time series -> chronological) -----------------
    f = dp.load_fmcg(str(RAW / "indian_fmcg_retail_sales_2024.csv"))
    f["month"], f["dow"], f["hour"] = f.Invoice_Date.dt.month, f.Invoice_Date.dt.dayofweek, f.Invoice_Date.dt.hour
    fs = sp.chronological_split(f, "Invoice_Date")
    sp.assert_chronological(fs, "Invoice_Date")
    sp.assert_no_overlap(fs, "external_txn_id")
    save(fs, "fmcg")
    exps = [run_experiment(fs.train, fs.val, fs.test, FMCG_NUM, FMCG_CAT, use_smote=s, model=m, seed=SEED)
            for m in ("logreg", "rf") for s in (False, True)]
    report["fmcg"] = {
        "checks": [c.as_dict() for c in dp.validate_fmcg(f)],
        "split_method": fs.method, "split": describe(fs, "Invoice_Date"),
        "drift_psi": drift(fs, ["Units", "Selling_Price", "Lead_Time_Days", "Stock_On_Hand"]),
        "smote_experiment": exps,
    }

    # ---------------- Grocery (snapshot -> stratified) --------------------
    g = dp.load_grocery(str(RAW / "e_grocery_inventory.csv"))
    gs = sp.stratified_split(g, ["Category", "stock_risk"], seed=SEED)
    sp.assert_no_overlap(gs, "SKU_ID")
    save(gs, "grocery")
    gexps = [run_experiment(gs.train, gs.val, gs.test, GRO_NUM, GRO_CAT, use_smote=s, model=m, seed=SEED)
             for m in ("logreg", "rf") for s in (False, True)]
    report["grocery"] = {
        "checks": [c.as_dict() for c in dp.validate_grocery(g)],
        "split_method": gs.method, "split": describe(gs, None),
        "drift_psi": drift(gs, ["Avg_Daily_Sales", "Lead_Time_Days", "Unit_Cost_USD", "Safety_Stock"]),
        "smote_experiment": gexps,
    }

    # ---------------- demand series summary for forecasting ---------------
    daily = (f.groupby(["sku_code", f.Invoice_Date.dt.normalize()])["Units"].sum().rename("units").reset_index())
    daily.rename(columns={"Invoice_Date": "date"}, inplace=True)
    for name, part in (("train", fs.train), ("val", fs.val), ("test", fs.test)):
        p = daily[daily.date.between(part.Invoice_Date.min().normalize(), part.Invoice_Date.max().normalize())]
        p.to_csv(OUT / "fmcg" / f"daily_demand_{name}.csv", index=False)
    dow = f.groupby(f.Invoice_Date.dt.dayofweek)["Units"].sum()
    mon = f.groupby(f.Invoice_Date.dt.month)["Units"].sum()
    report["fmcg"]["seasonality_signal"] = {
        "weekday_units_cv": round(float(dow.std() / dow.mean()), 4),
        "month_units_cv": round(float(mon.std() / mon.mean()), 4),
        "note": "coefficient of variation of total units by weekday / month (0 = no seasonality)",
    }

    (REP / "data_validation_report.json").write_text(json.dumps(report, indent=2, default=str))
    write_markdown(report)
    print(json.dumps({k: v["split"] for k, v in report.items()}, indent=2))


def write_markdown(r: dict) -> None:
    L = ["# Data validation and 70/15/15 split report", ""]
    for name, title in (("fmcg", "Indian FMCG retail sales 2024"), ("grocery", "E-Grocery inventory snapshot")):
        s = r[name]
        L += [f"## {title}", "", f"Split method: **{s['split_method']}**", "",
              "| set | rows | share | at-risk rate | range |", "|---|---|---|---|---|"]
        for p, v in s["split"].items():
            L.append(f"| {p} | {v['rows']:,} | {v['share']:.1%} | {v['stock_risk_rate']:.2%} | {v.get('from', '-')} .. {v.get('to', '-')} |")
        L += ["", "| check | status | detail |", "|---|---|---|"]
        L += [f"| {c['name']} | {c['status']} | {c['detail']} |" for c in s["checks"]]
        L += ["", "Drift (PSI vs train; <0.1 stable): " + ", ".join(
            f"{k} val={v['val_psi']} test={v['test_psi']}" for k, v in s["drift_psi"].items()), "",
            "SMOTE experiment (train-only oversampling; threshold tuned on validation):", "",
            "| model | SMOTE | fit rows | pos. rate | val F1 | test precision | test recall | test F1 | test PR-AUC |",
            "|---|---|---|---|---|---|---|---|---|"]
        for e in s["smote_experiment"]:
            t = e["test"]
            L.append(f"| {e['model']} | {e['smote']} | {e['train_rows_fit']:,} | {e['train_positive_rate_fit']:.2f} | "
                     f"{e['validation']['f1']:.3f} | {t['precision']:.3f} | {t['recall']:.3f} | {t['f1']:.3f} | {t['pr_auc']:.3f} |")
        L.append("")
    if "seasonality_signal" in r["fmcg"]:
        L += ["## Seasonality signal in FMCG demand", "", json.dumps(r["fmcg"]["seasonality_signal"]), ""]
    (REP / "data_validation_report.md").write_text("\n".join(L))


if __name__ == "__main__":
    main()
