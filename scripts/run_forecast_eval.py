"""Per-SKU forecasting evaluation on the 70/15/15 chronological split.

Select on validation, refit on train+val, report on the untouched test window.
    PYTHONPATH=. python scripts/run_forecast_eval.py
"""
import json
from pathlib import Path

import pandas as pd

from ml.evaluation import selection as sel
from ml.evaluation.metrics import wape

P = Path(__file__).resolve().parents[1] / "data/processed/fmcg"


def series(name: str) -> pd.DataFrame:
    d = pd.read_csv(P / f"daily_demand_{name}.csv", parse_dates=["date"])
    return d.pivot_table(index="date", columns="sku_code", values="units", aggfunc="sum")


def main() -> None:
    tr, va, te = series("train"), series("val"), series("test")
    rows = []
    for sku in tr.columns:
        s = sel.select_on_validation(tr[sku].dropna(), va[sku].dropna())
        full = pd.concat([tr[sku], va[sku]]).dropna()
        res = sel.evaluate_on_test(s, full, te[sku].dropna())
        rows.append({"sku": sku, "model": s.model, "status": s.status, "flag": s.quality_flag,
                     "test_wape_model": res[s.model]["wape"], "test_wape_naive": res["naive"]["wape"],
                     "test_smape_model": res[s.model]["smape"], "failures": len(s.failures)})
    df = pd.DataFrame(rows)
    df.to_csv(P.parents[2] / "reports/forecast_eval_by_sku.csv", index=False)
    summ = {"skus": len(df), "models_selected": df.model.value_counts().to_dict(),
            "status": df.status.value_counts().to_dict(),
            "mean_test_wape_selected": round(df.test_wape_model.mean(), 4),
            "mean_test_wape_naive": round(df.test_wape_naive.mean(), 4),
            "skus_where_selected_beats_naive_on_test": int((df.test_wape_model < df.test_wape_naive).sum()),
            "note": "ARIMA/Prophet skipped automatically if statsmodels/prophet are not installed (see failures)"}
    (P.parents[2] / "reports/forecast_eval_summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
