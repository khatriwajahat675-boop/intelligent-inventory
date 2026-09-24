"""Loading, cleaning and validating the two source datasets.

Dataset A - Indian FMCG retail sales 2024 (100k invoice lines, a real time axis)
    -> demand history (sales_transactions) for forecasting.
    SKU proxy       = "<Category>|<Brand>"  (64 SKUs)
    Warehouse proxy = City                  (8 locations)

Dataset B - E-Grocery inventory snapshot (1000 SKUs, ONE row per SKU, no sales
    history) -> product / supplier / warehouse master data, opening balances and
    reorder policy.  Numbers use European formatting ("$1.234,56", "70,68%").
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Check:
    name: str
    status: str  # PASS | WARN | FAIL
    detail: str

    def as_dict(self) -> dict:
        return {"name": self.name, "status": self.status, "detail": self.detail}


def parse_eu_number(s: pd.Series) -> pd.Series:
    """'$2.084,25' -> 2084.25 ; '70,68%' -> 70.68 ; '1.377' -> 1377.0"""
    cleaned = (
        s.astype(str).str.replace(r"[$%\s]", "", regex=True)
        .str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    )
    return pd.to_numeric(cleaned, errors="coerce")


# ----------------------------------------------------------------- FMCG ----
def load_fmcg(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Invoice_Date"] = pd.to_datetime(df["Invoice_Date"], errors="coerce")
    df = df.rename(columns={"Margin_%": "Margin_Pct"})
    df["sku_code"] = df["Category"].str.strip() + "|" + df["Brand"].str.strip()
    df["warehouse_code"] = df["City"].str.strip()
    # Invoice_ID collides (random 8-digit ids) -> build an idempotent txn id from
    # the full line content, so a genuine replay is detected but a collision is not.
    key = df[["Invoice_ID", "Invoice_Date", "sku_code", "warehouse_code", "Units",
              "Selling_Price", "Channel", "Payment_Mode"]].astype(str).agg("|".join, axis=1)
    df["external_txn_id"] = key.map(lambda s: hashlib.sha1(s.encode()).hexdigest()[:16])
    # target for the stock-risk classifier: stock at/below reorder level.
    df["stock_risk"] = (df["Stock_On_Hand"] <= df["Reorder_Level"]).astype(int)
    return df


def validate_fmcg(df: pd.DataFrame) -> list[Check]:
    c: list[Check] = []
    add = lambda n, ok, d, warn=False: c.append(  # noqa: E731
        Check(n, "PASS" if ok else ("WARN" if warn else "FAIL"), d))
    add("row count", len(df) > 0, f"{len(df):,} rows, {df.shape[1]} columns")
    add("dates parse", df["Invoice_Date"].notna().all(),
        f"range {df['Invoice_Date'].min()} .. {df['Invoice_Date'].max()}")
    add("no future-dated sales (EC-05)", (df["Invoice_Date"] <= pd.Timestamp.now()).all(), "ok")
    add("units > 0 (EC-03)", (df["Units"] > 0).all(), f"min={df['Units'].min()}")
    add("prices > 0", ((df["Cost_Price"] > 0) & (df["Selling_Price"] > 0)).all(), "ok")
    add("selling >= cost", (df["Selling_Price"] >= df["Cost_Price"]).all(), "no negative margin")
    add("lead time > 0 (FR-010 / EC-16)", (df["Lead_Time_Days"] > 0).all(),
        f"{df['Lead_Time_Days'].min()}..{df['Lead_Time_Days'].max()} days")
    add("stock on hand >= 0 (EC-11)", (df["Stock_On_Hand"] >= 0).all(), "ok")
    rev_bad = int(((df["Revenue"] - df["Units"] * df["Selling_Price"]).abs() > 0.01).sum())
    add("revenue = units x price", rev_bad == 0, f"{rev_bad} mismatches")
    dup_full = int(df.drop(columns=["external_txn_id"]).duplicated().sum())
    add("no fully duplicated lines (EC-02)", dup_full == 0, f"{dup_full} duplicates")
    id_dup = int(df["Invoice_ID"].duplicated().sum())
    add("Invoice_ID unique", id_dup == 0,
        f"{id_dup} repeated ids with different content -> id collisions, not replays; "
        "composite external_txn_id generated", warn=True)
    add("external_txn_id unique", df["external_txn_id"].is_unique, "composite idempotency key")
    for col in ("Customer_Age", "Customer_Gender"):
        pct = df[col].isna().mean() * 100
        add(f"missing {col}", pct == 0, f"{pct:.1f}% null (not used by forecasting)", warn=True)
    daily = df.groupby(["sku_code", df["Invoice_Date"].dt.normalize()])["Units"].sum()
    n_days = df["Invoice_Date"].dt.normalize().nunique()
    zero_share = 1 - len(daily) / (df["sku_code"].nunique() * n_days)
    add("intermittent demand (EC-52)", zero_share < 0.3,
        f"{zero_share:.1%} of SKU-days have zero sales", warn=True)
    return c


# -------------------------------------------------------------- GROCERY ----
_EU_NUMERIC = [
    "Avg_Daily_Sales", "Forecast_Next_30d", "Days_of_Inventory", "Unit_Cost_USD",
    "Last_Purchase_Price_USD", "Total_Inventory_Value_USD", "SKU_Churn_Rate",
    "Order_Frequency_per_month", "Supplier_OnTime_Pct", "Audit_Variance_Pct",
    "Demand_Forecast_Accuracy_Pct",
]
_PLAIN_NUMERIC = [
    "Stock_Age_Days", "Quantity_On_Hand", "Quantity_Reserved", "Quantity_Committed",
    "Damaged_Qty", "Returns_Qty", "Reorder_Point", "Safety_Stock", "Lead_Time_Days",
    "Count_Variance",
]


def load_grocery(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    for col in _EU_NUMERIC:
        df[col] = parse_eu_number(df[col])
    for col in _PLAIN_NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ("Received_Date", "Last_Purchase_Date", "Expiry_Date", "Audit_Date"):
        df[col] = pd.to_datetime(df[col], errors="coerce")
    df["Available"] = (df["Quantity_On_Hand"] - df["Quantity_Reserved"]
                       - df["Damaged_Qty"]).clip(lower=0)
    # binary target: Low Stock / Out of Stock = at risk.  (Out of Stock has only 2
    # rows, so the 4-class problem cannot be split 70/15/15 or oversampled sensibly.)
    df["stock_risk"] = df["Inventory_Status"].isin(["Low Stock", "Out of Stock"]).astype(int)
    return df


def validate_grocery(df: pd.DataFrame) -> list[Check]:
    c: list[Check] = []
    add = lambda n, ok, d, warn=False: c.append(  # noqa: E731
        Check(n, "PASS" if ok else ("WARN" if warn else "FAIL"), d))
    add("row count", len(df) > 0, f"{len(df):,} rows, {df.shape[1]} columns")
    add("SKU_ID unique (unique constraint, TRD 11.2)", df["SKU_ID"].is_unique, "1 row per SKU")
    num_cols = _EU_NUMERIC + _PLAIN_NUMERIC
    bad = {k: int(df[k].isna().sum()) for k in num_cols if df[k].isna().any()}
    add("numeric fields parse (EU format)", not bad, f"unparseable: {bad or 'none'}")
    dcols = ["Received_Date", "Last_Purchase_Date", "Expiry_Date", "Audit_Date"]
    add("dates parse", df[dcols].notna().all().all(), "ok")
    add("expiry after receipt", (df["Expiry_Date"] >= df["Received_Date"]).all(), "ok")
    add("lead time > 0 (EC-16)", (df["Lead_Time_Days"] > 0).all(),
        f"{df['Lead_Time_Days'].min():.0f}..{df['Lead_Time_Days'].max():.0f} days")
    add("safety stock >= 0 (EC-59)", (df["Safety_Stock"] >= 0).all(), "ok")
    add("on-hand >= 0 (EC-11)", (df["Quantity_On_Hand"] >= 0).all(), "ok")
    n_res = int((df["Quantity_Reserved"] > df["Quantity_On_Hand"]).sum())
    add("reserved <= on-hand (FR-005)", n_res == 0,
        f"{n_res} SKUs reserve more than on-hand -> available clamped to 0", warn=True)
    add("supplier id -> name is 1:1", df.groupby("Supplier_ID")["Supplier_Name"].nunique().max() == 1, "ok")
    add("cost > 0", (df["Unit_Cost_USD"] > 0).all(), "ok (needed for order-value limit, EC-64)")
    vc = df["Inventory_Status"].value_counts()
    add("class balance (target=stock_risk)", df["stock_risk"].mean() > 0.1,
        f"at-risk {df['stock_risk'].mean():.1%}; raw status counts {vc.to_dict()}", warn=True)
    add("no sales history in file", False,
        "single snapshot per SKU -> cannot train ARIMA/Prophet on it; used for master data, "
        "opening balances and the stock-risk classifier", warn=True)
    return c
