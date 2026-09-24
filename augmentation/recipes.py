"""Dataset-specific synthesis recipes: which fields interpolate, which are copied
atomically, and how derived/business fields are recomputed and clipped after
synthesis so augmented rows pass the SAME validation checks as real ones
(ml/pipelines/data_prep.validate_fmcg / validate_grocery)."""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from augmentation.synthesize import SynthesisRecipe


def _new_ids(n: int, prefix: str, start: int = 0) -> pd.Series:
    return pd.Series([f"{prefix}{i:09d}" for i in range(start, start + n)])


# --------------------------------------------------------------------- FMCG ---
FMCG_NUMERIC = ["Units", "Cost_Price", "Selling_Price", "Lead_Time_Days", "Stock_On_Hand",
                "Reorder_Level", "Customer_Age"]
FMCG_CATEGORICAL = ["City", "Store_Format", "Category", "Brand", "Channel", "Payment_Mode",
                    "Customer_Gender", "Loyalty_Flag"]
FMCG_INHERIT = ["Invoice_Date"]          # dates copied atomically -> preserves real calendar shape exactly
FMCG_CONTEXT = ["Category", "Brand", "City", "Store_Format", "Channel"]


def _fmcg_postprocess(synth: pd.DataFrame) -> pd.DataFrame:
    d = synth.copy()
    d["Units"] = d["Units"].round().clip(lower=1, upper=5).astype(int)                      # matches real range 1..5
    d["Lead_Time_Days"] = d["Lead_Time_Days"].round().clip(lower=3, upper=14).astype(int)    # EC-16
    d["Stock_On_Hand"] = d["Stock_On_Hand"].round().clip(lower=0).astype(int)                # EC-11
    d["Reorder_Level"] = d["Reorder_Level"].round().clip(lower=1).astype(int)
    d["Cost_Price"] = d["Cost_Price"].clip(lower=1.0)
    d["Selling_Price"] = np.maximum(d["Selling_Price"], d["Cost_Price"] * 1.02)              # no negative margin
    if "Customer_Age" in d:
        d["Customer_Age"] = d["Customer_Age"].round().clip(lower=18, upper=64)
    d["Revenue"] = d["Units"] * d["Selling_Price"]                                           # recompute, never interpolate
    d["Cost"] = d["Units"] * d["Cost_Price"]
    d["Margin"] = d["Revenue"] - d["Cost"]
    d["Margin_Pct"] = d["Margin"] / d["Revenue"]
    d["sku_code"] = d["Category"].str.strip() + "|" + d["Brand"].str.strip()
    d["warehouse_code"] = d["City"].str.strip()
    inv_date = pd.to_datetime(d["Invoice_Date"])          # inherited atomically -> recompute calendar features
    d["month"], d["dow"], d["hour"] = inv_date.dt.month, inv_date.dt.dayofweek, inv_date.dt.hour
    ids = _new_ids(len(d), "SYN")
    key = (ids + "|" + d["Invoice_Date"].astype(str) + "|" + d["sku_code"] + "|" + d["warehouse_code"]
          + "|" + d["Units"].astype(str))
    d["Invoice_ID"] = -(np.arange(len(d)) + 1)                     # negative range: never collides with real 8-digit ids
    d["external_txn_id"] = key.map(lambda s: hashlib.sha1(s.encode()).hexdigest()[:16])
    d["stock_risk"] = (d["Stock_On_Hand"] <= d["Reorder_Level"]).astype(int)
    return d


FMCG_RECIPE = SynthesisRecipe("fmcg", FMCG_NUMERIC, FMCG_CATEGORICAL, FMCG_INHERIT, FMCG_CONTEXT,
                              "sku_code", _fmcg_postprocess)


# ----------------------------------------------------------------- GROCERY ---
GRO_NUMERIC = ["Avg_Daily_Sales", "Forecast_Next_30d", "Stock_Age_Days", "Quantity_On_Hand",
              "Quantity_Reserved", "Quantity_Committed", "Damaged_Qty", "Returns_Qty", "Reorder_Point",
              "Safety_Stock", "Lead_Time_Days", "Unit_Cost_USD", "Last_Purchase_Price_USD",
              "SKU_Churn_Rate", "Order_Frequency_per_month", "Supplier_OnTime_Pct",
              "Demand_Forecast_Accuracy_Pct"]
GRO_CATEGORICAL = ["Category", "ABC_Class", "Supplier_ID", "Supplier_Name", "Warehouse_ID",
                   "Warehouse_Location", "FIFO_FEFO"]
# dates are copied atomically (never interpolated) so Received <= Expiry etc. always stays internally consistent
GRO_INHERIT = ["Received_Date", "Last_Purchase_Date", "Expiry_Date", "Audit_Date"]
GRO_CONTEXT = ["Category", "ABC_Class", "Warehouse_ID", "Supplier_ID"]


def _gro_postprocess(synth: pd.DataFrame) -> pd.DataFrame:
    d = synth.copy()
    d["Lead_Time_Days"] = d["Lead_Time_Days"].round().clip(lower=1, upper=14).astype(int)
    d["Safety_Stock"] = d["Safety_Stock"].round().clip(lower=0).astype(int)
    d["Reorder_Point"] = d["Reorder_Point"].round().clip(lower=1).astype(int)
    for c in ("Quantity_On_Hand", "Quantity_Reserved", "Quantity_Committed", "Damaged_Qty",
             "Returns_Qty", "Stock_Age_Days"):
        d[c] = d[c].round().clip(lower=0).astype(int)
    d["Quantity_Reserved"] = np.minimum(d["Quantity_Reserved"], d["Quantity_On_Hand"])       # FR-005
    d["Unit_Cost_USD"] = d["Unit_Cost_USD"].clip(lower=0.05)
    d["Last_Purchase_Price_USD"] = d["Last_Purchase_Price_USD"].clip(lower=0.05)
    d["Supplier_OnTime_Pct"] = d["Supplier_OnTime_Pct"].clip(lower=0, upper=100)
    d["Demand_Forecast_Accuracy_Pct"] = d["Demand_Forecast_Accuracy_Pct"].clip(lower=0, upper=100)
    d["Avg_Daily_Sales"] = d["Avg_Daily_Sales"].clip(lower=0.01)
    d["Forecast_Next_30d"] = d["Forecast_Next_30d"].clip(lower=0)
    d["Total_Inventory_Value_USD"] = d["Quantity_On_Hand"] * d["Unit_Cost_USD"]              # recompute, not interpolate
    # recompute status from the synthesised quantities so it stays internally consistent (never inherited stale)
    below_safety = d["Quantity_On_Hand"] <= (d["Reorder_Point"])
    out_of_stock = d["Quantity_On_Hand"] <= 0
    d["Inventory_Status"] = np.select([out_of_stock, below_safety], ["Out of Stock", "Low Stock"], default="In Stock")
    d["stock_risk"] = d["Inventory_Status"].isin(["Low Stock", "Out of Stock"]).astype(int)
    d["Available"] = (d["Quantity_On_Hand"] - d["Quantity_Reserved"] - d["Damaged_Qty"]).clip(lower=0)
    # not sampled/interpolated - derived (Days_of_Inventory) or genuinely inapplicable to a synthetic
    # SKU that has never been physically audited (Count_Variance, Audit_Variance_Pct -> neutral 0)
    d["Days_of_Inventory"] = (d["Quantity_On_Hand"] / d["Avg_Daily_Sales"].replace(0, np.nan)).fillna(0).round(2)
    d["Count_Variance"] = 0
    d["Audit_Variance_Pct"] = 0.0
    ids = _new_ids(len(d), "SKU-SYN-")
    d["SKU_ID"] = ids
    d["SKU_Name"] = "Synthetic " + d["Category"].astype(str) + " Product " + pd.Series(range(len(d))).astype(str)
    d["Batch_ID"] = "BATCH-SYN-" + pd.Series(range(len(d))).astype(str)
    return d


GROCERY_RECIPE = SynthesisRecipe("grocery", GRO_NUMERIC, GRO_CATEGORICAL, GRO_INHERIT, GRO_CONTEXT,
                                 "Category", _gro_postprocess)
