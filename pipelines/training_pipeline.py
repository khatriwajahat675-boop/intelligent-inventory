"""PIPELINE A - training.

1. Trains the full model zoo on the retrieval-grounded-augmented + SMOTE-
   balanced TRAIN split, evaluates on real VAL/TEST (reuses
   scripts/run_model_comparison.py - see there for the full metric/plot code)
   and persists every model's scorecard, plus which one is best, to the
   MongoDB `model_registry` collection.
2. Seeds the operational MongoDB collections (products, suppliers, inventory
   snapshots, sales transactions) from the REAL data only - never the
   synthetic augmentation - so Pipeline B (pipelines/automation_pipeline.py)
   has realistic data to run alerting/restock automation against. Mixing
   synthetic rows into the collections a live system reads from would be a
   data-integrity problem, not just a modelling one; augmentation stays
   confined to `data/processed/*/train_augmented.csv.gz` and this pipeline's
   own model-training step.

    PYTHONPATH=. python pipelines/training_pipeline.py                  # in-memory (no Atlas needed)
    PYTHONPATH=. python pipelines/training_pipeline.py --atlas          # requires MONGODB_URI in .env
    PYTHONPATH=. python pipelines/training_pipeline.py --dataset both --atlas
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from mongo.repository import Repositories
from mongo.schemas import InventorySnapshotDoc, ModelRegistryDoc, ProductDoc, SalesTransactionDoc, SupplierDoc
from scripts.run_model_comparison import FMCG_CAT, FMCG_NUM, GRO_CAT, GRO_NUM, run_dataset

log = logging.getLogger("pipelines.training")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data/processed"


def get_repositories(use_atlas: bool) -> Repositories:
    if not use_atlas:
        log.info("using the in-memory MongoDB fake (pass --atlas + set MONGODB_URI for a real cluster)")
        return Repositories.in_memory()
    from mongo.client import get_db
    from mongo.indexes import ensure_indexes
    db = get_db()
    ensure_indexes(db)
    log.info("connected to MongoDB Atlas database '%s'", db.name)
    return Repositories(db)


def persist_model_registry(repos: Repositories, report: dict) -> None:
    now = datetime.now(timezone.utc)
    for res in report["results"]:
        repos.model_registry.save_run(ModelRegistryDoc(
            dataset=report["dataset"], model_name=res["model"], metrics=res["test"],
            feature_importance_method=report["interpretability"]["method"], trained_at=now,
            train_rows=res["train_rows_used"], is_best=(res["model"] == report["best_model_by_test_f1"])))
    log.info("%s: persisted %d model_registry entries (best=%s)", report["dataset"], len(report["results"]),
             report["best_model_by_test_f1"])


def seed_grocery_operational_data(repos: Repositories) -> dict:
    """Real (non-augmented) product/supplier/inventory/sales master data only."""
    real = pd.concat([pd.read_csv(PROC / "grocery" / f"{p}.csv") for p in ("train", "val", "test")],
                     ignore_index=True)
    suppliers = real[["Supplier_ID", "Supplier_Name", "Supplier_OnTime_Pct"]].drop_duplicates("Supplier_ID")
    for row in suppliers.itertuples():
        repos.suppliers.upsert(SupplierDoc(supplier_id=row.Supplier_ID, name=row.Supplier_Name,
                                           on_time_pct=round(float(row.Supplier_OnTime_Pct), 2)))
    for row in real.itertuples():
        repos.products.upsert(ProductDoc(sku=row.SKU_ID, name=row.SKU_Name, category=row.Category,
                                         status="active", safety_stock=float(row.Safety_Stock), moq=1, pack_size=1,
                                         unit_cost=round(float(row.Unit_Cost_USD), 4), supplier_id=row.Supplier_ID,
                                         lead_time_days=float(row.Lead_Time_Days), warehouse=row.Warehouse_ID,
                                         avg_daily_sales_hint=round(float(row.Avg_Daily_Sales), 2)))
        repos.inventory.upsert_snapshot(InventorySnapshotDoc(
            sku=row.SKU_ID, warehouse=row.Warehouse_ID, on_hand=int(row.Quantity_On_Hand),
            reserved=int(min(row.Quantity_Reserved, row.Quantity_On_Hand)),
            damaged=int(min(row.Damaged_Qty, row.Quantity_On_Hand)), updated_at=datetime.now(timezone.utc)))
    return {"products": len(real), "suppliers": len(suppliers)}


def seed_fmcg_master_data(repos: Repositories) -> dict:
    """One ProductDoc per SKU (aggregated across warehouses - lead time/safety stock
    are per-SKU in this dataset, inventory is still tracked per (sku, warehouse)),
    one SupplierDoc per Brand, and an opening InventorySnapshotDoc per (sku, warehouse)
    taken from each combination's most recent real observation."""
    real = pd.read_csv(PROC / "fmcg" / "train.csv", parse_dates=["Invoice_Date"])
    brands = real["Brand"].unique()
    for b in brands:
        g = real[real.Brand == b]
        repos.suppliers.upsert(SupplierDoc(supplier_id=f"SUP-{b}", name=b, on_time_pct=None))
    for sku, g in real.groupby("sku_code"):
        repos.products.upsert(ProductDoc(
            sku=sku, name=sku.replace("|", " - "), category=g["Category"].iloc[0], status="active",
            safety_stock=float(g["Reorder_Level"].median()), moq=1, pack_size=1,
            unit_cost=round(float(g["Cost_Price"].median()), 4), supplier_id=f"SUP-{g['Brand'].iloc[0]}",
            lead_time_days=float(g["Lead_Time_Days"].median()), warehouse=g["warehouse_code"].mode().iat[0],
            avg_daily_sales_hint=None))
    n_snap = 0
    for (sku, wh), g in real.groupby(["sku_code", "warehouse_code"]):
        latest = g.sort_values("Invoice_Date").iloc[-1]
        repos.inventory.upsert_snapshot(InventorySnapshotDoc(sku=sku, warehouse=wh,
                                                              on_hand=int(latest["Stock_On_Hand"]), reserved=0,
                                                              damaged=0, updated_at=datetime.now(timezone.utc)))
        n_snap += 1
    return {"products": real["sku_code"].nunique(), "suppliers": len(brands), "inventory_snapshots": n_snap}


def seed_fmcg_sales(repos: Repositories, max_rows: int | None = 200_000) -> dict:
    """Real (non-augmented) sales transactions -> the operational sales_transactions collection.

    `max_rows` bounds how much history the in-memory fake / a free-tier Atlas
    cluster is asked to hold for a coursework demo; omit it for the full set."""
    real = pd.read_csv(PROC / "fmcg" / "train.csv", parse_dates=["Invoice_Date"])
    if max_rows and len(real) > max_rows:
        real = real.sample(n=max_rows, random_state=42).sort_values("Invoice_Date")
    docs = [SalesTransactionDoc(external_txn_id=r.external_txn_id, sku=r.sku_code, warehouse=r.warehouse_code,
                                quantity=int(r.Units), unit_price=float(r.Selling_Price),
                                sold_at=r.Invoice_Date.to_pydatetime().replace(tzinfo=timezone.utc))
           for r in real.itertuples()]
    return repos.sales.insert_many_idempotent(docs, source="fmcg_real")


def run(dataset: str, use_atlas: bool, seed_operational: bool, max_fmcg_sales_rows: int | None) -> dict:
    repos = get_repositories(use_atlas)
    run_report = {"started_at": datetime.now(timezone.utc).isoformat(), "datasets": []}

    if dataset in ("grocery", "both"):
        report = run_dataset("grocery", GRO_NUM, GRO_CAT, max_train_rows=20_506, limitation=None)
        persist_model_registry(repos, report)
        run_report["datasets"].append({"dataset": "grocery", "best_model": report["best_model_by_test_f1"],
                                       "best_test_f1": max(r["test"]["f1"] for r in report["results"])})
        if seed_operational:
            counts = seed_grocery_operational_data(repos)
            log.info("seeded grocery operational data: %s", counts)
            run_report["grocery_seed_counts"] = counts

    if dataset in ("fmcg", "both"):
        report = run_dataset("fmcg", FMCG_NUM, FMCG_CAT, max_train_rows=150_000,
                             limitation="stock_risk label carries no learnable signal in the source data "
                             "(see docs/STATUS.md) - reported for completeness, not as a working classifier.")
        persist_model_registry(repos, report)
        run_report["datasets"].append({"dataset": "fmcg", "best_model": report["best_model_by_test_f1"],
                                       "best_test_f1": max(r["test"]["f1"] for r in report["results"]),
                                       "note": "label unlearnable - see docs/STATUS.md"})
        if seed_operational:
            master_counts = seed_fmcg_master_data(repos)
            sales_counts = seed_fmcg_sales(repos, max_rows=max_fmcg_sales_rows)
            log.info("seeded fmcg master data: %s, sales transactions: %s", master_counts, sales_counts)
            run_report["fmcg_master_seed_counts"] = master_counts
            run_report["fmcg_sales_seed_counts"] = sales_counts

    run_report["finished_at"] = datetime.now(timezone.utc).isoformat()
    out = ROOT / "reports/pipelines"
    out.mkdir(parents=True, exist_ok=True)
    (out / "training_pipeline_run.json").write_text(json.dumps(run_report, indent=2, default=str))
    return run_report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["grocery", "fmcg", "both"], default="both")
    ap.add_argument("--atlas", action="store_true", help="use a real MongoDB Atlas cluster (needs MONGODB_URI)")
    ap.add_argument("--no-seed", action="store_true", help="skip seeding operational collections")
    ap.add_argument("--max-fmcg-sales-rows", type=int, default=200_000)
    args = ap.parse_args()
    report = run(args.dataset, args.atlas, not args.no_seed, args.max_fmcg_sales_rows)
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
