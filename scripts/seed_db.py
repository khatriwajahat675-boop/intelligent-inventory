"""Load a processed dataset into PostgreSQL for the demo (TRD 18.6 steps 1-3).

    PYTHONPATH=. python scripts/seed_db.py fmcg                 # sales from TRAIN+VAL only; TEST kept as 'future'
    PYTHONPATH=. python scripts/seed_db.py grocery --synthesize-history   # labelled synthetic demo demand

Assumptions (documented, not hidden): MOQ/pack size are not in either file -> MOQ=1, pack=1 unless set later
in the UI. FMCG 'City' is treated as warehouse, 'Category|Brand' as SKU, Brand as supplier.
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from backend.app import db as dbmod
from backend.app.models.entities import (Base, InventoryBalance, PolicyConfig, Product, ProductSupplier,
                                         SalesTransaction, Supplier, User, Warehouse)
from backend.app.security.passwords import hash_password

PROC = Path(__file__).resolve().parents[1] / "data/processed"


def seed_users(db):
    pw = os.environ.get("SEED_PASSWORD", "ChangeMe-Now-123")
    for email, role in [("admin@example.com", "admin"), ("manager@example.com", "inventory_manager"),
                        ("ops@example.com", "operations_user"), ("analyst@example.com", "analyst"),
                        ("automation@example.com", "automation_service")]:
        if not db.query(User).filter_by(email=email).first():
            db.add(User(email=email, password_hash=hash_password(pw), role=role))
    if not db.query(PolicyConfig).first():
        db.add(PolicyConfig(version="policy_v1", active=1, values_json={
            "auto_approve_enabled": False, "auto_approve_max_qty": 500, "auto_approve_max_value": 5000.0,
            "max_order_qty": 5000, "max_forecast_uncertainty": 0.35, "forecast_freshness_hours": 36}))


def seed_fmcg(db):
    tr = pd.read_csv(PROC / "fmcg/train.csv", parse_dates=["Invoice_Date"])
    va = pd.read_csv(PROC / "fmcg/val.csv", parse_dates=["Invoice_Date"])
    hist = pd.concat([tr, va])
    whs = {c: Warehouse(code=c, name=c, timezone="Asia/Kolkata") for c in sorted(hist.warehouse_code.unique())}
    sups = {b: Supplier(code=b[:32], name=b, default_lead_time_days=float(hist[hist.Brand == b].Lead_Time_Days.median()))
            for b in sorted(hist.Brand.unique())}
    db.add_all([*whs.values(), *sups.values()])
    db.flush()
    prods = {}
    for sku, g in hist.groupby("sku_code"):
        p = Product(sku=sku, name=sku.replace("|", " - "), category=g.Category.iloc[0],
                    safety_stock=float(g.Reorder_Level.median()))
        db.add(p)
        prods[sku] = (p, g)
    db.flush()
    for sku, (p, g) in prods.items():
        s = sups[g.Brand.iloc[0]]
        db.add(ProductSupplier(product_id=p.id, supplier_id=s.id, lead_time_days=float(g.Lead_Time_Days.median()),
                               unit_cost=float(g.Cost_Price.median()), preferred=1))
        for city, gc in g.groupby("warehouse_code"):
            db.add(InventoryBalance(product_id=p.id, warehouse_id=whs[city].id,
                                    on_hand=int(gc.sort_values("Invoice_Date").Stock_On_Hand.tail(20).median())))
    rows = [SalesTransaction(external_txn_id=r.external_txn_id, source="fmcg_csv", product_id=prods[r.sku_code][0].id,
                             warehouse_id=whs[r.warehouse_code].id, quantity=int(r.Units),
                             unit_price=float(r.Selling_Price), sold_at=r.Invoice_Date.tz_localize("Asia/Kolkata"),
                             business_date=r.Invoice_Date.date().isoformat())
            for r in hist.itertuples()]
    db.bulk_save_objects(rows)


def seed_grocery(db, synthesize: bool):
    g = pd.concat([pd.read_csv(PROC / f"grocery/{n}.csv", parse_dates=["Received_Date"]) for n in ("train", "val", "test")])
    whs = {c: Warehouse(code=c, name=n) for c, n in g[["Warehouse_ID", "Warehouse_Location"]].drop_duplicates().values}
    sups = {c: Supplier(code=c, name=n, default_lead_time_days=7, on_time_pct=float(g[g.Supplier_ID == c].Supplier_OnTime_Pct.mean()))
            for c, n in g[["Supplier_ID", "Supplier_Name"]].drop_duplicates().values}
    db.add_all([*whs.values(), *sups.values()])
    db.flush()
    rng = np.random.default_rng(42)
    for r in g.itertuples():
        p = Product(sku=r.SKU_ID, name=r.SKU_Name, category=r.Category, abc_class=r.ABC_Class, safety_stock=float(r.Safety_Stock))
        db.add(p)
        db.flush()
        db.add(ProductSupplier(product_id=p.id, supplier_id=sups[r.Supplier_ID].id, lead_time_days=float(r.Lead_Time_Days),
                               unit_cost=float(r.Unit_Cost_USD), preferred=1))
        db.add(InventoryBalance(product_id=p.id, warehouse_id=whs[r.Warehouse_ID].id, on_hand=int(r.Quantity_On_Hand),
                                reserved=min(int(r.Quantity_Reserved), int(r.Quantity_On_Hand)), damaged=int(min(r.Damaged_Qty, r.Quantity_On_Hand))))
        if synthesize:      # clearly labelled: the CSV has NO sales history
            days = pd.date_range(end=pd.Timestamp.today().normalize() - pd.Timedelta(days=1), periods=90)
            for d, q in zip(days, rng.poisson(r.Avg_Daily_Sales, len(days))):
                if q > 0:
                    db.add(SalesTransaction(external_txn_id=f"{r.SKU_ID}-{d.date()}", source="synthetic_demo", product_id=p.id,
                                            warehouse_id=whs[r.Warehouse_ID].id, quantity=int(q), sold_at=d.tz_localize("UTC"),
                                            business_date=d.date().isoformat()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["fmcg", "grocery"])
    ap.add_argument("--synthesize-history", action="store_true")
    a = ap.parse_args()
    engine = dbmod.init_engine()
    Base.metadata.create_all(engine)
    with dbmod.session_scope() as db:
        seed_users(db)
        seed_fmcg(db) if a.dataset == "fmcg" else seed_grocery(db, a.synthesize_history)
    print("seeded", a.dataset)
