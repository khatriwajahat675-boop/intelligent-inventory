"""Sales ingestion - idempotent, row-level validation (FR-006, EC-01..EC-06, EC-46)."""
from __future__ import annotations

import csv
import io
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.domain import inventory as inv
from backend.app.models.entities import (InventoryBalance, InventoryMovement, Product, SalesTransaction,
                                         Warehouse)
from backend.app.services import audit

REQUIRED = ("external_txn_id", "sku", "warehouse", "quantity", "sold_at")


class ImportTooLarge(ValueError):
    pass


def parse_csv(raw: bytes, max_bytes: int, max_rows: int) -> list[dict]:
    if len(raw) > max_bytes:
        raise ImportTooLarge("file too large")                          # EC-46
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
    missing = [c for c in REQUIRED if c not in (reader.fieldnames or [])]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    rows = []
    for i, r in enumerate(reader):
        if i >= max_rows:
            raise ImportTooLarge("too many rows")
        rows.append(r)
    return rows


def _parse_ts(value) -> datetime:
    ts = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


def ingest(db: Session, rows: list[dict], *, source: str, actor: str, business_tz: str,
           update_stock: bool, now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    tz = ZoneInfo(business_tz)
    skus = {p.sku: p for p in db.scalars(select(Product))}
    whs = {w.code: w for w in db.scalars(select(Warehouse))}
    accepted, duplicates, errors, touched = 0, 0, [], set()
    seen = {t for (t,) in db.execute(select(SalesTransaction.external_txn_id).where(SalesTransaction.source == source))}
    for n, r in enumerate(rows, start=1):
        try:
            txn = str(r.get("external_txn_id", "")).strip()
            if not txn:
                raise ValueError("external_txn_id required")
            if txn in seen:
                duplicates += 1                                          # EC-02: never counted twice
                continue
            p, w = skus.get(r.get("sku")), whs.get(r.get("warehouse"))
            if p is None:
                raise ValueError("unknown sku")                         # EC-03
            if w is None:
                raise ValueError("unknown warehouse")
            if p.status != "active":
                raise ValueError("sku archived")
            qty = int(r["quantity"])
            if qty <= 0:
                raise ValueError("quantity must be > 0")                # EC-03
            ts = _parse_ts(r["sold_at"]).astimezone(timezone.utc)
            if ts > now + timedelta(minutes=5):
                raise ValueError("future-dated sale")                   # EC-05
            if update_stock:
                bal = db.get(InventoryBalance, (p.id, w.id))
                if bal is None:
                    raise ValueError("no inventory balance for sku/warehouse")
                new = inv.apply_sale(inv.Balance(bal.on_hand, bal.reserved, bal.damaged), qty)  # EC-10
                bal.on_hand = new.on_hand
                db.add(InventoryMovement(product_id=p.id, warehouse_id=w.id, movement_type="sale",
                                         quantity=-qty, reference_type="sale", reference_id=txn, occurred_at=ts))
            db.add(SalesTransaction(external_txn_id=txn, source=source, product_id=p.id, warehouse_id=w.id,
                                    quantity=qty, unit_price=float(r["unit_price"]) if r.get("unit_price") else None,
                                    sold_at=ts, business_date=ts.astimezone(tz).date().isoformat()))  # EC-06/68
            seen.add(txn)
            touched.add((p.id, w.id))
            accepted += 1
        except (ValueError, KeyError, TypeError, inv.InventoryError) as exc:
            errors.append({"row": n, "error": str(exc)})                 # row-level errors (EC-03)
    audit.record(db, actor, "sales.ingest", "sales_batch", source,
                 after={"accepted": accepted, "duplicates": duplicates, "rejected": len(errors)})
    return {"accepted": accepted, "duplicates": duplicates, "rejected": len(errors), "errors": errors[:100],
            "stale_forecast_scopes": sorted(touched)}                    # EC-04/27: callers mark forecasts stale
