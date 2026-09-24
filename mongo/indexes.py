"""Index + uniqueness definitions - the same idempotency/uniqueness rules the
Postgres schema enforces (backend/app/models/entities.py) apply here too,
regardless of which database stores the data (TRD 11.2)."""
from __future__ import annotations


def ensure_indexes(db) -> None:
    db.products.create_index("sku", unique=True)
    db.suppliers.create_index("supplier_id", unique=True)
    db.sales_transactions.create_index([("source", 1), ("external_txn_id", 1)], unique=True)
    db.sales_transactions.create_index([("sku", 1), ("warehouse", 1), ("sold_at", 1)])
    db.inventory_snapshots.create_index([("sku", 1), ("warehouse", 1)], unique=True)
    db.recommendations.create_index("idempotency_key", unique=True)
    db.alerts.create_index([("dedupe_key", 1), ("status", 1)], unique=True,
                           partialFilterExpression={"status": "open"})   # only one OPEN alert per event (EC-38)
    db.audit_events.create_index([("action", 1), ("occurred_at", -1)])
    db.audit_events.create_index("entity_id")
    db.model_registry.create_index([("dataset", 1), ("trained_at", -1)])
