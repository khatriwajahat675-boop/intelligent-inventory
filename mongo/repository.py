"""Repository pattern over MongoDB collections.

Every method here takes typed arguments (a pydantic doc, or scalar values)
and builds its own filter internally - callers never pass a raw filter dict
sourced from outside this module (see mongo/security.py for why). Each
repository works identically against a real pymongo `Collection` or the
offline `InMemoryCollection` fake (mongo/fake_collection.py) - that is what
let the automation/training pipelines be executed end to end in this sandbox
without a live Atlas cluster.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Protocol

try:
    from pymongo.errors import DuplicateKeyError
except ImportError:
    from mongo.fake_collection import DuplicateKeyError  # same fallback class fake_collection uses

from mongo.schemas import (AlertDoc, AuditEventDoc, InventorySnapshotDoc, ModelRegistryDoc, ProductDoc,
                           RecommendationDoc, SalesTransactionDoc, SupplierDoc)


class CollectionLike(Protocol):
    def insert_one(self, doc: dict) -> Any: ...
    def find_one(self, filt: dict) -> dict | None: ...
    def find(self, filt: dict, sort=None, limit=None) -> list[dict]: ...
    def count_documents(self, filt: dict) -> int: ...
    def update_one(self, filt: dict, update: dict, upsert: bool = False) -> Any: ...
    def create_index(self, keys, **kw) -> Any: ...


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ProductRepository:
    def __init__(self, col: CollectionLike):
        self.col = col

    def upsert(self, doc: ProductDoc) -> None:
        self.col.update_one({"sku": doc.sku}, {"$set": doc.model_dump()}, upsert=True)

    def get(self, sku: str) -> dict | None:
        return self.col.find_one({"sku": sku})

    def list_active(self) -> list[dict]:
        return self.col.find({"status": "active"})


class SupplierRepository:
    def __init__(self, col: CollectionLike):
        self.col = col

    def upsert(self, doc: SupplierDoc) -> None:
        self.col.update_one({"supplier_id": doc.supplier_id}, {"$set": doc.model_dump()}, upsert=True)

    def get(self, supplier_id: str) -> dict | None:
        return self.col.find_one({"supplier_id": supplier_id})


class InventoryRepository:
    def __init__(self, col: CollectionLike):
        self.col = col

    def upsert_snapshot(self, doc: InventorySnapshotDoc) -> None:
        self.col.update_one({"sku": doc.sku, "warehouse": doc.warehouse}, {"$set": doc.model_dump()}, upsert=True)

    def get(self, sku: str, warehouse: str) -> dict | None:
        return self.col.find_one({"sku": sku, "warehouse": warehouse})

    def list_all(self) -> list[dict]:
        return self.col.find({})

    def apply_delta(self, sku: str, warehouse: str, on_hand_delta: int) -> None:
        """Used by the automation pipeline to simulate consumption when replaying history."""
        cur = self.get(sku, warehouse)
        new_on_hand = max(0, (cur["on_hand"] if cur else 0) + on_hand_delta)
        self.upsert_snapshot(InventorySnapshotDoc(sku=sku, warehouse=warehouse, on_hand=new_on_hand,
                                                   reserved=(cur or {}).get("reserved", 0),
                                                   damaged=(cur or {}).get("damaged", 0), updated_at=utcnow()))


class SalesRepository:
    def __init__(self, col: CollectionLike):
        self.col = col

    def insert_many_idempotent(self, docs: list[SalesTransactionDoc], source: str) -> dict:
        """EC-01/EC-02 semantics: duplicate (source, external_txn_id) is skipped, never double-counted."""
        accepted = duplicates = 0
        for d in docs:
            try:
                self.col.insert_one({**d.model_dump(), "source": source})
                accepted += 1
            except DuplicateKeyError:
                duplicates += 1
        return {"accepted": accepted, "duplicates": duplicates}

    def recent(self, sku: str, warehouse: str, days: int, now: datetime | None = None) -> list[dict]:
        now = now or utcnow()
        since = now - timedelta(days=days)
        return self.col.find({"sku": sku, "warehouse": warehouse, "sold_at": {"$gte": since}},
                             sort=[("sold_at", 1)])

    def all_for_sku(self, sku: str, warehouse: str) -> list[dict]:
        return self.col.find({"sku": sku, "warehouse": warehouse}, sort=[("sold_at", 1)])


class RecommendationRepository:
    def __init__(self, col: CollectionLike):
        self.col = col

    def create_if_absent(self, doc: RecommendationDoc) -> tuple[bool, dict]:
        """EC-35/EC-51: same idempotency key -> return the existing recommendation, create nothing new."""
        try:
            self.col.insert_one(doc.model_dump())
            return True, doc.model_dump()
        except DuplicateKeyError:
            return False, self.col.find_one({"idempotency_key": doc.idempotency_key})

    def open_for(self, sku: str, warehouse: str) -> list[dict]:
        return self.col.find({"sku": sku, "warehouse": warehouse, "status": {"$in": ["PENDING", "ON_HOLD"]}})

    def list_open(self, limit: int = 200) -> list[dict]:
        return self.col.find({"status": "PENDING"}, sort=[("created_at", -1)], limit=limit)


class AlertRepository:
    def __init__(self, col: CollectionLike):
        self.col = col

    def raise_if_absent(self, doc: AlertDoc) -> bool:
        """EC-38: at most one OPEN alert per dedupe_key (enforced by the partial unique index)."""
        try:
            self.col.insert_one(doc.model_dump())
            return True
        except DuplicateKeyError:
            return False

    def open_by_severity(self, limit: int = 200) -> list[dict]:
        order = {"critical": 0, "high": 1, "medium": 2, "info": 3}
        rows = self.col.find({"status": "open"}, limit=limit)
        return sorted(rows, key=lambda r: order.get(r["severity"], 9))


class AuditRepository:
    def __init__(self, col: CollectionLike):
        self.col = col

    def record(self, actor: str, action: str, entity_type: str, entity_id: str, *, before: dict | None = None,
              after: dict | None = None, reason: str | None = None) -> None:
        doc = AuditEventDoc(actor=actor, action=action, entity_type=entity_type, entity_id=str(entity_id),
                            before=before, after=after, reason=reason, occurred_at=utcnow())
        self.col.insert_one(doc.model_dump())

    def for_entity(self, entity_id: str) -> list[dict]:
        return self.col.find({"entity_id": str(entity_id)}, sort=[("occurred_at", -1)])


class ModelRegistryRepository:
    def __init__(self, col: CollectionLike):
        self.col = col

    def save_run(self, doc: ModelRegistryDoc) -> None:
        self.col.insert_one(doc.model_dump())

    def best_for(self, dataset: str) -> dict | None:
        rows = list(self.col.find({"dataset": dataset, "is_best": True}, sort=[("trained_at", -1)], limit=1))
        return rows[0] if rows else None


class Repositories:
    """Bundle used by both pipelines - one object to pass around, one place that
    knows the collection names."""

    def __init__(self, db):
        self.products = ProductRepository(db.products)
        self.suppliers = SupplierRepository(db.suppliers)
        self.inventory = InventoryRepository(db.inventory_snapshots)
        self.sales = SalesRepository(db.sales_transactions)
        self.recommendations = RecommendationRepository(db.recommendations)
        self.alerts = AlertRepository(db.alerts)
        self.audit = AuditRepository(db.audit_events)
        self.model_registry = ModelRegistryRepository(db.model_registry)

    @classmethod
    def in_memory(cls) -> "Repositories":
        """Offline test/demo double - no MongoDB Atlas cluster required."""
        from mongo.fake_collection import InMemoryCollection
        from mongo.indexes import ensure_indexes

        class _FakeDb(dict):
            def __getattr__(self, name):
                return self.setdefault(name, InMemoryCollection(name))

        db = _FakeDb()
        ensure_indexes(db)
        return cls(db)
