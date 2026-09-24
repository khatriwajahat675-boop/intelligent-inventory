"""Repository tests run against the in-memory fake (no MongoDB Atlas cluster
needed) - see mongo/fake_collection.py for why this is a faithful stand-in."""
import unittest
from datetime import datetime, timedelta, timezone

from mongo.repository import Repositories
from mongo.schemas import (AlertDoc, AuditEventDoc, InventorySnapshotDoc, ProductDoc, RecommendationDoc,
                           SalesTransactionDoc, SupplierDoc)

NOW = datetime(2026, 9, 20, tzinfo=timezone.utc)


class Products(unittest.TestCase):
    def test_upsert_is_idempotent(self):
        r = Repositories.in_memory()
        r.products.upsert(ProductDoc(sku="A", name="Widget", safety_stock=5))
        r.products.upsert(ProductDoc(sku="A", name="Widget v2", safety_stock=9))
        self.assertEqual(r.products.get("A")["name"], "Widget v2")
        self.assertEqual(len(r.products.list_active()), 1)

    def test_invalid_doc_rejected_before_it_reaches_the_database(self):
        with self.assertRaises(Exception):
            ProductDoc(sku="A", name="x", safety_stock=-1)      # pydantic ValidationError
        with self.assertRaises(Exception):
            ProductDoc(sku="A", name="x", extra_field="hack")   # extra="forbid"


class SalesIdempotency(unittest.TestCase):
    def test_duplicate_txn_not_double_counted(self):
        r = Repositories.in_memory()
        doc = SalesTransactionDoc(external_txn_id="T1", sku="A", warehouse="MAIN", quantity=3, sold_at=NOW)
        first = r.sales.insert_many_idempotent([doc], source="api")
        second = r.sales.insert_many_idempotent([doc], source="api")
        self.assertEqual(first, {"accepted": 1, "duplicates": 0})
        self.assertEqual(second, {"accepted": 0, "duplicates": 1})

    def test_recent_window(self):
        r = Repositories.in_memory()
        old = SalesTransactionDoc(external_txn_id="O", sku="A", warehouse="MAIN", quantity=1, sold_at=NOW - timedelta(days=40))
        new = SalesTransactionDoc(external_txn_id="N", sku="A", warehouse="MAIN", quantity=2, sold_at=NOW - timedelta(days=1))
        r.sales.insert_many_idempotent([old, new], source="api")
        recent = r.sales.recent("A", "MAIN", days=30, now=NOW)
        self.assertEqual([d["external_txn_id"] for d in recent], ["N"])


class Recommendations(unittest.TestCase):
    def test_idempotency_key_prevents_duplicate_recommendation(self):
        r = Repositories.in_memory()
        doc = RecommendationDoc(idempotency_key="reorder:A:MAIN:2026-09-20", sku="A", warehouse="MAIN",
                                recommended_qty=50, automation_state="REQUIRES_APPROVAL", reason="low stock",
                                explanation={}, created_at=NOW)
        created1, row1 = r.recommendations.create_if_absent(doc)
        created2, row2 = r.recommendations.create_if_absent(doc)
        self.assertTrue(created1)
        self.assertFalse(created2)
        self.assertEqual(row1["idempotency_key"], row2["idempotency_key"])
        self.assertEqual(r.recommendations.col.count_documents({}), 1)


class Alerts(unittest.TestCase):
    def test_dedupe_key_prevents_alert_storm(self):
        r = Repositories.in_memory()
        doc = AlertDoc(type="restock_needed", severity="high", entity_type="product", entity_id="A",
                       dedupe_key="restock:A:MAIN", message="low stock", created_at=NOW)
        self.assertTrue(r.alerts.raise_if_absent(doc))
        self.assertFalse(r.alerts.raise_if_absent(doc))     # second call is a no-op, not a duplicate alert

    def test_severity_ordering(self):
        r = Repositories.in_memory()
        for sev in ("info", "critical", "medium", "high"):
            r.alerts.raise_if_absent(AlertDoc(type="t", severity=sev, entity_type="x", entity_id=sev,
                                              dedupe_key=f"k:{sev}", message="m", created_at=NOW))
        ordered = [a["severity"] for a in r.alerts.open_by_severity()]
        self.assertEqual(ordered, ["critical", "high", "medium", "info"])


class AuditTrail(unittest.TestCase):
    def test_record_and_query(self):
        r = Repositories.in_memory()
        r.audit.record("svc:automation", "alert.raise", "product", "A", after={"qty": 5})
        events = r.audit.for_entity("A")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["action"], "alert.raise")


class InventorySnapshots(unittest.TestCase):
    def test_apply_delta_never_goes_negative(self):
        r = Repositories.in_memory()
        r.inventory.upsert_snapshot(InventorySnapshotDoc(sku="A", warehouse="MAIN", on_hand=5, updated_at=NOW))
        r.inventory.apply_delta("A", "MAIN", -100)
        self.assertEqual(r.inventory.get("A", "MAIN")["on_hand"], 0)


if __name__ == "__main__":
    unittest.main()
