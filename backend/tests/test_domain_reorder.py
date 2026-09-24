"""Unit + parameterised tests for the decision engine (TRD 18.1)."""
import unittest
from datetime import datetime, timedelta

from backend.app.domain import reorder as r

NOW = datetime(2026, 9, 11, 12, 0)
P = r.Product("SKU-001", "active", moq=10, pack_size=12, safety_stock=20, unit_cost=2.0)
S = r.Supplier("S1", "active", 10)


def fc(daily=None, **kw):
    d = daily if daily is not None else (8.2,) * 10
    return r.Forecast("fr_1", "prophet_v4", tuple(d), NOW - timedelta(hours=1), **kw)


class TrdExample(unittest.TestCase):
    def test_section_12_3_example(self):
        rec = r.evaluate(P, S, r.Stock(25, eligible_inbound=15), fc(), r.Policy(), NOW)
        e = rec.explanation
        self.assertEqual(e["lead_time_demand"], 82.0)
        self.assertEqual(e["reorder_point"], 102.0)
        self.assertEqual(e["raw_shortage"], 62.0)
        self.assertEqual(rec.recommended_qty, 72)
        self.assertIn("rounded_to_pack_size_12", rec.constraint_adjustments)
        self.assertEqual(rec.state, r.AutomationState.REQUIRES_APPROVAL)


class Guardrails(unittest.TestCase):
    def ev(self, product=P, supplier=S, stock=r.Stock(0), forecast="d", policy=r.Policy(), **kw):
        return r.evaluate(product, supplier, stock, fc() if forecast == "d" else forecast, policy, NOW, **kw)

    def test_archived_sku_blocked(self):
        p = r.Product("X", "archived", 1, 1, 0, 1)
        self.assertEqual(self.ev(product=p).reason, "sku_archived")

    def test_missing_and_inactive_supplier(self):
        self.assertEqual(self.ev(supplier=None).reason, "missing_supplier")
        self.assertEqual(self.ev(supplier=r.Supplier("S", "inactive", 5)).reason, "supplier_inactive")

    def test_invalid_lead_time(self):
        for lt in (0, -3, float("nan")):
            self.assertEqual(self.ev(supplier=r.Supplier("S", "active", lt)).reason, "invalid_lead_time")

    def test_no_forecast_requires_manual_review(self):
        rec = self.ev(forecast=None)
        self.assertEqual(rec.state, r.AutomationState.REQUIRES_APPROVAL)
        self.assertEqual(rec.recommended_qty, 0)

    def test_stock_sufficient_and_inbound_cover_suppress(self):
        self.assertEqual(self.ev(stock=r.Stock(500)).state, r.AutomationState.SUPPRESSED)
        rec = self.ev(stock=r.Stock(10, eligible_inbound=200))
        self.assertEqual((rec.state, rec.reason), (r.AutomationState.SUPPRESSED, "covered_by_inbound"))  # EC-30

    def test_duplicate_recommendation_suppressed(self):
        self.assertEqual(self.ev(has_open_recommendation=True).reason, "open_recommendation_exists")

    def test_negative_forecast_clamped(self):
        rec = self.ev(forecast=fc((-50,) * 10), stock=r.Stock(0))
        self.assertEqual(rec.explanation["lead_time_demand"], 0.0)  # EC-21
        self.assertGreaterEqual(rec.recommended_qty, 0)

    def test_damaged_and_reserved_not_available(self):
        self.assertEqual(r.Stock(100, reserved=60, damaged=50).available, 0)

    def test_moq_larger_than_shortage(self):  # EC-31
        p = r.Product("X", "active", moq=100, pack_size=10, safety_stock=0, unit_cost=1)
        rec = self.ev(product=p, stock=r.Stock(75))
        self.assertEqual(rec.recommended_qty, 100)
        self.assertIn("raised_to_moq_100", rec.constraint_adjustments)

    def test_cap_routes_to_approval(self):
        pol = r.Policy(auto_approve_enabled=True, max_order_qty=48)
        rec = self.ev(stock=r.Stock(0), policy=pol)
        self.assertEqual(rec.recommended_qty, 48)
        self.assertEqual(rec.state, r.AutomationState.REQUIRES_APPROVAL)

    def test_auto_approve_only_when_all_conditions_hold(self):
        pol = r.Policy(auto_approve_enabled=True)
        good = self.ev(stock=r.Stock(60), policy=pol)
        self.assertEqual(good.state, r.AutomationState.AUTO_APPROVE)
        cases = {
            "forecast_stale": r.Forecast("f", "m", (8.2,) * 10, NOW - timedelta(days=5)),
            "forecast_low_confidence": fc(quality_flag="low_confidence"),
            "forecast_uncertainty_high": fc(uncertainty=0.9),
        }
        for reason, f in cases.items():
            rec = self.ev(stock=r.Stock(60), forecast=f, policy=pol)
            self.assertEqual(rec.state, r.AutomationState.REQUIRES_APPROVAL, reason)
            self.assertIn(reason, rec.explanation["approval_reasons"])

    def test_value_limit_and_missing_cost(self):  # EC-33 / EC-64
        pol = r.Policy(auto_approve_enabled=True, auto_approve_max_value=10)
        self.assertIn("order_value_over_limit", self.ev(stock=r.Stock(0), policy=pol).explanation["approval_reasons"])
        p = r.Product("X", "active", 10, 12, 20, None)
        self.assertIn("unit_cost_missing",
                      self.ev(product=p, stock=r.Stock(0), policy=r.Policy(auto_approve_enabled=True)).explanation["approval_reasons"])

    def test_short_forecast_extended_and_flagged(self):
        rec = self.ev(forecast=fc((10,) * 3), stock=r.Stock(0))
        self.assertIn("forecast_horizon_extended_with_mean", rec.constraint_adjustments)
        self.assertEqual(rec.explanation["lead_time_demand"], 100.0)

    def test_decision_is_reproducible(self):
        a = self.ev(stock=r.Stock(25))
        b = self.ev(stock=r.Stock(25))
        self.assertEqual(a, b)

    def test_idempotency_key(self):
        self.assertEqual(r.idempotency_key("A", "MAIN", "2026-09-11"), "reorder:A:MAIN:2026-09-11")

    def test_property_qty_never_negative_and_multiple_of_pack(self):
        for onhand in range(0, 400, 7):
            for pack in (1, 6, 12, 25):
                p = r.Product("X", "active", 5, pack, 10, 1.0)
                rec = self.ev(product=p, stock=r.Stock(onhand))
                self.assertGreaterEqual(rec.recommended_qty, 0)
                if rec.recommended_qty:
                    self.assertEqual(rec.recommended_qty % pack, 0)


class ApprovalStateMachine(unittest.TestCase):
    def test_approve_once(self):
        st, v = r.transition("PENDING", 1, 1, "approve")
        self.assertEqual((st, v), ("APPROVED", 2))
        with self.assertRaises(r.ConflictError):       # double-click / replay
            r.transition(st, v, v, "approve")

    def test_version_conflict(self):                   # EC-36 / EC-61
        with self.assertRaises(r.ConflictError):
            r.transition("PENDING", 3, 2, "approve")

    def test_reject_then_no_approve(self):
        st, v = r.transition("PENDING", 1, 1, "reject")
        with self.assertRaises(r.ConflictError):
            r.transition(st, v, v, "approve")


if __name__ == "__main__":
    unittest.main()
