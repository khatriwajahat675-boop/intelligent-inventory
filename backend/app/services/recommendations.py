"""Reorder evaluation, approval workflow and PO creation (FR-011..FR-014, TRD 14)."""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.app.domain import inventory as invdom
from backend.app.domain import reorder as ro
from backend.app.models.entities import (Forecast, ForecastRun, InventoryBalance, Product, ProductSupplier,
                                         PurchaseOrder, PurchaseOrderLine, Recommendation, Supplier, Warehouse)
from backend.app.services import alerts, audit
from backend.app.services.policy import active_policy

SERVICE = "svc:reorder-engine"
OPEN_PO = ("sent", "partially_received")


def eligible_inbound(db: Session, product_id: int, warehouse_id: int) -> int:
    """EC-39/40/72: open remainder of SENT/partial POs for this exact warehouse."""
    rows = db.execute(
        select(PurchaseOrderLine.qty_ordered, PurchaseOrderLine.qty_received)
        .join(PurchaseOrder, PurchaseOrder.id == PurchaseOrderLine.po_id)
        .where(PurchaseOrderLine.product_id == product_id, PurchaseOrder.warehouse_id == warehouse_id,
               PurchaseOrder.status.in_(OPEN_PO))).all()
    return sum(max(0, o - r) for o, r in rows)


def _has_open(db: Session, product_id: int, warehouse_id: int) -> bool:
    pending = db.scalar(select(func.count()).select_from(Recommendation).where(
        Recommendation.product_id == product_id, Recommendation.warehouse_id == warehouse_id,
        Recommendation.status.in_(("PENDING", "ON_HOLD"))))
    unsent = db.scalar(select(func.count()).select_from(PurchaseOrder).join(
        PurchaseOrderLine, PurchaseOrderLine.po_id == PurchaseOrder.id).where(
        PurchaseOrderLine.product_id == product_id, PurchaseOrder.warehouse_id == warehouse_id,
        PurchaseOrder.status.in_(("draft", "approved"))))
    return bool(pending or unsent)


def evaluate_one(db: Session, product: Product, warehouse: Warehouse, *, now: datetime | None = None,
                 actor: str = SERVICE, request_id: str | None = None) -> tuple[ro.Recommendation, Recommendation | None]:
    now = now or datetime.now(timezone.utc)
    policy = active_policy(db)
    ps = db.execute(select(ProductSupplier, Supplier).join(Supplier, Supplier.id == ProductSupplier.supplier_id)
                    .where(ProductSupplier.product_id == product.id)
                    .order_by(ProductSupplier.preferred.desc(), ProductSupplier.lead_time_days)).first()
    supplier = ro.Supplier(str(ps[1].id), ps[1].status, ps[0].lead_time_days) if ps else None
    bal = db.get(InventoryBalance, (product.id, warehouse.id))
    stock = ro.Stock(bal.on_hand if bal else 0, bal.reserved if bal else 0, bal.damaged if bal else 0,
                     eligible_inbound(db, product.id, warehouse.id))
    run_ = db.scalar(select(ForecastRun).where(ForecastRun.status == "succeeded").order_by(ForecastRun.id.desc()))
    fc = None
    if run_:
        rows = db.scalars(select(Forecast).where(Forecast.run_id == run_.id, Forecast.product_id == product.id,
                                                 Forecast.warehouse_id == warehouse.id)
                          .order_by(Forecast.forecast_date)).all()
        if rows:
            unc = max((r.upper - r.lower) / (2 * r.yhat) if r.yhat > 0 and r.upper is not None else 0.0 for r in rows)
            gen = (run_.finished_at or run_.started_at)
            gen = gen if gen.tzinfo else gen.replace(tzinfo=timezone.utc)
            fc = ro.Forecast(str(run_.id), run_.model_version, tuple(r.yhat for r in rows), gen, rows[0].quality_flag, unc)
    dom_product = ro.Product(product.sku, product.status, ps[0].moq if ps else 1, ps[0].pack_size if ps else 1,
                             product.safety_stock, ps[0].unit_cost if ps else None)
    result = ro.evaluate(dom_product, supplier, stock, fc, policy, now, _has_open(db, product.id, warehouse.id))

    row = None
    if result.state == ro.AutomationState.BLOCKED or (result.state == ro.AutomationState.REQUIRES_APPROVAL
                                                      and result.recommended_qty == 0):
        alerts.raise_alert(db, "automation_blocked", "high", "product", product.id,
                           f"{product.sku}: {result.reason}", dedupe_key=f"blocked:{product.id}:{warehouse.id}:{result.reason}")
    elif result.recommended_qty > 0:
        key = ro.idempotency_key(product.sku, warehouse.code, now.date().isoformat())
        row = Recommendation(idempotency_key=key, product_id=product.id, warehouse_id=warehouse.id,
                             supplier_id=int(supplier.id) if supplier else None,
                             forecast_run_id=run_.id if run_ else None, policy_version=policy.version,
                             recommended_qty=result.recommended_qty, automation_state=result.state.value,
                             explanation_json={**result.explanation, "reason": result.reason,
                                               "constraint_adjustments": result.constraint_adjustments})
        try:
            with db.begin_nested():                                     # savepoint: duplicate key rolls back only this
                db.add(row)
                db.flush()
        except IntegrityError:                                          # EC-35/51: concurrent evaluator won
            return result, db.scalar(select(Recommendation).where(Recommendation.idempotency_key == key))
        audit.record(db, actor, "recommendation.create", "recommendation", row.id, after=result.explanation,
                     policy_version=policy.version, model_version=result.explanation.get("model_version"),
                     request_id=request_id)
        if result.explanation.get("approval_reasons") and "forecast_stale" in result.explanation["approval_reasons"]:
            alerts.raise_alert(db, "forecast_stale", "medium", "product", product.id, f"{product.sku}: stale forecast")
        sev = "critical" if result.explanation["available"] + result.explanation["eligible_inbound"] <= 0 else "high"
        alerts.raise_alert(db, "restock_needed", sev, "recommendation", row.id,
                           f"{product.sku}: order {row.recommended_qty}", dedupe_key=f"restock:{product.id}:{warehouse.id}")
        if result.state == ro.AutomationState.AUTO_APPROVE:
            decide(db, row, "approve", row.version, actor=SERVICE, reason="auto-approved within policy",
                   request_id=request_id)
    return result, row


def decide(db: Session, reco: Recommendation, action: str, expected_version: int, *, actor: str,
           reason: str | None = None, request_id: str | None = None) -> Recommendation:
    """Approve / reject / hold with version check (EC-36) - replays cannot approve twice."""
    if action == "reject" and not (reason or "").strip():
        raise ValueError("a reason is required to reject")
    before = {"status": reco.status, "version": reco.version, "qty": reco.recommended_qty}
    new_status, new_version = ro.transition(reco.status, reco.version, expected_version, action)
    reco.status, reco.version, reco.decision_reason = new_status, new_version, reason
    audit.record(db, actor, f"recommendation.{action}", "recommendation", reco.id, before=before,
                 after={"status": new_status, "version": new_version}, reason=reason,
                 policy_version=reco.policy_version, request_id=request_id)
    if new_status == "APPROVED":
        create_po_from_recommendation(db, reco, actor)
    return reco


def create_po_from_recommendation(db: Session, reco: Recommendation, actor: str) -> PurchaseOrder:
    existing = db.scalar(select(PurchaseOrder).where(PurchaseOrder.recommendation_id == reco.id))
    if existing:                                                        # unique(recommendation_id) is the final guard
        return existing
    ps = db.get(ProductSupplier, (reco.product_id, reco.supplier_id))
    cost = ps.unit_cost if ps else None
    po = PurchaseOrder(supplier_id=reco.supplier_id, warehouse_id=reco.warehouse_id, recommendation_id=reco.id,
                       status="approved", created_by=actor,
                       total_value=cost * reco.recommended_qty if cost is not None else None,
                       lines=[PurchaseOrderLine(product_id=reco.product_id, qty_ordered=reco.recommended_qty,
                                                unit_cost=cost)])
    db.add(po)
    db.flush()
    audit.record(db, actor, "purchase_order.create", "purchase_order", po.id,
                 after={"recommendation_id": reco.id, "qty": reco.recommended_qty}, policy_version=reco.policy_version)
    return po


def po_status_change(db: Session, po: PurchaseOrder, new_status: str, actor: str) -> PurchaseOrder:
    dom = invdom.PurchaseOrder(str(po.id), str(po.warehouse_id), po.status)
    old = po.status
    invdom.po_transition(dom, new_status)
    po.status = new_status
    audit.record(db, actor, f"purchase_order.{new_status}", "purchase_order", po.id,
                 before={"status": old}, after={"status": new_status})
    if new_status == "cancelled":                                       # EC-40: re-evaluate immediately
        for line in po.lines:
            prod, wh = db.get(Product, line.product_id), db.get(Warehouse, po.warehouse_id)
            evaluate_one(db, prod, wh, actor=actor)
    return po
