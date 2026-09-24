"""Receiving (FR-015) - stock, PO status and audit change in one transaction (EC-07..EC-09, EC-42)."""
from __future__ import annotations

from sqlalchemy.orm import Session

from backend.app.domain import inventory as invdom
from backend.app.models.entities import InventoryBalance, InventoryMovement, PurchaseOrder
from backend.app.services import alerts, audit


def receive(db: Session, po: PurchaseOrder, product_id: int, qty: int, *, actor: str,
            over_receipt_authorized: bool = False) -> PurchaseOrder:
    dom = invdom.PurchaseOrder(str(po.id), str(po.warehouse_id), po.status,
                               [invdom.POLine(str(l.product_id), l.qty_ordered, l.qty_received) for l in po.lines])
    added = invdom.receive(dom, str(product_id), qty, over_receipt_authorized)   # raises InventoryError
    line = next(l for l in po.lines if l.product_id == product_id)
    line.qty_received += added
    if over_receipt_authorized and line.qty_received > line.qty_ordered:
        alerts.raise_alert(db, "over_receipt", "high", "purchase_order", po.id, "over-receipt authorized")
    bal = db.get(InventoryBalance, (product_id, po.warehouse_id))
    if bal is None:
        bal = InventoryBalance(product_id=product_id, warehouse_id=po.warehouse_id, on_hand=0)
        db.add(bal)
    bal.on_hand += added
    db.add(InventoryMovement(product_id=product_id, warehouse_id=po.warehouse_id, movement_type="receipt",
                             quantity=added, reference_type="purchase_order", reference_id=str(po.id)))
    before = po.status
    po.status = dom.status
    audit.record(db, actor, "purchase_order.receive", "purchase_order", po.id, before={"status": before},
                 after={"status": po.status, "product_id": product_id, "received": added})
    return po
