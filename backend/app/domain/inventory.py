"""Inventory ledger and purchase-order rules (PRD 5, FR-005/014/015, EC-07..EC-11, EC-39..EC-42)."""
from __future__ import annotations

from dataclasses import dataclass, field


class InventoryError(ValueError):
    pass


@dataclass
class Balance:
    on_hand: int
    reserved: int = 0
    damaged: int = 0
    version: int = 1

    @property
    def available(self) -> int:
        return max(0, self.on_hand - self.reserved - self.damaged)


def apply_sale(b: Balance, qty: int, allow_backorder: bool = False) -> Balance:
    """EC-10: a sale can never drive on-hand negative unless backorders are enabled."""
    if qty <= 0:
        raise InventoryError("sale quantity must be positive")
    if qty > b.on_hand and not allow_backorder:
        raise InventoryError("insufficient stock")
    return Balance(b.on_hand - qty, b.reserved, b.damaged, b.version + 1)


def apply_adjustment(b: Balance, delta: int, reason: str, privileged: bool = False) -> Balance:
    """EC-11: correction that would make on-hand negative is blocked unless privileged."""
    if not reason.strip():
        raise InventoryError("adjustment requires a reason")
    new = b.on_hand + delta
    if new < 0 and not privileged:
        raise InventoryError("adjustment would create negative on-hand")
    return Balance(max(new, 0) if privileged else new, b.reserved, b.damaged, b.version + 1)


def apply_return(b: Balance, qty: int, sellable: bool) -> Balance:
    """EC-62/63: damaged returns go to quarantine, not sellable on-hand."""
    if qty <= 0:
        raise InventoryError("return quantity must be positive")
    if sellable:
        return Balance(b.on_hand + qty, b.reserved, b.damaged, b.version + 1)
    return Balance(b.on_hand + qty, b.reserved, b.damaged + qty, b.version + 1)


@dataclass
class POLine:
    sku: str
    qty_ordered: int
    qty_received: int = 0

    @property
    def open_qty(self) -> int:
        return max(0, self.qty_ordered - self.qty_received)


@dataclass
class PurchaseOrder:
    id: str
    warehouse: str
    status: str = "draft"     # draft|approved|sent|partially_received|received|cancelled
    lines: list[POLine] = field(default_factory=list)

    def eligible_inbound(self, sku: str, warehouse: str) -> int:
        """EC-39/40/72: only sent/partial POs of the SAME warehouse count as inbound."""
        if self.status not in ("sent", "partially_received") or self.warehouse != warehouse:
            return 0
        return sum(l.open_qty for l in self.lines if l.sku == sku)


_PO_FLOW = {
    "draft": {"approved", "cancelled"},
    "approved": {"sent", "cancelled"},
    "sent": {"partially_received", "received", "cancelled"},
    "partially_received": {"partially_received", "received", "cancelled"},
}


def po_transition(po: PurchaseOrder, new_status: str) -> None:
    if new_status not in _PO_FLOW.get(po.status, set()):
        raise InventoryError(f"invalid PO transition {po.status} -> {new_status}")
    po.status = new_status


def receive(po: PurchaseOrder, sku: str, qty: int, over_receipt_authorized: bool = False) -> int:
    """Record a receipt on a PO line; returns quantity to add to on-hand.

    EC-07 exact receipt closes the PO, EC-08 partial keeps remainder open,
    EC-09 over-receipt needs explicit authorization, EC-42 short-close is a cancel of the remainder.
    """
    if po.status not in ("sent", "partially_received"):
        raise InventoryError(f"cannot receive against a {po.status} PO")
    if qty <= 0:
        raise InventoryError("receipt quantity must be positive")
    line = next((l for l in po.lines if l.sku == sku), None)
    if line is None:
        raise InventoryError("SKU not on this purchase order")
    if qty > line.open_qty and not over_receipt_authorized:
        raise InventoryError("over-receipt requires authorization")
    line.qty_received += qty
    po.status = "received" if all(l.open_qty == 0 for l in po.lines) else "partially_received"
    return qty


def short_close(po: PurchaseOrder) -> None:
    """EC-42: authorized user closes a partially received PO; remaining inbound disappears."""
    if po.status != "partially_received":
        raise InventoryError("only partially received POs can be short-closed")
    for l in po.lines:
        l.qty_ordered = l.qty_received
    po.status = "received"
