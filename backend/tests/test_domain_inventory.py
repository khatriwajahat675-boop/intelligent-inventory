import unittest

from backend.app.domain import inventory as inv


def po(status="sent", wh="MAIN", qty=100):
    return inv.PurchaseOrder("PO1", wh, status, [inv.POLine("A", qty)])


class Ledger(unittest.TestCase):
    def test_sale_cannot_go_negative(self):                        # EC-10
        with self.assertRaises(inv.InventoryError):
            inv.apply_sale(inv.Balance(1), 2)
        self.assertEqual(inv.apply_sale(inv.Balance(5), 5).on_hand, 0)

    def test_adjustment_rules(self):                               # EC-11
        with self.assertRaises(inv.InventoryError):
            inv.apply_adjustment(inv.Balance(3), -5, "count")
        with self.assertRaises(inv.InventoryError):
            inv.apply_adjustment(inv.Balance(3), 1, "  ")
        self.assertEqual(inv.apply_adjustment(inv.Balance(3), -5, "audit", privileged=True).on_hand, 0)

    def test_returns(self):                                        # EC-62/63
        ok = inv.apply_return(inv.Balance(10), 2, sellable=True)
        bad = inv.apply_return(inv.Balance(10), 2, sellable=False)
        self.assertEqual((ok.available, bad.available), (12, 10))

    def test_version_increments(self):
        self.assertEqual(inv.apply_sale(inv.Balance(5), 1).version, 2)


class PurchaseOrders(unittest.TestCase):
    def test_exact_receipt_closes(self):                           # EC-07
        p = po(); inv.receive(p, "A", 100)
        self.assertEqual((p.status, p.eligible_inbound("A", "MAIN")), ("received", 0))

    def test_partial_then_remaining_inbound(self):                 # EC-08
        p = po(); inv.receive(p, "A", 40)
        self.assertEqual((p.status, p.eligible_inbound("A", "MAIN")), ("partially_received", 60))

    def test_over_receipt_needs_authorization(self):               # EC-09
        p = po()
        with self.assertRaises(inv.InventoryError):
            inv.receive(p, "A", 101)
        self.assertEqual(inv.receive(p, "A", 101, over_receipt_authorized=True), 101)

    def test_only_sent_pos_count_as_inbound(self):                 # EC-39
        for s in ("draft", "approved", "cancelled", "received"):
            self.assertEqual(po(s).eligible_inbound("A", "MAIN"), 0, s)

    def test_cancelled_po_removes_inbound(self):                   # EC-40
        p = po(); inv.po_transition(p, "cancelled")
        self.assertEqual(p.eligible_inbound("A", "MAIN"), 0)

    def test_wrong_warehouse_does_not_cover(self):                 # EC-72
        self.assertEqual(po(wh="OTHER").eligible_inbound("A", "MAIN"), 0)

    def test_short_close(self):                                    # EC-42
        p = po(); inv.receive(p, "A", 30); inv.short_close(p)
        self.assertEqual((p.status, p.eligible_inbound("A", "MAIN")), ("received", 0))

    def test_invalid_transitions(self):
        with self.assertRaises(inv.InventoryError):
            inv.po_transition(po("received"), "sent")
        with self.assertRaises(inv.InventoryError):
            inv.receive(po("draft"), "A", 1)


if __name__ == "__main__":
    unittest.main()
