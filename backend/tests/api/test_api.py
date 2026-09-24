"""API / integration / security tests mapped to PRD edge cases (TRD 18.2-18.4)."""
from datetime import datetime, timedelta, timezone

API = "/api/v1"


def sale(txn="T1", qty=1, sku="SKU-1", days_ago=1):
    return {"external_txn_id": txn, "sku": sku, "warehouse": "MAIN", "quantity": qty,
            "sold_at": (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()}


# ---- security / RBAC ---------------------------------------------------------
def test_unauthenticated_is_401(client):
    assert client.get(f"{API}/products").status_code == 401


def test_login_failure_is_generic_and_audited(client, login):
    r = client.post(f"{API}/auth/login", json={"email": "admin@t.io", "password": "wrong-password"})
    assert r.status_code == 401 and r.json()["code"] == "unauthenticated" and r.json()["request_id"]
    events = client.get(f"{API}/audit?action=auth.login_failed", headers=login("admin")).json()
    assert events


def test_analyst_cannot_approve_or_write(client, login, catalog):          # EC-44
    h = login("analyst")
    assert client.post(f"{API}/recommendations/1/approve", json={"expected_version": 1}, headers=h).status_code == 403
    assert client.post(f"{API}/sales", json=sale(), headers=h).status_code == 403
    denied = client.get(f"{API}/audit?action=security.denied", headers=h).json()
    assert denied                                                            # FR-002 telemetry


def test_logout_revokes_token(client, login):                               # EC-45
    h = login("ops")
    assert client.get(f"{API}/auth/me", headers=h).status_code == 200
    client.post(f"{API}/auth/logout", headers=h)
    assert client.get(f"{API}/auth/me", headers=h).status_code == 401


def test_login_rate_limit(client):
    codes = [client.post(f"{API}/auth/login", json={"email": "x@t.io", "password": "bad"}).status_code for _ in range(15)]
    assert 429 in codes


def test_injection_in_sku_rejected_or_inert(client, login):                 # EC-47
    h = login("manager")
    assert client.post(f"{API}/products", json={"sku": "x'; DROP TABLE products;--", "name": "n"}, headers=h).status_code == 422
    assert client.get(f"{API}/products?q=%27%20OR%201=1--", headers=h).status_code == 200


# ---- validation / catalog ----------------------------------------------------
def test_invalid_config_rejected(client, login, catalog):                    # EC-59 / EC-16
    h = login("manager")
    assert client.post(f"{API}/products", json={"sku": "N", "name": "n", "safety_stock": -1}, headers=h).status_code == 422
    r = client.put(f"{API}/products/{catalog['product_id']}/suppliers", headers=h,
                   json={"supplier_id": catalog["supplier_id"], "lead_time_days": 0})
    assert r.status_code == 422
    assert client.put(f"{API}/config/policy", json={"values": {"max_order_qty": -5}}, headers=login("admin")).status_code == 422


def test_optimistic_locking(client, login, catalog):                        # EC-61
    h, pid = login("manager"), catalog["product_id"]
    ok = client.patch(f"{API}/products/{pid}", json={"expected_version": 1, "name": "A"}, headers=h)
    stale = client.patch(f"{API}/products/{pid}", json={"expected_version": 1, "name": "B"}, headers=h)
    assert ok.status_code == 200 and stale.status_code == 409


# ---- sales ingestion -----------------------------------------------------------
def test_sale_idempotent_and_stock_updates(client, login, catalog):         # EC-01 / EC-02
    h = login("ops")
    assert client.post(f"{API}/sales", json=sale("A", 3), headers=h).json()["accepted"] == 1
    again = client.post(f"{API}/sales", json=sale("A", 3), headers=h).json()
    assert again["duplicates"] == 1 and again["accepted"] == 0
    assert client.get(f"{API}/inventory/MAIN/SKU-1", headers=h).json()["on_hand"] == 17


def test_bad_sales_rejected(client, login, catalog):                        # EC-03 / EC-05 / EC-10
    h = login("ops")
    assert client.post(f"{API}/sales", json=sale("F", days_ago=-3), headers=h).status_code == 422   # future
    assert client.post(f"{API}/sales", json=sale("Z", 0), headers=h).status_code == 422              # qty 0
    assert client.post(f"{API}/sales", json=sale("U", sku="NOPE"), headers=h).status_code == 422
    assert client.post(f"{API}/sales", json=sale("BIG", 999), headers=h).status_code == 422          # insufficient stock
    assert client.get(f"{API}/inventory/MAIN/SKU-1", headers=h).json()["on_hand"] == 20


def test_csv_import_partial_and_limits(client, login, catalog):             # EC-03 / EC-46
    h = login("manager")
    ts = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
    csv_ = f"external_txn_id,sku,warehouse,quantity,sold_at\nC1,SKU-1,MAIN,2,{ts}\nC2,SKU-1,MAIN,-1,{ts}\nC1,SKU-1,MAIN,2,{ts}\n"
    r = client.post(f"{API}/sales/import", files={"file": ("s.csv", csv_, "text/csv")}, headers=h).json()
    assert (r["accepted"], r["rejected"], r["duplicates"]) == (1, 1, 1)
    assert client.post(f"{API}/sales/import", files={"file": ("s.exe", b"x", "application/octet-stream")}, headers=h).status_code == 415
    assert client.post(f"{API}/sales/import", files={"file": ("s.csv", "a,b\n1,2\n", "text/csv")}, headers=h).status_code == 422


# ---- end-to-end journey (TRD 18.2 #2, #4, #5) -------------------------------------
def _history(client, h, days=60, per_day=6):
    for d in range(days, 0, -1):
        assert client.post(f"{API}/sales/import?update_stock=false", headers=h, files={"file": ("s.csv",
            "external_txn_id,sku,warehouse,quantity,sold_at\n" +
            f"H{d},SKU-1,MAIN,{per_day},{(datetime.now(timezone.utc) - timedelta(days=d)).isoformat()}\n", "text/csv")}).status_code == 202


def test_e2e_forecast_to_receipt(client, login, catalog):
    h = login("manager")
    _history(client, h)
    assert client.post(f"{API}/forecasts/run", json={"horizon_days": 14}, headers=h).status_code == 202
    fc = client.get(f"{API}/forecasts/SKU-1?warehouse=MAIN", headers=h).json()
    assert len(fc["points"]) == 14 and all(p["yhat"] >= 0 for p in fc["points"])
    assert client.post(f"{API}/forecasts/run", json={}, headers=h).json()["status"] == "already_exists"   # EC-51

    ev = client.post(f"{API}/recommendations/evaluate", json={"sku": "SKU-1"}, headers=h).json()
    assert ev["created"] == 1
    again = client.post(f"{API}/recommendations/evaluate", json={"sku": "SKU-1"}, headers=h).json()
    assert again["suppressed"] == 1                                                     # EC-35 no duplicate
    rec = client.get(f"{API}/recommendations", headers=h).json()[0]
    assert rec["recommended_qty"] % 6 == 0 and rec["reorder_point"] > rec["available"]  # explainability

    a = client.post(f"{API}/recommendations/{rec['recommendation_id']}/approve", json={"expected_version": rec["version"]}, headers=h)
    assert a.status_code == 200
    replay = client.post(f"{API}/recommendations/{rec['recommendation_id']}/approve", json={"expected_version": rec["version"]}, headers=h)
    assert replay.status_code == 409                                                     # double-click / replay

    po = client.get(f"{API}/purchase-orders", headers=h).json()[0]
    assert client.post(f"{API}/purchase-orders/{po['id']}/status", json={"status": "sent"}, headers=h).status_code == 200
    assert client.get(f"{API}/inventory/MAIN/SKU-1", headers=h).json()["inbound"] == po["lines"][0]["qty_ordered"]
    q = po["lines"][0]["qty_ordered"]
    r1 = client.post(f"{API}/purchase-orders/{po['id']}/receive", json={"sku": "SKU-1", "quantity": 1}, headers=h)
    assert r1.json()["status"] == "partially_received"                                   # EC-08
    over = client.post(f"{API}/purchase-orders/{po['id']}/receive", json={"sku": "SKU-1", "quantity": q}, headers=h)
    assert over.status_code == 422                                                       # EC-09
    client.post(f"{API}/purchase-orders/{po['id']}/receive", json={"sku": "SKU-1", "quantity": q - 1}, headers=h)
    assert client.get(f"{API}/purchase-orders?status=received", headers=h).json()

    trail = {e["action"] for e in client.get(f"{API}/audit?limit=200", headers=login("analyst")).json()}
    assert {"recommendation.create", "recommendation.approve", "purchase_order.create", "purchase_order.receive"} <= trail


def test_reject_needs_reason_and_is_audited(client, login, catalog):         # EC-37
    h = login("manager")
    _history(client, h)
    client.post(f"{API}/forecasts/run", json={}, headers=h)
    client.post(f"{API}/recommendations/evaluate", json={"sku": "SKU-1"}, headers=h)
    rec = client.get(f"{API}/recommendations", headers=h).json()[0]
    url = f"{API}/recommendations/{rec['recommendation_id']}/reject"
    assert client.post(url, json={"expected_version": rec["version"]}, headers=h).status_code == 422
    assert client.post(url, json={"expected_version": rec["version"], "reason": "overstocked"}, headers=h).status_code == 200


def test_no_history_is_manual_review_not_fabricated(client, login, catalog):  # EC-66 / PRD 5.3
    h = login("manager")
    client.post(f"{API}/forecasts/run", json={}, headers=h)
    client.post(f"{API}/recommendations/evaluate", json={"sku": "SKU-1"}, headers=h)
    assert client.get(f"{API}/recommendations", headers=h).json() == []
    assert client.get(f"{API}/alerts", headers=h).json()


def test_export_neutralises_formulas(client, login, catalog):                # EC-47 (CSV) / EC-70
    assert client.get(f"{API}/recommendations/export.csv", headers=login("ops")).status_code == 403
    assert client.get(f"{API}/recommendations/export.csv", headers=login("analyst")).status_code == 200


def test_dashboard_reports_freshness(client, login, catalog):                # EC-57
    d = client.get(f"{API}/dashboard", headers=login("analyst")).json()
    assert d["forecast_status"] == "no_forecast_yet" and d["as_of"]
