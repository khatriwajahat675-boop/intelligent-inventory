"""API test harness: in-memory SQLite, deterministic users. Requires: pip install -r backend/requirements-dev.txt"""
import os

os.environ.setdefault("DATABASE_URL", "sqlite://")
os.environ.setdefault("JWT_SECRET", "test-secret-test-secret-test-secret-1234")
os.environ["USE_CELERY"] = "false"

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.app import db as dbmod
from backend.app.api import routes
from backend.app.main import create_app
from backend.app.models.entities import Base, User
from backend.app.security.passwords import hash_password

PASSWORD = "Correct-Horse-9"
ROLES = {"admin": "admin", "manager": "inventory_manager", "ops": "operations_user", "analyst": "analyst"}


@pytest.fixture()
def client():
    routes._login_limiter = None
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    dbmod._engine = engine
    dbmod._Session = sessionmaker(bind=engine, expire_on_commit=False, autoflush=False)
    Base.metadata.create_all(engine)
    with dbmod._Session() as s:
        for name, role in ROLES.items():
            s.add(User(email=f"{name}@t.io", password_hash=hash_password(PASSWORD), role=role))
        s.commit()
    with TestClient(create_app()) as c:
        yield c


@pytest.fixture()
def login(client):
    def _login(name: str) -> dict:
        r = client.post("/api/v1/auth/login", json={"email": f"{name}@t.io", "password": PASSWORD})
        assert r.status_code == 200, r.text
        return {"Authorization": f"Bearer {r.json()['access_token']}"}
    return _login


@pytest.fixture()
def catalog(client, login):
    """One warehouse, supplier, SKU (lead 5d, MOQ 10, pack 6) and 20 units on hand."""
    h = login("manager")
    client.post("/api/v1/warehouses", json={"code": "MAIN", "name": "Main"}, headers=h)
    sup = client.post("/api/v1/suppliers", json={"code": "S1", "name": "Sup", "default_lead_time_days": 5}, headers=h).json()["id"]
    p = client.post("/api/v1/products", json={"sku": "SKU-1", "name": "Widget", "safety_stock": 10}, headers=h).json()["id"]
    client.put(f"/api/v1/products/{p}/suppliers", headers=h,
               json={"supplier_id": sup, "lead_time_days": 5, "moq": 10, "pack_size": 6, "unit_cost": 2.0, "preferred": True})
    client.post("/api/v1/inventory/adjustments", headers=h,
                json={"sku": "SKU-1", "warehouse": "MAIN", "delta": 20, "reason": "opening balance"})
    return {"product_id": p, "supplier_id": sup}
