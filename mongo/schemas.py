"""Pydantic document schemas. Every write goes through one of these first -
`extra="forbid"` means an unexpected field (or an injected operator key
disguised as a field) is rejected before it ever reaches MongoDB, and typed
fields reject the wrong shape (e.g. a dict where a string was expected)."""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ProductDoc(Strict):
    sku: str = Field(min_length=1, max_length=64)
    name: str = Field(min_length=1, max_length=255)
    category: str = ""
    status: str = Field("active", pattern="^(active|archived)$")
    safety_stock: float = Field(0, ge=0)
    moq: int = Field(1, gt=0)
    pack_size: int = Field(1, gt=0)
    unit_cost: float | None = Field(None, gt=0)
    supplier_id: str | None = None
    lead_time_days: float = Field(7, gt=0, le=365)
    warehouse: str = "MAIN"
    avg_daily_sales_hint: float | None = Field(None, ge=0, description=(
        "Declared demand-rate fallback from master-data import, used by the automation "
        "pipeline only when real transaction history is insufficient (cold start, EC-19)."))


class SupplierDoc(Strict):
    supplier_id: str = Field(min_length=1, max_length=32)
    name: str
    status: str = Field("active", pattern="^(active|inactive)$")
    on_time_pct: float | None = Field(None, ge=0, le=100)


class SalesTransactionDoc(Strict):
    external_txn_id: str = Field(min_length=1, max_length=64)
    source: str = "api"
    sku: str
    warehouse: str
    quantity: int = Field(gt=0)
    unit_price: float | None = Field(None, ge=0)
    sold_at: datetime


class InventorySnapshotDoc(Strict):
    sku: str
    warehouse: str
    on_hand: int = Field(ge=0)
    reserved: int = Field(0, ge=0)
    damaged: int = Field(0, ge=0)
    updated_at: datetime


class RecommendationDoc(Strict):
    idempotency_key: str
    sku: str
    warehouse: str
    recommended_qty: int = Field(gt=0)
    automation_state: str
    reason: str
    explanation: dict
    status: str = "PENDING"
    created_at: datetime


class AlertDoc(Strict):
    type: str
    severity: str = Field(pattern="^(critical|high|medium|info)$")
    entity_type: str
    entity_id: str
    dedupe_key: str
    message: str
    status: str = "open"
    created_at: datetime


class AuditEventDoc(Strict):
    actor: str
    action: str
    entity_type: str
    entity_id: str
    before: dict | None = None
    after: dict | None = None
    reason: str | None = None
    occurred_at: datetime


class ModelRegistryDoc(Strict):
    dataset: str
    model_name: str
    metrics: dict
    feature_importance_method: str
    trained_at: datetime
    train_rows: int
    is_best: bool = False
