"""Request/response schemas with strict validation (TRD 12.1, 15.1)."""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class LoginIn(Strict):
    email: str = Field(min_length=3, max_length=255)
    password: str = Field(min_length=1, max_length=128)


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str


class ProductIn(Strict):
    sku: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9._|\- ]+$")
    name: str = Field(min_length=1, max_length=255)
    category: str = ""
    uom: str = "unit"
    safety_stock: float = Field(0, ge=0)                       # EC-59
    reorder_mode: str = Field("forecast", pattern="^(forecast|min_max|manual)$")


class ProductPatch(Strict):
    expected_version: int
    name: str | None = Field(None, min_length=1, max_length=255)
    category: str | None = None
    safety_stock: float | None = Field(None, ge=0)
    status: str | None = Field(None, pattern="^(active|archived)$")
    reorder_mode: str | None = Field(None, pattern="^(forecast|min_max|manual)$")


class ProductOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    sku: str
    name: str
    category: str
    status: str
    safety_stock: float
    reorder_mode: str
    version: int


class SupplierIn(Strict):
    code: str = Field(min_length=1, max_length=32)
    name: str = Field(min_length=1, max_length=255)
    default_lead_time_days: float = Field(7, gt=0, le=365)     # EC-16


class ProductSupplierIn(Strict):
    supplier_id: int
    lead_time_days: float = Field(gt=0, le=365)
    moq: int = Field(1, gt=0)
    pack_size: int = Field(1, gt=0)
    unit_cost: float | None = Field(None, gt=0)
    preferred: bool = False


class SaleIn(Strict):
    external_txn_id: str = Field(min_length=1, max_length=64)
    sku: str
    warehouse: str
    quantity: int = Field(gt=0)
    unit_price: float | None = Field(None, ge=0)
    sold_at: datetime


class AdjustmentIn(Strict):
    sku: str
    warehouse: str
    delta: int
    reason: str = Field(min_length=3)
    privileged: bool = False


class ForecastRunIn(Strict):
    horizon_days: int = Field(14, ge=1, le=90)
    scope: str = "all"


class EvaluateIn(Strict):
    sku: str | None = None
    warehouse: str | None = None


class DecisionIn(Strict):
    expected_version: int
    reason: str | None = None


class POCreateIn(Strict):
    supplier_id: int
    warehouse: str
    lines: list[dict] = Field(min_length=1)


class POStatusIn(Strict):
    status: str = Field(pattern="^(approved|sent|cancelled)$")


class ReceiveIn(Strict):
    sku: str
    quantity: int = Field(gt=0)
    over_receipt_authorized: bool = False


class PolicyIn(Strict):
    values: dict


class WarehouseIn(Strict):
    code: str = Field(min_length=1, max_length=32)
    name: str = Field(min_length=1, max_length=255)
    timezone: str = "UTC"
