"""ORM models - TRD 11.1 core entities, 11.2 integrity rules."""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (JSON, CheckConstraint, DateTime, Float, ForeignKey, Index, Integer, String,
                        Text, UniqueConstraint, event)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.db import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(16), default="active")      # active | disabled
    session_id: Mapped[str] = mapped_column(String(64), default="")        # rotate to revoke tokens
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Product(Base):
    __tablename__ = "products"
    __table_args__ = (CheckConstraint("safety_stock >= 0", name="ck_product_safety_nonneg"),
                      CheckConstraint("status IN ('active','archived')", name="ck_product_status"))
    id: Mapped[int] = mapped_column(primary_key=True)
    sku: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(255))
    category: Mapped[str] = mapped_column(String(64), default="")
    uom: Mapped[str] = mapped_column(String(16), default="unit")
    status: Mapped[str] = mapped_column(String(16), default="active")
    safety_stock: Mapped[float] = mapped_column(Float, default=0)
    reorder_mode: Mapped[str] = mapped_column(String(16), default="forecast")  # forecast | min_max | manual
    abc_class: Mapped[str | None] = mapped_column(String(1), nullable=True)
    version: Mapped[int] = mapped_column(Integer, default=1)                # optimistic locking (EC-61)


class Supplier(Base):
    __tablename__ = "suppliers"
    __table_args__ = (CheckConstraint("default_lead_time_days > 0", name="ck_supplier_lead_pos"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    code: Mapped[str] = mapped_column(String(32), unique=True)
    name: Mapped[str] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(String(16), default="active")       # active | inactive
    default_lead_time_days: Mapped[float] = mapped_column(Float, default=7)
    on_time_pct: Mapped[float | None] = mapped_column(Float, nullable=True)


class ProductSupplier(Base):
    __tablename__ = "product_suppliers"
    __table_args__ = (CheckConstraint("lead_time_days > 0", name="ck_ps_lead_pos"),
                      CheckConstraint("moq > 0", name="ck_ps_moq_pos"),
                      CheckConstraint("pack_size > 0", name="ck_ps_pack_pos"))
    product_id: Mapped[int] = mapped_column(ForeignKey("products.id"), primary_key=True)
    supplier_id: Mapped[int] = mapped_column(ForeignKey("suppliers.id"), primary_key=True)
    lead_time_days: Mapped[float] = mapped_column(Float)
    moq: Mapped[int] = mapped_column(Integer, default=1)
    pack_size: Mapped[int] = mapped_column(Integer, default=1)
    unit_cost: Mapped[float | None] = mapped_column(Float, nullable=True)
    preferred: Mapped[int] = mapped_column(Integer, default=0)


class Warehouse(Base):
    __tablename__ = "warehouses"
    id: Mapped[int] = mapped_column(primary_key=True)
    code: Mapped[str] = mapped_column(String(32), unique=True)
    name: Mapped[str] = mapped_column(String(255))
    timezone: Mapped[str] = mapped_column(String(64), default="UTC")
    status: Mapped[str] = mapped_column(String(16), default="active")


class SalesTransaction(Base):
    """Atomic demand fact. (source, external_txn_id) is the idempotency key (EC-02)."""
    __tablename__ = "sales_transactions"
    __table_args__ = (UniqueConstraint("source", "external_txn_id", name="uq_sales_source_txn"),
                      CheckConstraint("quantity > 0", name="ck_sales_qty_pos"),
                      Index("ix_sales_prod_wh_time", "product_id", "warehouse_id", "sold_at"))
    id: Mapped[int] = mapped_column(primary_key=True)
    external_txn_id: Mapped[str] = mapped_column(String(64))
    source: Mapped[str] = mapped_column(String(32), default="api")
    product_id: Mapped[int] = mapped_column(ForeignKey("products.id"))
    warehouse_id: Mapped[int] = mapped_column(ForeignKey("warehouses.id"))
    quantity: Mapped[int] = mapped_column(Integer)
    unit_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    sold_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))       # stored UTC (EC-68)
    business_date: Mapped[str] = mapped_column(String(10))                    # YYYY-MM-DD in business tz (EC-06)


class InventoryMovement(Base):
    """Append-only stock ledger."""
    __tablename__ = "inventory_movements"
    id: Mapped[int] = mapped_column(primary_key=True)
    product_id: Mapped[int] = mapped_column(ForeignKey("products.id"), index=True)
    warehouse_id: Mapped[int] = mapped_column(ForeignKey("warehouses.id"))
    movement_type: Mapped[str] = mapped_column(String(24))   # sale|receipt|adjustment|return|damage
    quantity: Mapped[int] = mapped_column(Integer)           # signed
    reference_type: Mapped[str | None] = mapped_column(String(24), nullable=True)
    reference_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class InventoryBalance(Base):
    __tablename__ = "inventory_balances"
    __table_args__ = (CheckConstraint("on_hand >= 0", name="ck_bal_onhand_nonneg"),
                      CheckConstraint("reserved >= 0 AND damaged >= 0", name="ck_bal_nonneg"))
    product_id: Mapped[int] = mapped_column(ForeignKey("products.id"), primary_key=True)
    warehouse_id: Mapped[int] = mapped_column(ForeignKey("warehouses.id"), primary_key=True)
    on_hand: Mapped[int] = mapped_column(Integer, default=0)
    reserved: Mapped[int] = mapped_column(Integer, default=0)
    damaged: Mapped[int] = mapped_column(Integer, default=0)
    version: Mapped[int] = mapped_column(Integer, default=1)
    __mapper_args__ = {"version_id_col": version}                            # optimistic lock (EC-10)


class ForecastRun(Base):
    __tablename__ = "forecast_runs"
    __table_args__ = (UniqueConstraint("job_key", name="uq_forecast_job_key"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    job_key: Mapped[str] = mapped_column(String(128))                        # forecast:{scope}:{period} (EC-51)
    model_type: Mapped[str] = mapped_column(String(32))
    model_version: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(16), default="running")       # running|succeeded|failed
    training_window: Mapped[str] = mapped_column(String(64), default="")
    horizon_days: Mapped[int] = mapped_column(Integer, default=14)
    metrics_json: Mapped[dict] = mapped_column(JSON, default=dict)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class Forecast(Base):
    """Immutable once published (TRD 11.2): corrections create a new run."""
    __tablename__ = "forecasts"
    __table_args__ = (Index("ix_fc_prod_wh_date", "product_id", "warehouse_id", "forecast_date"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("forecast_runs.id"), index=True)
    product_id: Mapped[int] = mapped_column(ForeignKey("products.id"))
    warehouse_id: Mapped[int] = mapped_column(ForeignKey("warehouses.id"))
    forecast_date: Mapped[str] = mapped_column(String(10))
    yhat: Mapped[float] = mapped_column(Float)                               # clamped, non-negative
    yhat_raw: Mapped[float | None] = mapped_column(Float, nullable=True)     # diagnostics (EC-21)
    lower: Mapped[float | None] = mapped_column(Float, nullable=True)
    upper: Mapped[float | None] = mapped_column(Float, nullable=True)
    quality_flag: Mapped[str] = mapped_column(String(16), default="ok")


class Recommendation(Base):
    __tablename__ = "recommendations"
    __table_args__ = (UniqueConstraint("idempotency_key", name="uq_reco_idem"),
                      CheckConstraint("recommended_qty > 0", name="ck_reco_qty_pos"))
    id: Mapped[int] = mapped_column(primary_key=True)
    idempotency_key: Mapped[str] = mapped_column(String(160))
    product_id: Mapped[int] = mapped_column(ForeignKey("products.id"), index=True)
    warehouse_id: Mapped[int] = mapped_column(ForeignKey("warehouses.id"))
    supplier_id: Mapped[int | None] = mapped_column(ForeignKey("suppliers.id"), nullable=True)
    forecast_run_id: Mapped[int | None] = mapped_column(ForeignKey("forecast_runs.id"), nullable=True)
    policy_version: Mapped[str] = mapped_column(String(32))
    recommended_qty: Mapped[int] = mapped_column(Integer)
    automation_state: Mapped[str] = mapped_column(String(24))
    status: Mapped[str] = mapped_column(String(16), default="PENDING")       # PENDING|APPROVED|REJECTED|ON_HOLD|CANCELLED
    version: Mapped[int] = mapped_column(Integer, default=1)
    explanation_json: Mapped[dict] = mapped_column(JSON, default=dict)
    decision_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class PurchaseOrder(Base):
    __tablename__ = "purchase_orders"
    id: Mapped[int] = mapped_column(primary_key=True)
    supplier_id: Mapped[int] = mapped_column(ForeignKey("suppliers.id"))
    warehouse_id: Mapped[int] = mapped_column(ForeignKey("warehouses.id"))
    recommendation_id: Mapped[int | None] = mapped_column(ForeignKey("recommendations.id"), unique=True, nullable=True)
    status: Mapped[str] = mapped_column(String(24), default="draft")
    total_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    expected_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_by: Mapped[str] = mapped_column(String(64), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    lines: Mapped[list["PurchaseOrderLine"]] = relationship(back_populates="po", cascade="all, delete-orphan")


class PurchaseOrderLine(Base):
    __tablename__ = "purchase_order_lines"
    __table_args__ = (CheckConstraint("qty_ordered > 0 AND qty_received >= 0", name="ck_pol_qty"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    po_id: Mapped[int] = mapped_column(ForeignKey("purchase_orders.id"), index=True)
    product_id: Mapped[int] = mapped_column(ForeignKey("products.id"))
    qty_ordered: Mapped[int] = mapped_column(Integer)
    qty_received: Mapped[int] = mapped_column(Integer, default=0)
    unit_cost: Mapped[float | None] = mapped_column(Float, nullable=True)
    po: Mapped[PurchaseOrder] = relationship(back_populates="lines")


class Alert(Base):
    __tablename__ = "alerts"
    __table_args__ = (UniqueConstraint("dedupe_key", "status", name="uq_alert_dedupe_status"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    type: Mapped[str] = mapped_column(String(48))
    severity: Mapped[str] = mapped_column(String(16))          # critical|high|medium|info
    entity_type: Mapped[str] = mapped_column(String(32))
    entity_id: Mapped[str] = mapped_column(String(64))
    dedupe_key: Mapped[str] = mapped_column(String(160))       # EC-38: one open alert per unresolved event
    status: Mapped[str] = mapped_column(String(16), default="open")
    message: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class AuditEvent(Base):
    """Append-only at the application layer (FR-018, EC-58)."""
    __tablename__ = "audit_events"
    id: Mapped[int] = mapped_column(primary_key=True)
    actor: Mapped[str] = mapped_column(String(64))             # user email or service identity
    action: Mapped[str] = mapped_column(String(64), index=True)
    entity_type: Mapped[str] = mapped_column(String(32))
    entity_id: Mapped[str] = mapped_column(String(64))
    before_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    after_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    policy_version: Mapped[str | None] = mapped_column(String(32), nullable=True)
    model_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    request_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class PolicyConfig(Base):
    """Versioned business policy stored separately from infra config (TRD 17.4, EC-60)."""
    __tablename__ = "policy_configs"
    id: Mapped[int] = mapped_column(primary_key=True)
    version: Mapped[str] = mapped_column(String(32), unique=True)
    values_json: Mapped[dict] = mapped_column(JSON)
    active: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


@event.listens_for(AuditEvent, "before_update")
@event.listens_for(AuditEvent, "before_delete")
def _audit_is_append_only(*_):
    raise PermissionError("audit events are append-only")
