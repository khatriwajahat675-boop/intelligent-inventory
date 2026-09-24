"""Inventory decision engine - TRD 14.1 pseudocode + PRD 5.2/5.3 guardrails.

Pure functions, no I/O: the same inputs always give the same Recommendation, so a
decision can be reproduced later from the stored forecast, stock state, lead time
and policy version (Definition of Done, TRD 18.5).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum


class AutomationState(str, Enum):
    SUPPRESSED = "SUPPRESSED"
    AUTO_APPROVE = "AUTO_APPROVE"
    REQUIRES_APPROVAL = "REQUIRES_APPROVAL"
    BLOCKED = "BLOCKED"


@dataclass(frozen=True)
class Policy:
    version: str = "policy_v1"
    auto_approve_enabled: bool = False        # recommendation-only by default (PRD assumptions)
    auto_approve_max_qty: int = 500
    auto_approve_max_value: float = 5_000.0
    max_order_qty: int = 5_000                # hard cap (PRD 5.3)
    max_forecast_uncertainty: float = 0.35    # relative half-width of interval
    forecast_freshness: timedelta = timedelta(hours=36)


@dataclass(frozen=True)
class Product:
    sku: str
    status: str                 # active | archived
    moq: int
    pack_size: int
    safety_stock: float
    unit_cost: float | None = None


@dataclass(frozen=True)
class Supplier:
    id: str
    status: str                 # active | inactive
    lead_time_days: float


@dataclass(frozen=True)
class Stock:
    on_hand: float
    reserved: float = 0.0
    damaged: float = 0.0        # quarantined/damaged - never sellable (EC-63)
    eligible_inbound: float = 0.0   # only sent/approved PO remainder (EC-39)

    @property
    def available(self) -> float:
        return max(0.0, self.on_hand - self.reserved - self.damaged)


@dataclass(frozen=True)
class Forecast:
    run_id: str
    model_version: str
    daily: tuple[float, ...]            # point forecast per day starting today
    generated_at: datetime
    quality_flag: str = "ok"            # ok | low_confidence | fallback
    uncertainty: float = 0.0            # relative interval half-width


@dataclass
class Recommendation:
    state: AutomationState
    reason: str
    sku: str
    recommended_qty: int = 0
    explanation: dict = field(default_factory=dict)
    constraint_adjustments: list[str] = field(default_factory=list)


def idempotency_key(sku: str, warehouse: str, decision_window: str) -> str:
    """TRD 14.3: reorder:{product}:{warehouse}:{decision_window}."""
    return f"reorder:{sku}:{warehouse}:{decision_window}"


def lead_time_demand(daily: tuple[float, ...], lead_time_days: float) -> tuple[float, bool]:
    """Sum of non-negative forecast over the lead time. Returns (demand, horizon_extended)."""
    days = max(1, math.ceil(lead_time_days))
    clean = [max(0.0, float(x)) for x in daily]           # EC-21
    extended = len(clean) < days
    if extended and clean:
        clean = clean + [sum(clean) / len(clean)] * (days - len(clean))
    return float(sum(clean[:days])), extended


def apply_moq_pack_caps(shortage: float, moq: int, pack: int, cap: int) -> tuple[int, list[str]]:
    """Round shortage up to MOQ then pack multiple, then apply the cap (rounded DOWN to a pack)."""
    adj: list[str] = []
    qty = math.ceil(shortage)
    if qty < moq:
        qty = moq
        adj.append(f"raised_to_moq_{moq}")
    rounded = math.ceil(qty / pack) * pack
    if rounded != qty:
        adj.append(f"rounded_to_pack_size_{pack}")
    qty = rounded
    if qty > cap:
        qty = (cap // pack) * pack
        adj.append(f"capped_at_{cap}")
    return int(qty), adj


def evaluate(
    product: Product,
    supplier: Supplier | None,
    stock: Stock,
    forecast: Forecast | None,
    policy: Policy,
    now: datetime,
    has_open_recommendation: bool = False,
) -> Recommendation:
    def out(state, reason, **kw):
        return Recommendation(state, reason, product.sku, **kw)

    # ---- hard guardrails (PRD 5.3) -----------------------------------------
    if product.status != "active":
        return out(AutomationState.BLOCKED, "sku_archived")                  # EC-14
    if supplier is None:
        return out(AutomationState.BLOCKED, "missing_supplier")              # EC-13
    if supplier.status != "active":
        return out(AutomationState.BLOCKED, "supplier_inactive")             # EC-17
    if not supplier.lead_time_days or supplier.lead_time_days <= 0 or math.isnan(supplier.lead_time_days):
        return out(AutomationState.BLOCKED, "invalid_lead_time")             # EC-16
    if product.moq <= 0 or product.pack_size <= 0 or product.safety_stock < 0:
        return out(AutomationState.BLOCKED, "invalid_product_policy")        # EC-59
    if forecast is None or not forecast.daily:
        return out(AutomationState.REQUIRES_APPROVAL, "forecast_unavailable_manual_review")  # never fabricate

    # ---- calculation (TRD 14.1) -------------------------------------------
    ltd, extended = lead_time_demand(forecast.daily, supplier.lead_time_days)
    reorder_point = ltd + product.safety_stock
    position = stock.available + stock.eligible_inbound
    shortage = reorder_point - position
    expl = {
        "forecast_run_id": forecast.run_id, "model_version": forecast.model_version,
        "policy_version": policy.version, "lead_time_days": supplier.lead_time_days,
        "lead_time_demand": round(ltd, 2), "safety_stock": product.safety_stock,
        "available": stock.available, "eligible_inbound": stock.eligible_inbound,
        "reorder_point": round(reorder_point, 2), "raw_shortage": round(shortage, 2),
        "forecast_quality": forecast.quality_flag,
    }
    if shortage <= 0:
        reason = "covered_by_inbound" if stock.available < reorder_point else "stock_sufficient"  # EC-30
        return out(AutomationState.SUPPRESSED, reason, explanation=expl)     # never a negative qty
    if has_open_recommendation:
        return out(AutomationState.SUPPRESSED, "open_recommendation_exists", explanation=expl)  # EC-35

    qty, adj = apply_moq_pack_caps(shortage, product.moq, product.pack_size, policy.max_order_qty)
    if extended:
        adj.append("forecast_horizon_extended_with_mean")
    if qty <= 0:
        return out(AutomationState.REQUIRES_APPROVAL, "cap_below_moq_manual_review",
                   explanation=expl, constraint_adjustments=adj)
    value = qty * product.unit_cost if product.unit_cost is not None else None
    expl.update(recommended_qty=qty, order_value=value)

    # ---- automation level -------------------------------------------------
    reasons = []
    if not policy.auto_approve_enabled:
        reasons.append("auto_approval_disabled")
    if now - forecast.generated_at > policy.forecast_freshness:
        reasons.append("forecast_stale")                                     # EC-27
    if forecast.quality_flag != "ok":
        reasons.append(f"forecast_{forecast.quality_flag}")                  # EC-19/28
    if forecast.uncertainty > policy.max_forecast_uncertainty:
        reasons.append("forecast_uncertainty_high")                          # EC-34
    if value is None:
        reasons.append("unit_cost_missing")                                  # EC-64
    elif value > policy.auto_approve_max_value:
        reasons.append("order_value_over_limit")                             # EC-33
    if qty > policy.auto_approve_max_qty:
        reasons.append("quantity_over_limit")
    if any(a.startswith("capped_at_") for a in adj):
        reasons.append("cap_applied")
    state = AutomationState.REQUIRES_APPROVAL if reasons else AutomationState.AUTO_APPROVE
    expl["approval_reasons"] = reasons
    text = "Projected stock below reorder point before supplier replenishment arrives"
    return Recommendation(state, text, product.sku, qty, expl, adj)


# ---- approval state machine (TRD 14.3: version check prevents double approval) ----
class ConflictError(Exception):
    """Version/status mismatch - caller must reload and reconfirm (EC-36, EC-61)."""


_ALLOWED = {
    "PENDING": {"approve": "APPROVED", "reject": "REJECTED", "hold": "ON_HOLD"},
    "ON_HOLD": {"approve": "APPROVED", "reject": "REJECTED"},
}


def transition(status: str, version: int, expected_version: int, action: str) -> tuple[str, int]:
    if version != expected_version:
        raise ConflictError(f"recommendation changed (v{version} != expected v{expected_version})")
    nxt = _ALLOWED.get(status, {}).get(action)
    if nxt is None:
        raise ConflictError(f"cannot {action} a recommendation in status {status}")  # double-click safe
    return nxt, version + 1
