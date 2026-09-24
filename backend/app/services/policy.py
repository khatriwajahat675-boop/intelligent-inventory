from datetime import timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.domain.reorder import Policy
from backend.app.models.entities import PolicyConfig

_FIELDS = {"auto_approve_enabled": bool, "auto_approve_max_qty": int, "auto_approve_max_value": float,
           "max_order_qty": int, "max_forecast_uncertainty": float, "forecast_freshness_hours": float}


def validate_policy_values(v: dict) -> dict:
    """FR-020 / EC-59: invalid configuration is rejected before activation."""
    unknown = set(v) - set(_FIELDS)
    if unknown:
        raise ValueError(f"unknown policy fields: {sorted(unknown)}")
    out = {k: _FIELDS[k](val) for k, val in v.items()}
    for k in ("auto_approve_max_qty", "auto_approve_max_value", "max_order_qty", "forecast_freshness_hours"):
        if k in out and out[k] <= 0:
            raise ValueError(f"{k} must be > 0")
    if not 0 <= out.get("max_forecast_uncertainty", 0.0) <= 5:
        raise ValueError("max_forecast_uncertainty must be within [0, 5]")
    return out


def active_policy(db: Session) -> Policy:
    row = db.scalar(select(PolicyConfig).where(PolicyConfig.active == 1).order_by(PolicyConfig.id.desc()))
    if row is None:
        return Policy()
    v = dict(row.values_json)
    hours = v.pop("forecast_freshness_hours", None)
    return Policy(version=row.version, **v,
                  **({"forecast_freshness": timedelta(hours=hours)} if hours else {}))
