"""Forecast execution (FR-007/008, TRD 13). One product/warehouse failing never fails the batch (EC-23)."""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.app.models.entities import (Forecast, ForecastRun, InventoryBalance, Product, SalesTransaction)
from backend.app.services import alerts, audit
from ml.evaluation import selection as sel
from ml.models import baselines as bl

SERVICE = "svc:forecast-worker"


def build_series(db: Session, product_id: int, warehouse_id: int) -> pd.Series:
    rows = db.execute(
        select(SalesTransaction.business_date, func.sum(SalesTransaction.quantity))
        .where(SalesTransaction.product_id == product_id, SalesTransaction.warehouse_id == warehouse_id)
        .group_by(SalesTransaction.business_date)).all()
    if not rows:
        return pd.Series(dtype=float)
    s = pd.Series({pd.Timestamp(d): float(q) for d, q in rows}).sort_index()
    return sel.reindex_daily(s)                                          # EC-20: explicit missing days


def forecast_one(y: pd.Series, horizon: int) -> dict:
    """Select on a chronological validation tail, refit on everything, predict, clamp."""
    if len(y) < sel.MIN_HISTORY_DAYS:
        s = sel.Selection("moving_average", "cold_start", quality_flag="low_confidence")
    else:
        h = max(7, min(28, len(y) // 6))
        s = sel.select_on_validation(y.iloc[:-h], y.iloc[-h:])
    try:
        raw = bl.build(s.model).fit(y).predict(horizon)
    except Exception:                                                    # EC-22: fall back, never fail the SKU
        s = sel.Selection("moving_average", "baseline", quality_flag="fallback")
        raw = bl.build("moving_average").fit(y).predict(horizon)
    resid = float(y.tail(28).std() or 0.0)
    return {"selection": s, "raw": raw, "clamped": sel.clamp_non_negative(raw), "resid_std": resid}


def run(db: Session, *, horizon: int = 14, scope: str = "all", period: str | None = None,
        actor: str = SERVICE) -> ForecastRun:
    period = period or date.today().isoformat()
    job_key = f"forecast:{scope}:{period}"
    existing = db.scalar(select(ForecastRun).where(ForecastRun.job_key == job_key))
    if existing is not None:
        return existing                                                  # EC-51: restart/double-run is a no-op
    run_ = ForecastRun(job_key=job_key, model_type="auto", model_version=f"auto_{period}", horizon_days=horizon)
    db.add(run_)
    try:
        db.flush()
    except IntegrityError:                                               # lost a race with another worker
        db.rollback()
        return db.scalar(select(ForecastRun).where(ForecastRun.job_key == job_key))
    ok = fail = 0
    stats: dict[str, int] = {}
    for prod, bal in db.execute(select(Product, InventoryBalance).join(
            InventoryBalance, InventoryBalance.product_id == Product.id).where(Product.status == "active")):
        y = build_series(db, prod.id, bal.warehouse_id)
        if y.empty:
            alerts.raise_alert(db, "forecast_no_history", "medium", "product", prod.id,
                               f"{prod.sku}: no sales history - manual policy required")     # EC-66
            fail += 1
            continue
        try:
            res = forecast_one(y, horizon)
        except Exception as exc:
            alerts.raise_alert(db, "forecast_failed", "high", "product", prod.id, f"{prod.sku}: {exc}")
            fail += 1
            continue
        start = y.index[-1] + timedelta(days=1)
        flag = res["selection"].quality_flag
        stats[res["selection"].model] = stats.get(res["selection"].model, 0) + 1
        for i in range(horizon):
            yh, band = float(res["clamped"][i]), 1.28 * res["resid_std"]
            db.add(Forecast(run_id=run_.id, product_id=prod.id, warehouse_id=bal.warehouse_id,
                            forecast_date=(start + timedelta(days=i)).date().isoformat(), yhat=yh,
                            yhat_raw=float(res["raw"][i]), lower=max(0.0, yh - band), upper=yh + band,
                            quality_flag=flag))
        ok += 1
    run_.status = "succeeded" if ok else "failed"
    run_.finished_at = datetime.now(timezone.utc)
    run_.metrics_json = {"skus_ok": ok, "skus_failed": fail, "models": stats,
                         "coverage": ok / max(ok + fail, 1)}
    if not ok:
        alerts.raise_alert(db, "forecast_run_failed", "high", "forecast_run", run_.id, "forecast job produced no forecasts")
    audit.record(db, actor, "forecast.run", "forecast_run", run_.id, after=run_.metrics_json,
                 model_version=run_.model_version)
    return run_
