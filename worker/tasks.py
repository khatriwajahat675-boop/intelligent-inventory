from datetime import date, datetime, timedelta, timezone

from sqlalchemy import select

from backend.app import db as dbmod
from backend.app.models.entities import ForecastRun, Product, PurchaseOrder, Warehouse
from backend.app.services import alerts, forecasting, recommendations
from worker.celery_app import app


@app.task(name="worker.tasks.run_forecast_task", bind=True, max_retries=3, default_retry_delay=60)
def run_forecast_task(self, horizon: int = 14, scope: str = "all", period: str | None = None,
                      actor: str = "svc:forecast-worker"):
    period = period or date.today().isoformat()
    try:
        with dbmod.session_scope() as db:
            run = forecasting.run(db, horizon=horizon, scope=scope, period=period, actor=actor)
            return {"job_key": run.job_key, "status": run.status}
    except Exception as exc:                        # DB blip -> retry; job_key makes the retry safe
        raise self.retry(exc=exc)


@app.task(name="worker.tasks.evaluate_reorders_task")
def evaluate_reorders_task():
    """Only runs when the latest forecast succeeded (TRD 13.1 step 10)."""
    with dbmod.session_scope() as db:
        last = db.scalar(select(ForecastRun).order_by(ForecastRun.id.desc()))
        if last is None or last.status != "succeeded":
            return {"skipped": "no acceptable forecast"}
        n = 0
        for p in db.scalars(select(Product).where(Product.status == "active")):
            for w in db.scalars(select(Warehouse)):
                recommendations.evaluate_one(db, p, w)
                n += 1
        return {"evaluated": n}


@app.task(name="worker.tasks.po_delay_check_task")
def po_delay_check_task():
    with dbmod.session_scope() as db:
        now = datetime.now(timezone.utc)
        late = db.scalars(select(PurchaseOrder).where(PurchaseOrder.status.in_(("sent", "partially_received")),
                                                       PurchaseOrder.expected_at < now - timedelta(hours=1)))
        for po in late:                              # dedupe_key prevents an alert storm (EC-41)
            alerts.raise_alert(db, "po_delayed", "high", "purchase_order", po.id, f"PO {po.id} is late")
