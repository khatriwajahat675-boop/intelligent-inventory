"""PIPELINE B - alerting and restock automation.

For every (SKU, warehouse) inventory snapshot in MongoDB:
  1. Analyse the purchase/sale pattern - real transaction velocity and trend
     from `sales_transactions` when there is enough history, falling back to
     the product master data's declared `avg_daily_sales_hint` otherwise
     (never fabricated - see `consumption_pattern()`).
  2. Feed that pattern into the SAME pure decision engine the FastAPI backend
     uses (`backend/app/domain/reorder.py`, 32 unit tests) to get a
     recommended reorder quantity, guardrails and an automation state -
     there is exactly one reorder algorithm in this codebase, reused here
     rather than re-implemented against Mongo.
  3. Flag diminishing stock, write a de-duplicated alert, create (or find the
     existing) recommendation, and record an audit event - all idempotent
     and safe to re-run or run concurrently (see docs/AUTOMATION_PIPELINE.md).

    PYTHONPATH=. python pipelines/automation_pipeline.py --once
    PYTHONPATH=. python pipelines/automation_pipeline.py --atlas --loop 900   # every 15 minutes
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from backend.app.domain import reorder as ro
from mongo.repository import Repositories
from mongo.schemas import AlertDoc, RecommendationDoc

log = logging.getLogger("pipelines.automation")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

ROOT = Path(__file__).resolve().parents[1]
MIN_TRANSACTIONS_FOR_TREND = 10
TREND_WINDOW_DAYS = 60
TREND_CHANGE_THRESHOLD = 0.15


def get_repositories(use_atlas: bool) -> Repositories:
    if not use_atlas:
        log.info("using the in-memory MongoDB fake (pass --atlas + set MONGODB_URI for a real cluster)")
        return Repositories.in_memory()
    from mongo.client import get_db
    return Repositories(get_db())


def consumption_pattern(history: list[dict], fallback_hint: float | None, now: datetime) -> dict:
    """Daily consumption rate + trend label from real sales history, with an
    honest fallback: PRD 5.3 - "use a configured safe fallback... do not
    silently fabricate predictions." A hint is used and labelled as such; with
    neither history nor a hint, the rate is 0 and the caller treats it as cold
    start, exactly like the forecasting service's own cold-start rule."""
    n = len(history)
    if n >= MIN_TRANSACTIONS_FOR_TREND:
        span_days = max(1, (history[-1]["sold_at"] - history[0]["sold_at"]).days)
        daily_rate = sum(h["quantity"] for h in history) / span_days
        mid = now - timedelta(days=TREND_WINDOW_DAYS / 2)
        recent = [h for h in history if h["sold_at"] >= mid]
        prior = [h for h in history if h["sold_at"] < mid]
        recent_days = max(1, (history[-1]["sold_at"] - mid).days)
        prior_days = max(1, (mid - history[0]["sold_at"]).days)
        recent_rate = sum(h["quantity"] for h in recent) / recent_days
        prior_rate = sum(h["quantity"] for h in prior) / prior_days if prior else 0.0
        change = (recent_rate - prior_rate) / prior_rate if prior_rate > 0 else 0.0
        trend = ("increasing" if change > TREND_CHANGE_THRESHOLD else
                 "decreasing" if change < -TREND_CHANGE_THRESHOLD else "stable")
        return {"daily_rate": daily_rate, "trend": trend, "source": "real_transaction_history",
               "n_transactions": n, "span_days": span_days, "recent_daily_rate": round(recent_rate, 3),
               "prior_daily_rate": round(prior_rate, 3), "pct_change": round(change, 3), "quality_flag": "ok"}
    if fallback_hint is not None:
        return {"daily_rate": fallback_hint, "trend": "unknown", "source": "master_data_hint",
               "n_transactions": n, "quality_flag": "low_confidence"}
    return {"daily_rate": 0.0, "trend": "unknown", "source": "no_data", "n_transactions": n,
           "quality_flag": "low_confidence"}


def evaluate_one(repos: Repositories, snapshot: dict, now: datetime, policy: ro.Policy,
                 actor: str = "svc:automation-pipeline") -> dict:
    sku, warehouse = snapshot["sku"], snapshot["warehouse"]
    product = repos.products.get(sku)
    if product is None or product["status"] != "active":
        return {"sku": sku, "warehouse": warehouse, "skipped": "no active product master data"}

    supplier_doc = repos.suppliers.get(product["supplier_id"]) if product.get("supplier_id") else None
    supplier = (ro.Supplier(supplier_doc["supplier_id"], supplier_doc["status"], product["lead_time_days"])
               if supplier_doc else None)
    stock = ro.Stock(snapshot["on_hand"], snapshot.get("reserved", 0), snapshot.get("damaged", 0),
                     eligible_inbound=0.0)   # this pipeline evaluates alerting/restock; PO inbound tracking
                                             # lives in the FastAPI backend's purchase_orders collection/table

    history = repos.sales.recent(sku, warehouse, days=TREND_WINDOW_DAYS * 2, now=now)
    pattern = consumption_pattern(history, product.get("avg_daily_sales_hint"), now)
    horizon_days = max(1, int(round(product["lead_time_days"])) + 7)      # lead time + a week of buffer visibility
    forecast = (ro.Forecast(run_id="automation_pipeline_velocity", model_version="consumption_pattern_v1",
                           daily=tuple([pattern["daily_rate"]] * horizon_days), generated_at=now,
                           quality_flag=pattern["quality_flag"], uncertainty=0.15)
               if pattern["daily_rate"] > 0 or pattern["source"] != "no_data" else None)

    dom_product = ro.Product(sku, product["status"], product["moq"], product["pack_size"],
                             product["safety_stock"], product.get("unit_cost"))
    has_open = bool(repos.recommendations.open_for(sku, warehouse))
    decision = ro.evaluate(dom_product, supplier, stock, forecast, policy, now, has_open)

    result = {"sku": sku, "warehouse": warehouse, "pattern": pattern, "decision_state": decision.state.value,
             "reason": decision.reason, "recommended_qty": decision.recommended_qty}

    available = stock.available
    if available <= dom_product.safety_stock:
        severity = "critical" if available <= 0 else "high" if available < dom_product.safety_stock * 0.5 else "medium"
        created = repos.alerts.raise_if_absent(AlertDoc(
            type="diminishing_stock", severity=severity, entity_type="product", entity_id=sku,
            dedupe_key=f"diminishing:{sku}:{warehouse}",
            message=f"{sku}@{warehouse}: available {available:.0f} <= safety stock {dom_product.safety_stock:.0f} "
                    f"(consumption {pattern['daily_rate']:.2f}/day, trend {pattern['trend']}, source {pattern['source']})",
            created_at=now))
        if created:
            repos.audit.record(actor, "alert.diminishing_stock", "product", sku, after={"available": available})
        result["diminishing_stock_alert_raised"] = created

    if decision.state in (ro.AutomationState.BLOCKED,) or (
            decision.state == ro.AutomationState.REQUIRES_APPROVAL and decision.recommended_qty == 0):
        repos.alerts.raise_if_absent(AlertDoc(type="automation_blocked", severity="high", entity_type="product",
                                              entity_id=sku, dedupe_key=f"blocked:{sku}:{warehouse}:{decision.reason}",
                                              message=f"{sku}@{warehouse}: {decision.reason}", created_at=now))
    elif decision.recommended_qty > 0:
        key = ro.idempotency_key(sku, warehouse, now.date().isoformat())
        created, row = repos.recommendations.create_if_absent(RecommendationDoc(
            idempotency_key=key, sku=sku, warehouse=warehouse, recommended_qty=decision.recommended_qty,
            automation_state=decision.state.value, reason=decision.reason,
            explanation={**decision.explanation, "consumption_pattern": pattern,
                        "constraint_adjustments": decision.constraint_adjustments}, created_at=now))
        result["recommendation_created"] = created
        result["recommendation_id"] = row.get("_id")
        if created:
            repos.audit.record(actor, "recommendation.create", "product", sku,
                               after={"qty": decision.recommended_qty, "state": decision.state.value})
            repos.alerts.raise_if_absent(AlertDoc(
                type="restock_needed", severity="critical" if available <= 0 else "high", entity_type="product",
                entity_id=sku, dedupe_key=f"restock:{sku}:{warehouse}",
                message=f"{sku}@{warehouse}: reorder {decision.recommended_qty} units ({decision.reason})",
                created_at=now))
    return result


def run_once(repos: Repositories, policy: ro.Policy | None = None, now: datetime | None = None) -> dict:
    from collections import Counter
    now = now or datetime.now(timezone.utc)
    policy = policy or ro.Policy()
    results = [evaluate_one(repos, snap, now, policy) for snap in repos.inventory.list_all()]
    summary = {
        "run_at": now.isoformat(), "skus_evaluated": len(results),
        "skipped_no_master_data": sum(1 for r in results if "skipped" in r),
        "diminishing_stock_alerts": sum(1 for r in results if r.get("diminishing_stock_alert_raised")),
        "recommendations_created": sum(1 for r in results if r.get("recommendation_created")),
        "blocked": sum(1 for r in results if r.get("decision_state") == "BLOCKED"),
        "open_alerts_by_severity": dict(Counter(a["severity"] for a in repos.alerts.open_by_severity())),
    }
    out = ROOT / "reports/pipelines"
    out.mkdir(parents=True, exist_ok=True)
    (out / "automation_pipeline_last_run.json").write_text(json.dumps({"summary": summary, "results": results},
                                                                       indent=2, default=str))
    log.info("automation pass complete: %s", summary)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas", action="store_true", help="use a real MongoDB Atlas cluster (needs MONGODB_URI)")
    ap.add_argument("--loop", type=int, default=None, metavar="SECONDS",
                    help="run continuously, sleeping SECONDS between passes (omit for a single pass)")
    ap.add_argument("--once", action="store_true", help="explicit single pass (default when --loop is omitted)")
    args = ap.parse_args()
    repos = get_repositories(args.atlas)

    if args.loop:
        log.info("starting automation loop, interval=%ss (Ctrl+C to stop)", args.loop)
        while True:
            try:
                run_once(repos)
            except Exception:
                log.exception("automation pass failed - will retry next interval rather than crash the loop")
            time.sleep(args.loop)
    else:
        print(json.dumps(run_once(repos), indent=2, default=str))


if __name__ == "__main__":
    main()
