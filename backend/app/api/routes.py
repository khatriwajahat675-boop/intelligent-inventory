"""REST API - /api/v1 (TRD 12). Every endpoint enforces RBAC server-side."""
from __future__ import annotations

import csv
import io
import uuid
from datetime import date, datetime, timezone

from fastapi import APIRouter, BackgroundTasks, Depends, File, Header, HTTPException, Query, Request, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.app import db as dbmod
from backend.app.api.deps import current_user, page, require
from backend.app.config import Settings, get_settings
from backend.app.db import get_db
from backend.app.domain import inventory as invdom
from backend.app.domain import reorder as ro
from backend.app.models.entities import (Alert, AuditEvent, Forecast, ForecastRun, InventoryBalance,
                                         InventoryMovement, PolicyConfig, Product, ProductSupplier,
                                         PurchaseOrder, PurchaseOrderLine, Recommendation, Supplier, User, Warehouse)
from backend.app.schemas import api as S
from backend.app.security import passwords, tokens
from backend.app.security.ratelimit import RateLimiter
from backend.app.services import audit, forecasting, policy as policysvc, recommendations as reco, sales
from backend.app.services import purchase_orders as posvc

router = APIRouter(prefix="/api/v1")
_login_limiter: RateLimiter | None = None


def _rid(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


def _by_sku(db: Session, sku: str) -> Product:
    p = db.scalar(select(Product).where(Product.sku == sku))
    if p is None:
        raise HTTPException(404, "unknown sku")
    return p


def _wh(db: Session, code: str) -> Warehouse:
    w = db.scalar(select(Warehouse).where(Warehouse.code == code))
    if w is None:
        raise HTTPException(404, "unknown warehouse")
    return w


def _csv_safe(v) -> str:
    """Neutralise spreadsheet formula injection in exports (EC-47, TRD 15.1)."""
    s = "" if v is None else str(v)
    return "'" + s if s[:1] in ("=", "+", "-", "@", "\t", "\r") else s


# ------------------------------------------------------------------ auth ---
@router.post("/auth/login", response_model=S.TokenOut)
def login(body: S.LoginIn, request: Request, db: Session = Depends(get_db), settings: Settings = Depends(get_settings)):
    global _login_limiter
    _login_limiter = _login_limiter or RateLimiter(settings.login_rate_limit_per_minute)
    key = f"{request.client.host if request.client else '?'}:{body.email.lower()}"
    if not _login_limiter.allow(key):
        raise HTTPException(429, "too many attempts")                         # credential stuffing defence
    user = db.scalar(select(User).where(User.email == body.email.lower()))
    ok = user is not None and user.status == "active" and passwords.verify_password(body.password, user.password_hash)
    if not ok:
        audit.record(db, body.email.lower(), "auth.login_failed", "user", "-", request_id=_rid(request))
        db.commit()
        raise HTTPException(401, "invalid credentials")                       # same message for unknown/disabled/wrong
    user.session_id = uuid.uuid4().hex                                       # fresh id per login (fixation-safe)
    audit.record(db, user.email, "auth.login", "user", user.id, request_id=_rid(request))
    return S.TokenOut(access_token=tokens.create_token(settings.jwt_secret, user.id, user.role, user.session_id,
                                                       settings.access_token_minutes, settings.jwt_algorithm),
                      role=user.role)


@router.post("/auth/logout", status_code=204)
def logout(user: User = Depends(current_user), db: Session = Depends(get_db)):
    user.session_id = ""                                                     # revokes every token immediately
    audit.record(db, user.email, "auth.logout", "user", user.id)


@router.get("/auth/me")
def me(user: User = Depends(current_user)):
    return {"id": user.id, "email": user.email, "role": user.role}


# -------------------------------------------------------------- catalog ---
@router.get("/products", response_model=list[S.ProductOut])
def list_products(q: str | None = None, status: str | None = None, pg=Depends(page), db: Session = Depends(get_db),
                  _: User = Depends(require("products:read"))):
    stmt = select(Product).order_by(Product.id).limit(pg[0]).offset(pg[1])
    if q:
        stmt = stmt.where(Product.sku.ilike(f"%{q}%") | Product.name.ilike(f"%{q}%"))   # parameterised (EC-47)
    if status:
        stmt = stmt.where(Product.status == status)
    return db.scalars(stmt).all()


@router.post("/products", response_model=S.ProductOut, status_code=201)
def create_product(body: S.ProductIn, request: Request, db: Session = Depends(get_db),
                   user: User = Depends(require("products:write"))):
    p = Product(**body.model_dump())
    db.add(p)
    try:
        db.flush()
    except IntegrityError:
        raise HTTPException(409, "sku already exists")
    audit.record(db, user.email, "product.create", "product", p.id, after=body.model_dump(), request_id=_rid(request))
    return p


@router.patch("/products/{product_id}", response_model=S.ProductOut)
def patch_product(product_id: int, body: S.ProductPatch, request: Request, db: Session = Depends(get_db),
                  user: User = Depends(require("products:write"))):
    p = db.get(Product, product_id)
    if p is None:
        raise HTTPException(404, "not found")
    if p.version != body.expected_version:                                   # EC-61: no silent overwrite
        raise HTTPException(409, "product was modified by someone else; reload and retry")
    changes = body.model_dump(exclude_unset=True, exclude={"expected_version"})
    before = {k: getattr(p, k) for k in changes}
    for k, v in changes.items():
        setattr(p, k, v)
    p.version += 1
    audit.record(db, user.email, "product.update", "product", p.id, before=before, after=changes, request_id=_rid(request))
    if changes.get("status") == "archived":                                  # EC-14: cancel pending automation
        for r in db.scalars(select(Recommendation).where(Recommendation.product_id == p.id,
                                                          Recommendation.status.in_(("PENDING", "ON_HOLD")))):
            r.status, r.version = "CANCELLED", r.version + 1
            audit.record(db, user.email, "recommendation.cancel", "recommendation", r.id, reason="sku archived")
    return p


@router.get("/suppliers")
def list_suppliers(pg=Depends(page), db: Session = Depends(get_db), _: User = Depends(require("suppliers:read"))):
    return [{"id": s.id, "code": s.code, "name": s.name, "status": s.status,
             "default_lead_time_days": s.default_lead_time_days}
            for s in db.scalars(select(Supplier).order_by(Supplier.id).limit(pg[0]).offset(pg[1]))]


@router.post("/suppliers", status_code=201)
def create_supplier(body: S.SupplierIn, request: Request, db: Session = Depends(get_db),
                    user: User = Depends(require("suppliers:write"))):
    s = Supplier(**body.model_dump())
    db.add(s)
    try:
        db.flush()
    except IntegrityError:
        raise HTTPException(409, "supplier code exists")
    audit.record(db, user.email, "supplier.create", "supplier", s.id, after=body.model_dump(), request_id=_rid(request))
    return {"id": s.id}


@router.put("/products/{product_id}/suppliers")
def link_supplier(product_id: int, body: S.ProductSupplierIn, request: Request, db: Session = Depends(get_db),
                  user: User = Depends(require("suppliers:write"))):
    if db.get(Product, product_id) is None or db.get(Supplier, body.supplier_id) is None:
        raise HTTPException(404, "product or supplier not found")
    link = db.get(ProductSupplier, (product_id, body.supplier_id))
    before = None
    if link:
        before = {"lead_time_days": link.lead_time_days, "moq": link.moq, "pack_size": link.pack_size}
        link.lead_time_days, link.moq, link.pack_size = body.lead_time_days, body.moq, body.pack_size
        link.unit_cost, link.preferred = body.unit_cost, int(body.preferred)
    else:
        db.add(ProductSupplier(product_id=product_id, supplier_id=body.supplier_id, lead_time_days=body.lead_time_days,
                               moq=body.moq, pack_size=body.pack_size, unit_cost=body.unit_cost,
                               preferred=int(body.preferred)))
    audit.record(db, user.email, "product_supplier.upsert", "product", product_id, before=before,   # EC-15: history kept
                 after=body.model_dump(), request_id=_rid(request))
    return {"ok": True}


@router.get("/warehouses")
def list_warehouses(db: Session = Depends(get_db), _: User = Depends(require("inventory:read"))):
    return [{"id": w.id, "code": w.code, "name": w.name, "timezone": w.timezone} for w in db.scalars(select(Warehouse))]


@router.post("/warehouses", status_code=201)
def create_warehouse(body: S.WarehouseIn, request: Request, db: Session = Depends(get_db),
                     user: User = Depends(require("products:write"))):
    w = Warehouse(**body.model_dump())
    db.add(w)
    try:
        db.flush()
    except IntegrityError:
        raise HTTPException(409, "warehouse code exists")
    audit.record(db, user.email, "warehouse.create", "warehouse", w.id, after=body.model_dump(), request_id=_rid(request))
    return {"id": w.id}


# ---------------------------------------------------------------- sales ---
@router.post("/sales", status_code=201)
def create_sale(body: S.SaleIn, request: Request, idempotency_key: str | None = Header(None),
                db: Session = Depends(get_db), user: User = Depends(require("sales:write")),
                settings: Settings = Depends(get_settings)):
    row = body.model_dump()
    row["sold_at"] = body.sold_at.isoformat()
    if idempotency_key:
        row["external_txn_id"] = idempotency_key
    res = sales.ingest(db, [row], source="api", actor=user.email, business_tz=settings.business_timezone, update_stock=True)
    if res["rejected"]:
        raise HTTPException(422, res["errors"][0]["error"])
    return res


@router.post("/sales/import", status_code=202)
async def import_sales(request: Request, file: UploadFile = File(...), update_stock: bool = False,
                       db: Session = Depends(get_db), user: User = Depends(require("sales:import")),
                       settings: Settings = Depends(get_settings)):
    if not (file.filename or "").lower().endswith(".csv"):
        raise HTTPException(415, "only .csv accepted")
    raw = await file.read(settings.max_import_bytes + 1)
    try:
        rows = sales.parse_csv(raw, settings.max_import_bytes, settings.max_import_rows)
    except sales.ImportTooLarge as exc:
        raise HTTPException(413, str(exc))
    except (ValueError, UnicodeDecodeError) as exc:
        raise HTTPException(422, str(exc))
    return sales.ingest(db, rows, source="csv", actor=user.email, business_tz=settings.business_timezone,
                        update_stock=update_stock)


# ------------------------------------------------------------ inventory ---
@router.get("/inventory/{warehouse}/{sku}")
def stock_position(warehouse: str, sku: str, db: Session = Depends(get_db), _: User = Depends(require("inventory:read"))):
    p, w = _by_sku(db, sku), _wh(db, warehouse)
    b = db.get(InventoryBalance, (p.id, w.id))
    on_hand, reserved, damaged = (b.on_hand, b.reserved, b.damaged) if b else (0, 0, 0)
    inbound = reco.eligible_inbound(db, p.id, w.id)
    avail = max(0, on_hand - reserved - damaged)
    return {"sku": sku, "warehouse": warehouse, "on_hand": on_hand, "reserved": reserved, "damaged": damaged,
            "available": avail, "inbound": inbound, "inventory_position": avail + inbound}


@router.post("/inventory/adjustments")
def adjust(body: S.AdjustmentIn, request: Request, db: Session = Depends(get_db),
           user: User = Depends(require("inventory:adjust"))):
    if body.privileged and user.role not in ("admin", "inventory_manager"):
        raise HTTPException(403, "privileged adjustment not permitted")
    p, w = _by_sku(db, body.sku), _wh(db, body.warehouse)
    b = db.get(InventoryBalance, (p.id, w.id))
    if b is None:
        b = InventoryBalance(product_id=p.id, warehouse_id=w.id, on_hand=0)
        db.add(b)
    try:
        new = invdom.apply_adjustment(invdom.Balance(b.on_hand, b.reserved, b.damaged), body.delta, body.reason, body.privileged)
    except invdom.InventoryError as exc:
        raise HTTPException(422, str(exc))                                   # EC-11
    before, b.on_hand = {"on_hand": b.on_hand}, new.on_hand
    db.add(InventoryMovement(product_id=p.id, warehouse_id=w.id, movement_type="adjustment", quantity=body.delta,
                             reference_type="adjustment", reference_id=body.reason[:60]))
    audit.record(db, user.email, "inventory.adjust", "inventory", f"{p.id}:{w.id}", before=before,
                 after={"on_hand": new.on_hand}, reason=body.reason, request_id=_rid(request))
    return {"on_hand": new.on_hand}


# ------------------------------------------------------------ forecasts ---
def _forecast_job(horizon: int, scope: str, period: str, actor: str) -> None:
    """Runs outside the request, in its own session/transaction."""
    with dbmod.session_scope() as db:
        forecasting.run(db, horizon=horizon, scope=scope, period=period, actor=actor)


@router.post("/forecasts/run", status_code=202)
def run_forecast(body: S.ForecastRunIn, tasks: BackgroundTasks, db: Session = Depends(get_db),
                 user: User = Depends(require("forecasts:run")), settings: Settings = Depends(get_settings)):
    period = date.today().isoformat()
    job_key = f"forecast:{body.scope}:{period}"
    if db.scalar(select(ForecastRun).where(ForecastRun.job_key == job_key)):
        return {"job_key": job_key, "status": "already_exists"}              # EC-51 idempotent
    if settings.use_celery:
        from worker.tasks import run_forecast_task
        run_forecast_task.delay(body.horizon_days, body.scope, period, user.email)
    else:
        tasks.add_task(_forecast_job, body.horizon_days, body.scope, period, user.email)
    return {"job_key": job_key, "status": "queued"}                          # API returns quickly (TRD 16.3)


@router.get("/forecasts/runs/{job_key}")
def forecast_run_status(job_key: str, db: Session = Depends(get_db), _: User = Depends(require("forecasts:read"))):
    r = db.scalar(select(ForecastRun).where(ForecastRun.job_key == job_key))
    if r is None:
        raise HTTPException(404, "no such run")
    return {"id": r.id, "status": r.status, "model_version": r.model_version, "metrics": r.metrics_json,
            "started_at": r.started_at, "finished_at": r.finished_at}


@router.get("/forecasts/{sku}")
def get_forecast(sku: str, warehouse: str, db: Session = Depends(get_db), _: User = Depends(require("forecasts:read"))):
    p, w = _by_sku(db, sku), _wh(db, warehouse)
    run_ = db.scalar(select(ForecastRun).where(ForecastRun.status == "succeeded").order_by(ForecastRun.id.desc()))
    if run_ is None:
        raise HTTPException(404, "no forecast available")
    rows = db.scalars(select(Forecast).where(Forecast.run_id == run_.id, Forecast.product_id == p.id,
                                             Forecast.warehouse_id == w.id).order_by(Forecast.forecast_date)).all()
    ts = run_.finished_at or run_.started_at
    return {"sku": sku, "warehouse": warehouse, "run_id": run_.id, "model_version": run_.model_version,
            "as_of": ts, "quality_flag": rows[0].quality_flag if rows else "missing",
            "points": [{"date": r.forecast_date, "yhat": r.yhat, "lower": r.lower, "upper": r.upper} for r in rows]}


# ------------------------------------------------------ recommendations ---
@router.post("/recommendations/evaluate")
def evaluate(body: S.EvaluateIn, request: Request, db: Session = Depends(get_db),
             user: User = Depends(require("recommendations:evaluate"))):
    prods = [_by_sku(db, body.sku)] if body.sku else list(db.scalars(select(Product).where(Product.status == "active")))
    whs = [_wh(db, body.warehouse)] if body.warehouse else list(db.scalars(select(Warehouse)))
    out = {"created": 0, "suppressed": 0, "blocked": 0}
    for p in prods:
        for w in whs:
            result, row = reco.evaluate_one(db, p, w, actor=user.email, request_id=_rid(request))
            key = {"SUPPRESSED": "suppressed", "BLOCKED": "blocked"}.get(result.state.value, "created")
            out[key] += 1
    return out


def _reco_out(r: Recommendation, p: Product, w: Warehouse) -> dict:
    return {"recommendation_id": r.id, "sku": p.sku, "warehouse": w.code, "recommended_qty": r.recommended_qty,
            "automation_state": r.automation_state, "status": r.status, "version": r.version,
            "policy_version": r.policy_version, "forecast_run_id": r.forecast_run_id, **r.explanation_json}


@router.get("/recommendations")
def list_recos(status: str | None = None, pg=Depends(page), db: Session = Depends(get_db),
               _: User = Depends(require("recommendations:read"))):
    stmt = (select(Recommendation, Product, Warehouse).join(Product, Product.id == Recommendation.product_id)
            .join(Warehouse, Warehouse.id == Recommendation.warehouse_id)
            .order_by(Recommendation.id.desc()).limit(pg[0]).offset(pg[1]))
    if status:
        stmt = stmt.where(Recommendation.status == status)
    return [_reco_out(*row) for row in db.execute(stmt)]


@router.get("/recommendations/export.csv")
def export_recos(status: str | None = None, db: Session = Depends(get_db), user: User = Depends(require("export"))):
    stmt = (select(Recommendation, Product, Warehouse).join(Product, Product.id == Recommendation.product_id)
            .join(Warehouse, Warehouse.id == Recommendation.warehouse_id).order_by(Recommendation.id).limit(10_000))
    if status:
        stmt = stmt.where(Recommendation.status == status)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["id", "sku", "warehouse", "qty", "state", "status", "reason"])
    for r, p, wh in db.execute(stmt):
        w.writerow([_csv_safe(x) for x in (r.id, p.sku, wh.code, r.recommended_qty, r.automation_state, r.status,
                                           r.explanation_json.get("reason"))])
    buf.seek(0)
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv")   # capped at 10k rows (EC-56)


def _decide(action: str, rid: int, body: S.DecisionIn, request: Request, db: Session, user: User):
    r = db.get(Recommendation, rid)
    if r is None:
        raise HTTPException(404, "not found")
    try:
        reco.decide(db, r, action, body.expected_version, actor=user.email, reason=body.reason, request_id=_rid(request))
    except ro.ConflictError as exc:
        raise HTTPException(409, str(exc))
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    return {"id": r.id, "status": r.status, "version": r.version}


@router.post("/recommendations/{rid}/approve")
def approve(rid: int, body: S.DecisionIn, request: Request, db: Session = Depends(get_db),
            user: User = Depends(require("recommendations:decide"))):
    return _decide("approve", rid, body, request, db, user)


@router.post("/recommendations/{rid}/reject")
def reject(rid: int, body: S.DecisionIn, request: Request, db: Session = Depends(get_db),
           user: User = Depends(require("recommendations:decide"))):
    return _decide("reject", rid, body, request, db, user)


@router.post("/recommendations/{rid}/hold")
def hold(rid: int, body: S.DecisionIn, request: Request, db: Session = Depends(get_db),
         user: User = Depends(require("recommendations:decide"))):
    return _decide("hold", rid, body, request, db, user)


# ------------------------------------------------------ purchase orders ---
def _po_out(po: PurchaseOrder) -> dict:
    return {"id": po.id, "status": po.status, "supplier_id": po.supplier_id, "warehouse_id": po.warehouse_id,
            "total_value": po.total_value, "lines": [{"product_id": l.product_id, "qty_ordered": l.qty_ordered,
                                                       "qty_received": l.qty_received} for l in po.lines]}


@router.get("/purchase-orders")
def list_pos(status: str | None = None, pg=Depends(page), db: Session = Depends(get_db),
             _: User = Depends(require("purchase_orders:read"))):
    stmt = select(PurchaseOrder).order_by(PurchaseOrder.id.desc()).limit(pg[0]).offset(pg[1])
    if status:
        stmt = stmt.where(PurchaseOrder.status == status)
    return [_po_out(p) for p in db.scalars(stmt)]


@router.post("/purchase-orders", status_code=201)
def create_po(body: S.POCreateIn, request: Request, db: Session = Depends(get_db),
              user: User = Depends(require("purchase_orders:write"))):
    sup, wh = db.get(Supplier, body.supplier_id), _wh(db, body.warehouse)
    if sup is None or sup.status != "active":
        raise HTTPException(422, "supplier missing or inactive")
    lines = []
    for ln in body.lines:
        p = _by_sku(db, str(ln.get("sku")))
        qty = int(ln.get("quantity", 0))
        if qty <= 0 or p.status != "active":
            raise HTTPException(422, f"invalid line for {p.sku}")
        lines.append(PurchaseOrderLine(product_id=p.id, qty_ordered=qty))
    po = PurchaseOrder(supplier_id=sup.id, warehouse_id=wh.id, lines=lines, created_by=user.email)
    db.add(po)
    db.flush()
    audit.record(db, user.email, "purchase_order.create", "purchase_order", po.id, after={"manual": True},
                 request_id=_rid(request))
    return _po_out(po)


@router.post("/purchase-orders/{po_id}/status")
def po_status(po_id: int, body: S.POStatusIn, db: Session = Depends(get_db),
              user: User = Depends(require("purchase_orders:write"))):
    po = db.get(PurchaseOrder, po_id)
    if po is None:
        raise HTTPException(404, "not found")
    try:
        reco.po_status_change(db, po, body.status, user.email)
    except invdom.InventoryError as exc:
        raise HTTPException(409, str(exc))
    return _po_out(po)


@router.post("/purchase-orders/{po_id}/receive")
def po_receive(po_id: int, body: S.ReceiveIn, db: Session = Depends(get_db),
               user: User = Depends(require("purchase_orders:receive"))):
    po = db.get(PurchaseOrder, po_id)
    if po is None:
        raise HTTPException(404, "not found")
    if body.over_receipt_authorized and user.role not in ("admin", "inventory_manager"):
        raise HTTPException(403, "over-receipt requires manager authorization")     # EC-09
    try:
        posvc.receive(db, po, _by_sku(db, body.sku).id, body.quantity, actor=user.email,
                      over_receipt_authorized=body.over_receipt_authorized)
    except invdom.InventoryError as exc:
        raise HTTPException(422, str(exc))
    return _po_out(po)


# --------------------------------------------- alerts, audit, dashboard ---
@router.get("/alerts")
def list_alerts(status: str = "open", pg=Depends(page), db: Session = Depends(get_db),
                _: User = Depends(require("alerts:read"))):
    order = {"critical": 0, "high": 1, "medium": 2, "info": 3}
    rows = db.scalars(select(Alert).where(Alert.status == status).order_by(Alert.id.desc()).limit(pg[0]).offset(pg[1])).all()
    return sorted(({"id": a.id, "type": a.type, "severity": a.severity, "message": a.message,
                    "created_at": a.created_at} for a in rows), key=lambda a: order.get(a["severity"], 9))


@router.get("/audit")
def list_audit(action: str | None = None, entity_type: str | None = None, pg=Depends(page),
               db: Session = Depends(get_db), _: User = Depends(require("audit:read"))):
    stmt = select(AuditEvent).order_by(AuditEvent.id.desc()).limit(pg[0]).offset(pg[1])
    if action:
        stmt = stmt.where(AuditEvent.action == action)
    if entity_type:
        stmt = stmt.where(AuditEvent.entity_type == entity_type)
    return [{"id": e.id, "actor": e.actor, "action": e.action, "entity_type": e.entity_type, "entity_id": e.entity_id,
             "before": e.before_json, "after": e.after_json, "reason": e.reason, "policy_version": e.policy_version,
             "model_version": e.model_version, "occurred_at": e.occurred_at} for e in db.scalars(stmt)]


@router.get("/dashboard")
def dashboard(db: Session = Depends(get_db), _: User = Depends(require("inventory:read"))):
    """FR-017: KPIs with an explicit as-of time so stale data is visible (EC-57)."""
    last = db.scalar(select(ForecastRun).where(ForecastRun.status == "succeeded").order_by(ForecastRun.id.desc()))
    last_any = db.scalar(select(ForecastRun).order_by(ForecastRun.id.desc()))
    at_risk = db.scalar(select(func.count()).select_from(InventoryBalance).join(
        Product, Product.id == InventoryBalance.product_id).where(
        (InventoryBalance.on_hand - InventoryBalance.reserved - InventoryBalance.damaged) <= Product.safety_stock))
    return {
        "as_of": datetime.now(timezone.utc),
        "skus_active": db.scalar(select(func.count()).select_from(Product).where(Product.status == "active")),
        "skus_below_safety_stock": at_risk,
        "pending_recommendations": db.scalar(select(func.count()).select_from(Recommendation).where(Recommendation.status == "PENDING")),
        "open_purchase_orders": db.scalar(select(func.count()).select_from(PurchaseOrder).where(
            PurchaseOrder.status.in_(("approved", "sent", "partially_received")))),
        "open_alerts": db.scalar(select(func.count()).select_from(Alert).where(Alert.status == "open")),
        "last_successful_forecast": (last.finished_at if last else None),
        "forecast_status": ("failed_last_run" if last_any and last_any.status == "failed" else
                            "ok" if last else "no_forecast_yet"),
        "forecast_metrics": last.metrics_json if last else None,
    }


# --------------------------------------------------------------- policy ---
@router.get("/config/policy")
def get_policy(db: Session = Depends(get_db), _: User = Depends(require("config:read"))):
    p = policysvc.active_policy(db)
    return {"version": p.version, "auto_approve_enabled": p.auto_approve_enabled,
            "auto_approve_max_qty": p.auto_approve_max_qty, "auto_approve_max_value": p.auto_approve_max_value,
            "max_order_qty": p.max_order_qty, "max_forecast_uncertainty": p.max_forecast_uncertainty,
            "forecast_freshness_hours": p.forecast_freshness.total_seconds() / 3600}


@router.put("/config/policy")
def put_policy(body: S.PolicyIn, request: Request, db: Session = Depends(get_db),
               user: User = Depends(require("config:write"))):
    try:
        values = policysvc.validate_policy_values(body.values)              # EC-59: reject before activation
    except (ValueError, TypeError) as exc:
        raise HTTPException(422, str(exc))
    n = db.scalar(select(func.count()).select_from(PolicyConfig)) or 0
    for old in db.scalars(select(PolicyConfig).where(PolicyConfig.active == 1)):
        old.active = 0
    new = PolicyConfig(version=f"policy_v{n + 1}", values_json=values, active=1)
    db.add(new)
    audit.record(db, user.email, "policy.update", "policy", new.version, after=values, request_id=_rid(request))
    return {"version": new.version}                                          # EC-60: pending recos keep their own version


@router.get("/health")
def health(db: Session = Depends(get_db)):
    db.execute(select(1))
    return {"status": "ok"}
