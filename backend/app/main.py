"""Application entrypoint: request IDs, structured logs, error envelope (TRD 12.1, 16.2)."""
from __future__ import annotations

import json
import logging
import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.exc import OperationalError

from backend.app.api.routes import router
from backend.app.config import get_settings

log = logging.getLogger("app")
logging.basicConfig(level=logging.INFO, format="%(message)s")


def _err(status: int, code: str, message: str, request: Request, fields=None, retryable=False):
    return JSONResponse(status_code=status, content={
        "code": code, "message": message, "field_errors": fields or [],
        "request_id": getattr(request.state, "request_id", None), "retryable": retryable})


def create_app() -> FastAPI:
    settings = get_settings()                       # fails fast on missing secrets
    app = FastAPI(title="Intelligent Inventory Automation", version="1.0.0",
                  docs_url="/docs" if settings.env != "production" else None)
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["*"], allow_headers=["*"],
                       allow_credentials=False)

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request.state.request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        t0 = time.perf_counter()
        response = await call_next(request)
        response.headers["x-request-id"] = request.state.request_id
        response.headers.setdefault("x-content-type-options", "nosniff")
        response.headers.setdefault("x-frame-options", "DENY")
        log.info(json.dumps({"request_id": request.state.request_id, "method": request.method, "path": request.url.path,
                             "status": response.status_code, "ms": round((time.perf_counter() - t0) * 1000, 1),
                             "actor": getattr(request.state, "actor", None)}))    # never logs bodies/passwords
        return response

    @app.exception_handler(HTTPException)
    async def http_exc(request, exc):
        return _err(exc.status_code, {401: "unauthenticated", 403: "forbidden", 404: "not_found", 409: "conflict",
                                      429: "rate_limited"}.get(exc.status_code, "error"), str(exc.detail), request,
                    retryable=exc.status_code in (429, 503))

    @app.exception_handler(RequestValidationError)
    async def validation_exc(request, exc):
        fields = [{"field": ".".join(str(x) for x in e["loc"]), "message": e["msg"]} for e in exc.errors()]
        return _err(422, "validation_error", "request validation failed", request, fields)

    @app.exception_handler(OperationalError)
    async def db_down(request, exc):                # EC-49: fail closed, retriable, no partial state
        return _err(503, "database_unavailable", "temporarily unavailable, retry", request, retryable=True)

    @app.exception_handler(Exception)
    async def unhandled(request, exc):
        log.exception("unhandled", extra={"request_id": request.state.request_id})
        return _err(500, "internal_error", "unexpected error", request)

    app.include_router(router)
    return app


# run with:  uvicorn backend.app.main:create_app --factory
