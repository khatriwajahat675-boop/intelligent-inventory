from __future__ import annotations

import uuid

import jwt
from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from backend.app.config import Settings, get_settings
from backend.app.db import get_db
from backend.app.models.entities import User
from backend.app.security import rbac, tokens
from backend.app.services import audit

bearer = HTTPBearer(auto_error=False)


def current_user(request: Request, creds: HTTPAuthorizationCredentials | None = Depends(bearer),
                 db: Session = Depends(get_db), settings: Settings = Depends(get_settings)) -> User:
    if creds is None:
        raise HTTPException(401, "authentication required")
    try:
        claims = tokens.decode_token(settings.jwt_secret, creds.credentials, settings.jwt_algorithm)
    except jwt.InvalidTokenError:
        raise HTTPException(401, "invalid or expired token")
    user = db.get(User, int(claims["sub"]))
    # EC-45: disabled user or rotated session id -> token dead at next check
    if user is None or user.status != "active" or user.session_id != claims["sid"]:
        raise HTTPException(401, "session revoked")
    request.state.actor = user.email
    return user


def require(permission: str):
    def dep(request: Request, user: User = Depends(current_user), db: Session = Depends(get_db)) -> User:
        if not rbac.has_permission(user.role, permission):
            audit.record(db, user.email, "security.denied", "endpoint", request.url.path,
                         reason=permission, request_id=getattr(request.state, "request_id", None))
            db.commit()                                     # keep security telemetry although we raise (FR-002)
            raise HTTPException(403, "forbidden")
        return user
    return dep


def page(limit: int = 50, offset: int = 0) -> tuple[int, int]:
    if not 1 <= limit <= 200 or offset < 0:
        raise HTTPException(422, "limit must be 1..200 and offset >= 0")
    return limit, offset


def new_request_id() -> str:
    return uuid.uuid4().hex
