"""Short-lived JWT access tokens with a revocable session id (EC-45)."""
from datetime import datetime, timedelta, timezone

import jwt


def create_token(secret: str, user_id: int, role: str, session_id: str, minutes: int = 15, alg: str = "HS256") -> str:
    now = datetime.now(timezone.utc)
    return jwt.encode({"sub": str(user_id), "role": role, "sid": session_id, "iat": now,
                       "exp": now + timedelta(minutes=minutes)}, secret, algorithm=alg)


def decode_token(secret: str, token: str, alg: str = "HS256") -> dict:
    """Raises jwt.InvalidTokenError (incl. ExpiredSignatureError) - callers map it to 401."""
    return jwt.decode(token, secret, algorithms=[alg], options={"require": ["exp", "sub", "sid"]})
