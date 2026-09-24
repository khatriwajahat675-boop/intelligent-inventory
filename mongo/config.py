"""MongoDB Atlas connection settings - env-only, fails fast on a missing/malformed
URI, and TLS is required (Atlas SRV URIs are TLS by default; a plain `mongodb://`
URI must say so explicitly). Never construct a connection string by string-
concatenating user input - it belongs in .env, read once, here."""
from __future__ import annotations

import re

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class MongoSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", env_prefix="MONGODB_")

    uri: str = Field(..., description="mongodb+srv://user:pass@cluster.mongodb.net/?retryWrites=true&w=majority")
    db_name: str = "inventory"
    server_selection_timeout_ms: int = 8000
    connect_timeout_ms: int = 8000
    max_pool_size: int = 50
    app_name: str = "intelligent-inventory"

    @field_validator("uri")
    @classmethod
    def _require_tls(cls, v: str) -> str:
        if not (v.startswith("mongodb+srv://") or v.startswith("mongodb://")):
            raise ValueError("MONGODB_URI must start with mongodb:// or mongodb+srv://")
        if v.startswith("mongodb://") and "tls=true" not in v.replace(" ", "").lower():
            raise ValueError("a plain mongodb:// URI must explicitly include tls=true "
                             "(mongodb+srv:// Atlas URIs are TLS by default and don't need this)")
        return v


_settings: MongoSettings | None = None


def get_mongo_settings() -> MongoSettings:
    global _settings
    if _settings is None:
        _settings = MongoSettings()          # raises ValidationError if MONGODB_URI is unset - fail fast, not silently
    return _settings


def redact_uri(uri: str) -> str:
    """`mongodb+srv://user:pass@cluster/...` -> `mongodb+srv://***:***@cluster/...` for logs/errors."""
    return re.sub(r"//[^:@/]+:[^:@/]+@", "//***:***@", uri)
