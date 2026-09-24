"""MongoDB Atlas client factory. pymongo is imported lazily so the rest of this
package (and its offline-testable repositories) stays importable without it."""
from __future__ import annotations

import logging

from mongo.config import get_mongo_settings, redact_uri

log = logging.getLogger("mongo")
_client = None


def get_client():
    global _client
    if _client is None:
        try:
            from pymongo import MongoClient
            from pymongo.server_api import ServerApi
        except ImportError as exc:
            raise RuntimeError(
                "pymongo is not installed in this environment. Run: pip install 'pymongo[srv]'  "
                "(the [srv] extra is required for mongodb+srv:// Atlas connection strings). "
                "See docs/MONGODB.md for full setup steps."
            ) from exc
        s = get_mongo_settings()
        log.info("connecting to MongoDB Atlas: %s (db=%s)", redact_uri(s.uri), s.db_name)
        _client = MongoClient(
            s.uri,
            serverSelectionTimeoutMS=s.server_selection_timeout_ms,
            connectTimeoutMS=s.connect_timeout_ms,
            maxPoolSize=s.max_pool_size,
            appName=s.app_name,
            server_api=ServerApi("1"),
            tz_aware=True,
            retryWrites=True,
            tls=True,                          # explicit even though srv:// implies it - never silently plaintext
        )
    return _client


def get_db():
    return get_client()[get_mongo_settings().db_name]


def ping() -> bool:
    from pymongo.errors import PyMongoError
    try:
        get_client().admin.command("ping")
        return True
    except PyMongoError:
        return False


def close_client() -> None:
    global _client
    if _client is not None:
        _client.close()
        _client = None
