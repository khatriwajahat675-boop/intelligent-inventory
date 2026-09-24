"""NoSQL-injection defence and secret-handling helpers.

The rule enforced everywhere in mongo/repository.py: every query sent to
MongoDB is built from typed, validated arguments (pydantic models, literal
strings/ints/dates) - NEVER from a raw dict/JSON blob that came from outside
the process (an HTTP body, a chat message, a CLI arg passed straight
through). That is the actual injection defence; a classic MongoDB attack
looks like a client sending `{"password": {"$ne": null}}` as a "value" to
bypass an equality check, or `{"$where": "sleep(10000)"}` to execute
arbitrary JS server-side. `sanitize_filter()` below is the last line of
defence for the one place (an admin/debug query helper) that still accepts
a free-form filter - every other repository method takes scalars/typed
models and constructs its own filter internally, where these attacks have
no surface at all.
"""
from __future__ import annotations

_FORBIDDEN_OPERATORS = {"$where", "$function", "$accumulator", "$expr"}
_ALLOWED_OPERATORS = {"$in", "$nin", "$gte", "$lte", "$gt", "$lt", "$eq", "$exists", "$and", "$or", "$ne"}


def sanitize_filter(filt: dict) -> dict:
    """Reject JS-execution operators and any operator not on the allow-list.
    Raises ValueError - callers must not silently drop or "fix" the filter."""
    if not isinstance(filt, dict):
        raise ValueError("filter must be a dict")
    _check(filt, depth=0)
    return filt


def _check(node, depth: int) -> None:
    if depth > 6:
        raise ValueError("filter nesting too deep (possible injection attempt)")
    if isinstance(node, dict):
        for k, v in node.items():
            if k in _FORBIDDEN_OPERATORS:
                raise ValueError(f"forbidden operator in query: {k}")
            if isinstance(k, str) and k.startswith("$") and k not in _ALLOWED_OPERATORS:
                raise ValueError(f"unrecognised/disallowed query operator: {k}")
            _check(v, depth + 1)
    elif isinstance(node, list):
        for item in node:
            _check(item, depth + 1)


_SECRET_KEY_MARKERS = ("password", "password_hash", "uri", "connection_string", "secret", "token", "api_key")


def redact_secrets(d: dict) -> dict:
    """Recursively mask anything that looks like a credential before it is logged."""
    out = {}
    for k, v in d.items():
        if isinstance(k, str) and any(m in k.lower() for m in _SECRET_KEY_MARKERS):
            out[k] = "***REDACTED***"
        elif isinstance(v, dict):
            out[k] = redact_secrets(v)
        else:
            out[k] = v
    return out
