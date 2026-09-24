"""A minimal in-process stand-in for a pymongo Collection.

Exists so mongo/repository.py can be unit tested (and the two pipelines
dry-run end to end) without a real MongoDB Atlas cluster - pymongo could not
be installed in the authoring sandbox (see docs/STATUS.md). It implements
only the operations this project's repositories actually use, with the same
method names and semantics (including raising DuplicateKeyError on a unique-
index violation) so repository.py's code is identical against the fake and
against a real `pymongo.collection.Collection` - swapping mongo/client.py's
get_db() for a real Atlas connection is the only change needed in production.
"""
from __future__ import annotations

import itertools
from typing import Any, Callable

try:
    from pymongo.errors import DuplicateKeyError
except ImportError:                                       # pymongo not installed - define a compatible stand-in
    class DuplicateKeyError(Exception):
        pass

_counter = itertools.count(1)


def _get(doc: dict, path: str) -> Any:
    cur = doc
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _matches(doc: dict, filt: dict) -> bool:
    for key, cond in filt.items():
        val = _get(doc, key)
        if isinstance(cond, dict) and any(k.startswith("$") for k in cond):
            for op, arg in cond.items():
                if op == "$eq" and val != arg:
                    return False
                if op == "$ne" and val == arg:
                    return False
                if op == "$in" and val not in arg:
                    return False
                if op == "$nin" and val in arg:
                    return False
                if op == "$gte" and not (val is not None and val >= arg):
                    return False
                if op == "$lte" and not (val is not None and val <= arg):
                    return False
                if op == "$gt" and not (val is not None and val > arg):
                    return False
                if op == "$lt" and not (val is not None and val < arg):
                    return False
                if op == "$exists" and (val is not None) != bool(arg):
                    return False
        elif val != cond:
            return False
    return True


class InMemoryCollection:
    """Unique indexes are enforced with an O(1)-per-insert hash set, not a
    rescan of every existing document - a naive O(n) "scan all docs" check
    makes bulk-loading n rows O(n^2), which was measured to make a 200,000-row
    sales-transaction seed effectively hang. See mongo/tests/test_repository.py
    and pipelines/training_pipeline.py's FMCG sales seed for the workload this
    was sized against.
    """

    def __init__(self, name: str = "fake"):
        self.name = name
        self._docs: list[dict] = []
        self._unique_indexes: list[dict] = []      # [{"fields": [...], "partial": {...}|None, "seen": set()}]

    # -- index management -------------------------------------------------
    def create_index(self, keys, unique: bool = False, partialFilterExpression: dict | None = None, **_):
        if not unique:
            return
        if isinstance(keys, str):
            fields = [keys]
        else:
            fields = [k[0] if isinstance(k, tuple) else k for k in keys]
        self._unique_indexes.append({"fields": fields, "partial": partialFilterExpression, "seen": set()})

    def _unique_key(self, index: dict, doc: dict):
        if index["partial"] and not _matches(doc, index["partial"]):
            return None                              # doc is outside the partial index's scope - not tracked
        return tuple(_get(doc, f) for f in index["fields"])

    def _register_unique(self, doc: dict) -> None:
        for index in self._unique_indexes:
            key = self._unique_key(index, doc)
            if key is not None:
                index["seen"].add(key)

    def _check_unique(self, doc: dict) -> None:
        for index in self._unique_indexes:
            key = self._unique_key(index, doc)
            if key is not None and key in index["seen"]:
                raise DuplicateKeyError(f"duplicate key in {self.name}: {dict(zip(index['fields'], key))}")

    # -- CRUD ---------------------------------------------------------------
    def insert_one(self, doc: dict):
        doc = dict(doc)
        doc.setdefault("_id", next(_counter))
        self._check_unique(doc)
        self._register_unique(doc)
        self._docs.append(doc)
        return type("Result", (), {"inserted_id": doc["_id"]})()

    def find_one(self, filt: dict | None = None):
        filt = filt or {}
        for d in self._docs:
            if _matches(d, filt):
                return dict(d)
        return None

    def find(self, filt: dict | None = None, sort: list[tuple] | None = None, limit: int | None = None):
        filt = filt or {}
        out = [dict(d) for d in self._docs if _matches(d, filt)]
        if sort:
            for field, direction in reversed(sort):
                out.sort(key=lambda d: (_get(d, field) is None, _get(d, field)), reverse=direction < 0)
        return out[:limit] if limit else out

    def count_documents(self, filt: dict | None = None) -> int:
        return len(self.find(filt))

    def update_one(self, filt: dict, update: dict, upsert: bool = False):
        for d in self._docs:
            if _matches(d, filt):
                self._apply_update(d, update)
                return type("Result", (), {"matched_count": 1, "upserted_id": None})()
        if upsert:
            new_doc = {k: v for k, v in filt.items() if not k.startswith("$") and not isinstance(v, dict)}
            self._apply_update(new_doc, update)
            new_doc.setdefault("_id", next(_counter))
            self._check_unique(new_doc)
            self._register_unique(new_doc)
            self._docs.append(new_doc)
            return type("Result", (), {"matched_count": 0, "upserted_id": new_doc["_id"]})()
        return type("Result", (), {"matched_count": 0, "upserted_id": None})()

    @staticmethod
    def _apply_update(doc: dict, update: dict) -> None:
        for k, v in update.get("$set", {}).items():
            doc[k] = v
        for k, v in update.get("$inc", {}).items():
            doc[k] = doc.get(k, 0) + v
        for k, v in update.get("$setOnInsert", {}).items():
            doc.setdefault(k, v)
