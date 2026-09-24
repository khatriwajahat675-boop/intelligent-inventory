# MongoDB Atlas setup (for running the pipelines for real)

This sandbox has no network access to MongoDB's servers and `pymongo` could
not be installed (see docs/STATUS.md), so `mongo/` was written and unit
tested against `mongo/fake_collection.py` - a same-interface, in-memory
stand-in. Everything below is what running it for real requires.

## 1. Create the cluster
1. Create a free Atlas account/cluster at cloud.mongodb.com (M0 free tier is enough for the FYP demo).
2. **Network access**: add your IP (or, for coursework only, `0.0.0.0/0` - never do this for anything real).
3. **Database access**: create a database user with a **least-privilege custom role**, not `atlasAdmin`:
   - `readWrite` scoped to the single `inventory` database only.
   - Never use the cluster's admin user from application code.
4. Copy the `mongodb+srv://...` connection string.

## 2. Configure this project
```bash
pip install 'pymongo[srv]'
cp .env.example .env
# edit .env:
#   MONGODB_URI=mongodb+srv://<user>:<url-encoded-password>@<cluster>.mongodb.net/?retryWrites=true&w=majority
#   MONGODB_DB_NAME=inventory
```
`mongo/config.py` reads `MONGODB_URI` from the environment only and refuses to start
if it's missing or isn't a `mongodb://`/`mongodb+srv://` URI - the connection string
is never hard-coded or committed (see `.gitignore`).

## 3. Create indexes once
```bash
PYTHONPATH=. python -c "from mongo.client import get_db; from mongo.indexes import ensure_indexes; ensure_indexes(get_db())"
```

## 4. Run the pipelines
```bash
PYTHONPATH=. python pipelines/training_pipeline.py
PYTHONPATH=. python pipelines/automation_pipeline.py --once      # single pass
PYTHONPATH=. python pipelines/automation_pipeline.py --loop 900  # every 15 minutes
```
Both default to `Repositories.in_memory()` (no cluster needed) unless `MONGODB_URI`
is set, so they can be demoed without Atlas and then pointed at a real cluster with
zero code changes - only the environment variable changes.

## Security posture (see mongo/security.py, mongo/config.py)
* Credentials only ever come from `.env` / the process environment, never from a
  request body, a CLI flag that could show up in shell history, or a hard-coded
  string in source. `.env` is git-ignored.
* TLS is enforced: Atlas `mongodb+srv://` URIs are TLS by default; a plain
  `mongodb://` URI is rejected unless it explicitly says `tls=true`.
* The database user Atlas issues should be scoped to `readWrite` on one database,
  never a cluster admin - the application does not need (and must not have)
  permission to create users, change network access rules, or drop databases.
* No query in this codebase is built from an unvalidated raw dict - every
  repository method takes typed/scalar arguments and constructs its own filter
  (see `mongo/security.py` module docstring). `sanitize_filter()` is the guarded
  fallback for the one place that still takes a free-form filter.
* Every document is validated by a `pydantic` schema (`mongo/schemas.py`,
  `extra="forbid"`) before it is sent to MongoDB - an unexpected field or a
  wrong type is rejected in Python, not silently stored.
* Logs never contain a connection string or password: `redact_uri()` /
  `redact_secrets()` mask them; `mongo/client.py` logs the redacted URI, never
  the real one.
* Idempotency and dedupe are enforced with unique/partial-unique indexes
  (`mongo/indexes.py`), the same defence-in-depth pattern as the Postgres
  backend's unique constraints - not just "check then insert" in application
  code, which race under concurrent writers.
