# Pipeline B - alerting & restock automation

`pipelines/automation_pipeline.py` is the second of the two pipelines requested
for this phase (`pipelines/training_pipeline.py` is the first - see
`docs/MONGODB.md` for how they fit together and how to run both against a real
Atlas cluster). This file documents what it does, why it is safe to re-run, and
how the "what should be ordered next" decision is actually made.

## What one pass does

For every `(sku, warehouse)` document in the `inventory_snapshots` collection:

1. **Read the product's purchase/sale pattern.** `consumption_pattern()` looks
   at up to 120 days of that SKU+warehouse's real `sales_transactions` history.
   With >= 10 transactions it computes a daily consumption rate *and* a trend
   label (`increasing` / `decreasing` / `stable`) by comparing the most recent
   30 days against the prior 30 (>15% change either way moves the trend off
   `stable`). With fewer than 10 real transactions it falls back to the
   product master data's declared `avg_daily_sales_hint` and marks the result
   `source: "master_data_hint"`, `quality_flag: "low_confidence"`. With
   neither, it returns a rate of 0 and `source: "no_data"` - it never invents
   a number (PRD 5.3: "do not silently fabricate predictions").
2. **Decide what to order.** The pattern feeds a `backend/app/domain/reorder.py`
   `Forecast` (constant daily rate over `lead_time_days + 7`), which goes
   through `ro.evaluate()` - the *same* pure decision engine the FastAPI
   backend uses (guardrails, MOQ/pack-size rounding, approval-state machine,
   value-limit checks; 32 unit tests in `backend/tests/test_domain_reorder.py`).
   There is exactly one reorder algorithm in this codebase; this pipeline
   reuses it against Mongo-sourced inputs instead of re-implementing it.
3. **Flag diminishing stock.** If available stock (on-hand - reserved -
   damaged) is at or below safety stock, an alert is raised: `critical` at
   zero available, `high` below half of safety stock, `medium` otherwise.
4. **Create a recommendation and/or a blocked-automation alert**, depending on
   the decision engine's output state (`AUTO_APPROVED`, `REQUIRES_APPROVAL`,
   `BLOCKED`).
5. **Record an audit event** for every recommendation created (actor,
   quantity, decision state, timestamp) - append-only, per TRD 11.2.

A run's aggregate counts and every per-SKU result are written to
`reports/pipelines/automation_pipeline_last_run.json`.

## Why it is safe to re-run (idempotency)

* **Alerts** dedupe on `dedupe_key` (e.g. `diminishing:{sku}:{warehouse}`,
  `restock:{sku}:{warehouse}`) via a partial-unique index scoped to
  `status: "open"` (`mongo/indexes.py`) - re-running the pass ten times in a
  row raises each alert at most once while it stays open; resolving it lets a
  fresh occurrence re-raise.
* **Recommendations** dedupe on `idempotency_key = sku:warehouse:YYYY-MM-DD`
  (`ro.idempotency_key`) via a unique index - at most one open recommendation
  per SKU+warehouse+day, matching the backend's own rule.
* **Sales ingestion** (`SalesRepository.insert_many_idempotent`) dedupes on
  `external_txn_id` (composite, since raw `Invoice_ID` collides - see
  `docs/STATUS.md`), so re-importing the same CSV is a no-op on the second run.

These are the same three idempotency guarantees the FastAPI backend enforces
at the API layer (TRD 8.4); here they are enforced by MongoDB unique/partial
indexes instead of a Postgres constraint, and by the equivalent hash-set logic
in `mongo/fake_collection.py`'s `InMemoryCollection` when running without a
real cluster.

## Running it

```bash
# one pass, in-memory fake (no cluster needed - good for a first look)
PYTHONPATH=. python pipelines/automation_pipeline.py --once

# one pass against a real Atlas cluster (needs MONGODB_URI in .env)
PYTHONPATH=. python pipelines/automation_pipeline.py --atlas --once

# continuous loop, e.g. every 15 minutes, in production
PYTHONPATH=. python pipelines/automation_pipeline.py --atlas --loop 900
```

`pipelines/training_pipeline.py` must be run first (or against the same
cluster beforehand) so `products`, `suppliers`, `inventory_snapshots` and
`sales_transactions` are populated - Pipeline B only reads, it never seeds
master data.

## Verified in the authoring sandbox

`scripts/demo_pipelines_end_to_end.py` wires both pipelines together in one
process against a shared `mongo.repository.Repositories.in_memory()` instance
(the in-memory fake has no cross-process persistence, so this script exists
purely to prove the loop end to end without a live Atlas cluster - see its own
module docstring). Last verified run (2026-09-23, ~3m25s, dominated by model
training, not by Mongo I/O):

* 70,116 real FMCG sales transactions seeded, 0 duplicates.
* 1,512 SKU+warehouse snapshots evaluated in Pipeline B.
* 126 diminishing-stock alerts, 387 restock recommendations, 0 blocked.
* 1,025 open alerts by severity: 40 critical / 139 high / 21 medium (grocery +
  FMCG combined - both master datasets are seeded in the same demo run).

Full output: `reports/pipelines/end_to_end_demo_summary.json` and
`reports/pipelines/automation_pipeline_last_run.json` (per-SKU detail).

### Performance note (`mongo/fake_collection.py`)

The first version of `InMemoryCollection` rescanned every existing document on
every `insert_one()` call to check unique-index constraints - an O(n) check
per insert, making the 70k-200k row sales seed an O(n^2) operation that never
finished inside a 5-minute budget. It was rewritten to track each unique
index as a Python `set` of already-seen key tuples, checked and updated in
O(1) per insert (see the class docstring in that file). This only affects the
offline fake used for tests and this demo script; a real MongoDB Atlas
collection already enforces unique indexes in the database engine and was
never subject to this bug.
