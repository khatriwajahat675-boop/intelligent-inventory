# Implementation status and honest limitations

## Full project run (2026-09-23) - `reports/RUN_LOG.md`
Every executable stage was run end to end in one continuous pass via `scripts/run_everything.sh`
(not part of the delivered code path - a one-off orchestration script written purely to produce this
proof-of-execution log). All 11 stages passed, 0 failures, ~11.5 minutes total: data validation + split (31s),
forecast eval, all 3 test suites (73 tests), embedding model selection, 2M-row augmentation (82s), 6-model
comparison + SHAP/charts (158s), Pipeline A on both datasets (161s), Pipeline B standalone on an empty DB
(0s, no crash), and Pipeline A -> B chained realistically (209s). **Not runnable here:** `docker compose up`
(no Docker daemon in this container - CLI present but no `dockerd`) and anything needing FastAPI/SQLAlchemy/
pytest/sentence-transformers/shap/pymongo from PyPI (org egress policy returns 403) - see the next section.

## Verified in the authoring sandbox (73 tests + 4 pipeline/script runs)
* Data validation, chronological / stratified 70/15/15 splits, PSI drift, SMOTE, stock-risk experiments.
* Metrics, baselines, validation-based model selection, cold start / intermittent / negative-clamp / failure isolation.
* Decision engine (TRD 12.3 example reproduced exactly: LTD 82, ROP 102, shortage 62, qty 72), guardrails, approval
  state machine, ledger and PO rules, RBAC matrix, JWT, rate limiter.
* Test suite grew from 58 -> 73 tests across this phase: 38 backend (`backend/tests`), 20 ML/SMOTE
  (`ml/tests/test_ml.py`), 15 MongoDB repository/security (`mongo/tests`). All green as of 2026-09-23.

## Phase 2: retrieval-grounded augmentation, multi-model comparison, MongoDB pipelines
* **Augmentation** (`augmentation/`): not literal RAG (there is no free-text corpus to retrieve passages
  from), so per the agreed scope (see `docs/AUGMENTATION.md`) this implements *retrieval-grounded synthesis* -
  a HuggingFace sentence-transformer embeds each row's categorical context, a model-selection harness scores
  candidate embedding backends by k-NN neighbor-label purity and picks the best one, then SMOTE-style numeric
  interpolation + atomic categorical copying generates synthetic rows within each retrieved neighborhood.
  HuggingFace/PyPI had no network access in the authoring sandbox, so `augmentation/embedding_backends.py`'s
  `HFSentenceTransformerBackend` could not actually download a model; the harness fell back to (and selected)
  its offline `TfidfEmbeddingBackend` (TF-IDF + TruncatedSVD), which implements the identical retrieval
  interface. **On your machine, `pip install sentence-transformers` and re-run
  `python -m augmentation.model_selection` first** - it will try every model in `CANDIDATE_HF_MODELS`, score
  them for real, and the rest of the pipeline picks up whichever one wins with zero code changes.
  Generated 2,000,000 synthetic rows total (target ~20 lacs), split proportionally: FMCG 1,980,198 / grocery
  19,802 (train split only; validation/test stay 100% real). 0 validation failures on either dataset.
  Full report: `reports/augmentation/augmentation_report.md`.
* **Model comparison** (`ml/evaluation/`, `scripts/run_model_comparison.py`): 6 models per dataset
  (majority-class baseline, logistic regression, decision tree, random forest, HistGradientBoosting, MLP),
  each scored on the held-out real TEST split for precision, recall, F1, ROC-AUC, PR-AUC, confidence (mean
  predicted-class probability), support, and the full confusion matrix (TP/FP/TN/FN) - see
  `reports/model_comparison/model_comparison_report.md` and the 14 PNG charts under
  `reports/model_comparison/plots/` (metric bars, confusion matrix, ROC/PR overlays, confidence-vs-support,
  feature importance, learning curve; colorblind-safe palette from the dataviz skill). Grocery: best model
  `hist_gradient_boosting`, F1 0.866 / ROC-AUC 0.984 / PR-AUC 0.957. FMCG: best-by-F1 is `decision_tree`, but
  every model sits at ROC-AUC ~0.49-0.50 - see the FMCG label-unlearnability finding below, restated in the
  report itself so it isn't mistaken for a working classifier.
* **Interpretability** (`ml/evaluation/interpretability.py`): the `shap` package was not installed in the
  sandbox, so it falls back to `sklearn.inspection.permutation_importance` (+ partial dependence for the top
  feature) behind the *same* `explain()` function SHAP would use - **`pip install shap` on your machine and
  it will use `TreeExplainer`/`KernelExplainer` automatically**, no other code changes needed. Top drivers
  reported per dataset (e.g. grocery: `Lead_Time_Days`, `SKU_Churn_Rate`).
* **MongoDB Atlas layer** (`mongo/`): pydantic `extra="forbid"` schemas validate every document before it
  reaches the database; `security.py` rejects `$where`/`$function`/`$accumulator`/`$expr` and any operator
  outside an explicit allow-list (NoSQL injection prevention, tested in `mongo/tests/test_security.py`); TLS
  is enforced in `config.py`; unique/partial-unique indexes (`indexes.py`) mirror the Postgres idempotency
  constraints. **Not run against a real Atlas cluster** - `pymongo` could not be installed in the sandbox
  (see the package-index note below). Everything was built and tested against
  `mongo/fake_collection.py`'s `InMemoryCollection`, which implements the identical `CollectionLike`
  interface, so `mongo/client.py`'s `get_db()` is the only thing to point at a real cluster - see
  `docs/MONGODB.md` for the Atlas setup steps.
* **Pipeline A - training** (`pipelines/training_pipeline.py`): trains the full model zoo on the
  augmented+SMOTE train split, evaluates on real val/test, and persists every model's scorecard (plus which
  one is best) to the `model_registry` collection; separately seeds `products`/`suppliers`/
  `inventory_snapshots`/`sales_transactions` from **real data only** (augmentation never touches the
  operational collections a live system would read from).
* **Pipeline B - alerting & restock automation** (`pipelines/automation_pipeline.py`): analyses real
  purchase/sale velocity and trend per SKU+warehouse (falling back honestly to a declared hint or "no data",
  never fabricating a number), reuses the *same* `backend/app/domain/reorder.py` decision engine the FastAPI
  backend uses to decide what to order next, raises de-duplicated diminishing-stock/restock alerts, and
  writes idempotent recommendations. Full detail, idempotency guarantees, and the last verified run's numbers
  (70,116 real transactions seeded, 1,512 SKUs evaluated, 126 alerts, 387 recommendations) are in
  `docs/AUTOMATION_PIPELINE.md`.
* **Performance bugs found and fixed in this phase** (both while still in-sandbox, both re-verified against
  their existing test suites afterward - no behavior change, just complexity): `ml/pipelines/smote.py`'s
  naive O(n^2) pairwise-distance broadcast was replaced with `sklearn.neighbors.NearestNeighbors` (was OOM-
  killed on grocery's ~5k-row minority class); `mongo/fake_collection.py`'s per-insert O(n) unique-constraint
  rescan was replaced with O(1) hash-set tracking (was making a 200k-row sales seed hang past 5 minutes).
  Neither bug can occur against a real MongoDB Atlas cluster, which enforces unique indexes natively.

## Written but NOT executed here (package index was blocked: FastAPI, SQLAlchemy, bcrypt, statsmodels, prophet, pytest)
* `backend/app/{db,models,services,api,main}.py`, `worker/`, `scripts/seed_db.py`, `backend/tests/api/`, ARIMA/Prophet wrappers.
  They compile, and API tests are ready; run `pytest` on your machine first and fix whatever the environment reveals.

## Not built yet (deliberately deferred / needs your decision)
* React/Next.js frontend (TRD 10.1) - compose has a placeholder. Suggested next step.
* LSTM candidate (TRD marks it optional; defend ARIMA/Prophet/baseline unless history volume justifies it).
* Alembic initial revision (see `backend/migrations/README.md`), CSRF protections (only needed if cookie auth is adopted;
  bearer tokens are used), object storage, HTTPS certificates.
* Backup/restore drill and model-artifact registry with checksums (TRD 11.3/13.5) - schema hooks exist, jobs do not.

## Findings about the data that affect your evaluation chapter
* FMCG demand has almost **no seasonality** (weekday CV 0.9%, month CV 2.6%) -> advanced models will not beat a moving
  average by much; that is a data property, and the baseline gate (EC-28) handles it correctly. State this in the report.
* FMCG `Stock_On_Hand`/`Reorder_Level` are per-invoice random draws, so the stock-risk label built from them has no
  learnable signal (test PR-AUC 0.016 = base rate). Use the grocery file for the classifier; use FMCG only for demand.
* Grocery stock-risk test set is small (148 rows, 35 positives): the RF+SMOTE gain (F1 0.906 -> 0.925, recall 0.83 ->
  0.89) is within noise; SMOTE *hurt* logistic regression F1. Report both honestly, ideally with cross-validation.
* `Invoice_ID` is not unique (62 collisions); a composite `external_txn_id` is generated for idempotency.
* MOQ and pack size do not exist in either file (assumed 1/1) - set real values before demonstrating MOQ/pack rules.
