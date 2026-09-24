<<<<<<< HEAD
# intelligent-inventory
=======
# Intelligent Inventory Automation (FYP)

Smart e-commerce inventory forecasting and automated restocking, built to the *Intelligent Inventory Automation
PRD + TRD v1.0* (FastAPI + PostgreSQL + Redis/Celery worker, Python forecasting, Docker Compose).

## What is in the repo

| Path | Purpose | PRD/TRD |
|---|---|---|
| `ml/pipelines/` | data loading/validation, **70/15/15 splits**, **SMOTE** | 6.2, 13.1, 13.4 |
| `ml/models/` | baselines (naive, moving avg, seasonal naive, weekday mean), ARIMA + Prophet wrappers, stock-risk classifier | 6.2, 13 |
| `ml/evaluation/` | MAE/RMSE/WAPE/sMAPE, validation-based model selection with baseline gate | 13.2 |
| `backend/app/domain/` | pure decision engine, inventory + PO rules (fully unit tested) | 5, 14.1 |
| `backend/app/models,services,api` | PostgreSQL schema, workflows, REST API `/api/v1` | 11, 12 |
| `backend/app/security/` | bcrypt, JWT with revocable session, RBAC, rate limiting | 15 |
| `worker/` | Celery tasks + beat schedule | 14.2 |
| `infra/`, `docker-compose.yml`, `.github/workflows/ci.yml` | DevOps | 17 |
| `backend/tests`, `ml/tests`, `mongo/tests` | unit / model / API / security / MongoDB repository tests | 18 |
| `reports/` | data validation + split report, forecast evaluation, augmentation report, model comparison + charts | 18.5 |
| `augmentation/` | retrieval-grounded (RAG-style) synthetic data generation, HF embedding model selection | phase 2 |
| `ml/evaluation/` | multi-model comparison (F1/recall/PR-AUC/ROC-AUC/confidence/support/confusion matrix), SHAP/permutation interpretability, charts | phase 2 |
| `mongo/` | MongoDB Atlas layer: pydantic schemas, injection-safe filters, repositories, indexes, in-memory test fake | phase 2 |
| `pipelines/training_pipeline.py` | Pipeline A - trains + registers all models, seeds operational Mongo collections from real data | phase 2 |
| `pipelines/automation_pipeline.py` | Pipeline B - alerting + restock automation from real consumption patterns | phase 2 |

## Quick start

```bash
# 1. data validation, 70/15/15 split, SMOTE experiment, forecast evaluation (pandas/numpy/sklearn only)
PYTHONPATH=. python scripts/prepare_datasets.py
PYTHONPATH=. python scripts/run_forecast_eval.py

# 2. tests
pip install -r backend/requirements-dev.txt
PYTHONPATH=. pytest -q

# 3. run the stack
cp .env.example .env            # edit secrets
docker compose up --build -d
docker compose exec api python scripts/seed_db.py fmcg      # or: grocery --synthesize-history
# API docs: http://localhost:8000/docs   (users: admin@/manager@/ops@/analyst@/automation@example.com)

# 4. augment data (~2M synthetic rows) + compare 6 models with SHAP/interpretability + charts
pip install sentence-transformers shap      # optional - falls back to TF-IDF / permutation importance without them
PYTHONPATH=. python -m augmentation.model_selection
PYTHONPATH=. python -m augmentation.run_augmentation
PYTHONPATH=. python scripts/run_model_comparison.py

# 5. MongoDB Atlas pipelines (edit .env with MONGODB_URI first - see docs/MONGODB.md)
pip install pymongo
PYTHONPATH=. python pipelines/training_pipeline.py --atlas
PYTHONPATH=. python pipelines/automation_pipeline.py --atlas --loop 900
```

## How the two datasets are used

* **Indian FMCG 2024** (100k invoice lines, real dates) -> demand history. SKU = `Category|Brand` (64), warehouse = City (8).
  Split **chronologically** 70/15/15, day-aligned: train 2024-01-01..09-12, val ..11-05, test ..12-30.
* **E-Grocery inventory** (1000 SKUs, one row each, no sales history) -> master data (products, 10 suppliers, 5
  warehouses, lead times, safety stock, opening balances) and a **stock-risk classifier**. Split with a seeded
  stratified 70/15/15 (no time axis exists). `--synthesize-history` creates *clearly labelled* synthetic demand for demos.
* **SMOTE** is applied to the *training split only* (val/test keep the real class balance) and only to the
  classifier - never to forecasting.

See `reports/data_validation_report.md` for every check, drift statistic and experiment result, and `docs/STATUS.md`
for what is done vs. still open.
>>>>>>> 99fa566 (Initial project implementation)
