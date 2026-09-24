# Full project run log

Generated: 2026-09-23T07:04:42Z

| # | stage | command | result | duration |
|---|---|---|---|---|
| 1 | data validation + 70/15/15 split | `python scripts/prepare_datasets.py` | OK | 31s |
| 2 | forecast evaluation (baselines) | `python scripts/run_forecast_eval.py` | OK | 0s |
| 3 | backend unit tests (38) | `python -m unittest discover -s backend/tests -p 'test_*.py'` | OK | 1s |
| 4 | ML/SMOTE unit tests (20) | `python -m unittest discover -s ml/tests -p 'test_*.py'` | OK | 1s |
| 5 | MongoDB repository/security tests (15) | `python -m unittest discover -s mongo/tests -p 'test_*.py'` | OK | 0s |
| 6 | embedding model selection (RAG-style) | `python -m augmentation.model_selection` | OK | 2s |
| 7 | retrieval-grounded augmentation (~2M rows) | `python -m augmentation.run_augmentation` | OK | 82s |
| 8 | 6-model comparison + SHAP + charts | `python scripts/run_model_comparison.py` | OK | 158s |
| 9 | Pipeline A: training + registry + seed | `python pipelines/training_pipeline.py --dataset both` | OK | 161s |
| 10 | Pipeline B: alerting + restock (empty DB, standalone process) | `python pipelines/automation_pipeline.py --once` | OK | 0s |
| 11 | Pipeline A -> B chained (shared in-memory DB, realistic flow) | `python scripts/demo_pipelines_end_to_end.py` | OK | 209s |

**All 11 stages completed successfully.**
