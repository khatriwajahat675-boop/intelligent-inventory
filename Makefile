.PHONY: data test up seed
data:   ; PYTHONPATH=. python scripts/prepare_datasets.py && PYTHONPATH=. python scripts/run_forecast_eval.py
test:   ; PYTHONPATH=. python -m pytest backend/tests ml/tests -q
up:     ; docker compose up --build -d
seed:   ; docker compose exec api python scripts/seed_db.py fmcg
