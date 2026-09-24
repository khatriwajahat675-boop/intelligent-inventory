"""End-to-end demo: Pipeline A seeds + trains, Pipeline B alerts + recommends,
both against the SAME in-memory MongoDB fake within one process.

Note on why this script exists: pipelines/training_pipeline.py and
pipelines/automation_pipeline.py are independently runnable CLI programs -
that is the point, they are two separate pipelines meant to run on a
schedule against a real, persistent MongoDB Atlas cluster (Pipeline A
nightly/on-demand for training, Pipeline B every few minutes for
alerting/restock). The in-memory fake (mongo/fake_collection.py) has no
persistence across separate process invocations, so running each pipeline
as its own `python pipelines/x.py` command against the fake would each see
an empty database. This script wires them together in-process purely to
prove the full loop end to end without a live Atlas cluster - see
docs/MONGODB.md for the real two-process/Atlas setup.

    PYTHONPATH=. python scripts/demo_pipelines_end_to_end.py
"""
from __future__ import annotations

import json
from pathlib import Path

from mongo.repository import Repositories
from pipelines import automation_pipeline as auto
from pipelines import training_pipeline as train

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    repos = Repositories.in_memory()

    print("=== Pipeline A: training + seeding ===")
    from scripts.run_model_comparison import FMCG_CAT, FMCG_NUM, GRO_CAT, GRO_NUM, run_dataset
    gro = run_dataset("grocery", GRO_NUM, GRO_CAT, max_train_rows=20_506, limitation=None)
    train.persist_model_registry(repos, gro)
    print("grocery seed:", train.seed_grocery_operational_data(repos))

    fmcg = run_dataset("fmcg", FMCG_NUM, FMCG_CAT, max_train_rows=150_000,
                       limitation="label unlearnable - see docs/STATUS.md")
    train.persist_model_registry(repos, fmcg)
    print("fmcg master seed:", train.seed_fmcg_master_data(repos))
    print("fmcg sales seed:", train.seed_fmcg_sales(repos, max_rows=200_000))

    print("\n=== Pipeline B: alerting + restock automation ===")
    summary = auto.run_once(repos)
    print(json.dumps(summary, indent=2, default=str))

    print("\n=== sample recommendations ===")
    for rec in repos.recommendations.list_open(limit=5):
        print(f"  {rec['sku']}@{rec['warehouse']}: qty={rec['recommended_qty']} state={rec['automation_state']} "
             f"pattern_source={rec['explanation']['consumption_pattern']['source']}")

    print("\n=== sample open alerts (by severity) ===")
    for a in repos.alerts.open_by_severity()[:8]:
        print(f"  [{a['severity']}] {a['type']}: {a['message']}")

    (ROOT / "reports/pipelines").mkdir(parents=True, exist_ok=True)
    (ROOT / "reports/pipelines/end_to_end_demo_summary.json").write_text(json.dumps({
        "model_registry_entries": repos.model_registry.col.count_documents({}),
        "products_seeded": repos.products.col.count_documents({}),
        "inventory_snapshots": repos.inventory.col.count_documents({}),
        "sales_transactions": repos.sales.col.count_documents({}),
        "recommendations_created": repos.recommendations.col.count_documents({}),
        "open_alerts": repos.alerts.col.count_documents({"status": "open"}),
        "audit_events": repos.audit.col.count_documents({}),
        "automation_summary": summary,
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
