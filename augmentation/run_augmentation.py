"""Driver: augment the TRAIN split of both datasets to ~2,000,000 synthetic rows
combined (the volume split proportionally to each dataset's original size),
validate the result, and write augmented train sets + a report.

    PYTHONPATH=. python augmentation/run_augmentation.py                 # full ~2M run
    PYTHONPATH=. python augmentation/run_augmentation.py --target 20000  # quick smoke test

VAL and TEST are never touched - only data/processed/{fmcg,grocery}/train.csv
are read; augmented output goes to train_augmented.csv.gz beside them.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from augmentation.embedding_backends import build_backend
from augmentation.recipes import FMCG_RECIPE, GROCERY_RECIPE
from augmentation.synthesize import synthesize_rows
from ml.pipelines import data_prep as dp

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data/processed"
REP = ROOT / "reports/augmentation"


def _model_name() -> str:
    f = REP / "embedding_model_comparison.json"
    if f.exists():
        return json.loads(f.read_text())["selected_model"]
    return "tfidf"


def augment_one(name: str, recipe, real_train: pd.DataFrame, target_n: int, backend_name: str,
                validate_fn) -> dict:
    t0 = time.perf_counter()
    real_tagged = real_train.copy()
    real_tagged["is_synthetic"] = False
    real_tagged["synthesis_backend"] = None
    real_tagged["synthesis_seed_index"] = -1
    real_tagged["synthesis_neighbor_index"] = -1

    backend = build_backend(backend_name)
    synth = synthesize_rows(real_train, recipe, backend, target_n, k=10, seed=42)
    elapsed = time.perf_counter() - t0

    combined = pd.concat([real_tagged, synth], ignore_index=True, sort=False)
    checks = validate_fn(combined)
    n_fail = sum(1 for c in checks if c.status == "FAIL")

    out_path = PROC / name / "train_augmented.csv.gz"
    combined.to_csv(out_path, index=False, compression="gzip")

    return {
        "dataset": name, "embedding_backend": backend.name, "real_rows": len(real_train),
        "synthetic_rows_generated": len(synth), "combined_rows": len(combined),
        "target_synthetic_rows": target_n, "seconds": round(elapsed, 2),
        "rows_per_sec": round(target_n / max(elapsed, 1e-6), 1),
        "output_file": str(out_path.relative_to(ROOT)),
        "output_bytes": out_path.stat().st_size,
        "validation_checks_failed": n_fail,
        "validation_checks": [c.as_dict() for c in checks],
        "synthetic_positive_rate": round(float(synth["stock_risk"].mean()), 4) if len(synth) else None,
        "real_positive_rate": round(float(real_train["stock_risk"].mean()), 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=2_000_000,
                    help="total NEW synthetic rows across both datasets (default ~20 lacs)")
    ap.add_argument("--backend", default=None, help="override embedding backend/model name")
    args = ap.parse_args()

    fmcg_train = pd.read_csv(PROC / "fmcg/train.csv", parse_dates=["Invoice_Date"])
    gro_train = pd.read_csv(PROC / "grocery/train.csv", parse_dates=["Received_Date", "Last_Purchase_Date",
                                                                      "Expiry_Date", "Audit_Date"])
    # original combined size decides the proportional split (both datasets chosen -> weight by real size)
    orig_total = 100_000 + 1_000    # full FMCG + grocery datasets (not just their train slices)
    fmcg_target = round(args.target * 100_000 / orig_total)
    gro_target = args.target - fmcg_target

    backend_name = args.backend or _model_name()
    REP.mkdir(parents=True, exist_ok=True)

    results = [
        augment_one("fmcg", FMCG_RECIPE, fmcg_train, fmcg_target, backend_name, dp.validate_fmcg),
        augment_one("grocery", GROCERY_RECIPE, gro_train, gro_target, backend_name, dp.validate_grocery),
    ]
    report = {"requested_total_synthetic_rows": args.target, "embedding_backend": backend_name,
              "proportional_split": {"fmcg": fmcg_target, "grocery": gro_target}, "datasets": results}
    (REP / "augmentation_report.json").write_text(json.dumps(report, indent=2, default=str))
    write_markdown(report)
    print(json.dumps({d["dataset"]: {k: d[k] for k in ("real_rows", "synthetic_rows_generated", "combined_rows",
                                                        "seconds", "validation_checks_failed")} for d in results},
                     indent=2))


def write_markdown(report: dict) -> None:
    L = ["# Data augmentation report (retrieval-grounded synthesis)", "",
        f"Embedding backend used: **{report['embedding_backend']}** "
        "(see reports/augmentation/embedding_model_comparison.json for how it was selected)", "",
        f"Requested synthetic volume: **{report['requested_total_synthetic_rows']:,}** rows, split "
        f"proportionally to original dataset size: FMCG {report['proportional_split']['fmcg']:,}, "
        f"grocery {report['proportional_split']['grocery']:,}.", "",
        "Only the TRAIN split was augmented; validation and test sets are 100% real (see docs/AUGMENTATION.md).", ""]
    for d in report["datasets"]:
        L += [f"## {d['dataset']}", "",
             f"- real train rows: {d['real_rows']:,}", f"- synthetic rows generated: {d['synthetic_rows_generated']:,}",
             f"- combined (real + synthetic) train rows: {d['combined_rows']:,}",
             f"- generation time: {d['seconds']}s ({d['rows_per_sec']:,}/sec)",
             f"- output file: `{d['output_file']}` ({d['output_bytes'] / 1e6:.1f} MB gzip)",
             f"- real positive rate (stock_risk): {d['real_positive_rate']:.2%}  |  "
             f"synthetic positive rate: {d['synthetic_positive_rate']:.2%}",
             f"- validation checks failed: {d['validation_checks_failed']}", "",
             "| check | status | detail |", "|---|---|---|"]
        L += [f"| {c['name']} | {c['status']} | {c['detail']} |" for c in d["validation_checks"]]
        L.append("")
    (REP / "augmentation_report.md").write_text("\n".join(L))


if __name__ == "__main__":
    main()
