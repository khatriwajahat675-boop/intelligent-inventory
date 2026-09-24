"""Find the "one best" embedding model for retrieval-grounded synthesis.

Methodology (run on a stratified sample for speed, default n=3000 rows):
for each candidate model, embed each row's categorical-context text, find its
k nearest neighbours by cosine similarity, and score **retrieval purity** -
the fraction of neighbours that share the row's own category AND brand/class.
High purity means "this model's neighbourhoods are the groups a domain expert
would also group together", which is exactly the property the synthesis step
in augmentation/synthesize.py depends on. Ties are broken by encoding speed
(rows/sec), since the real run embeds ~100k-1M real rows.

This harness iterates over augmentation.embedding_backends.CANDIDATE_HF_MODELS.
In THIS sandbox (no network access to huggingface.co - see docs/AUGMENTATION.md)
every HF candidate fails to load and is recorded as "unavailable"; the offline
TF-IDF backend is scored alongside them so the harness still produces a real,
executed comparison. On a machine with internet access, re-running this
unmodified script downloads and scores the real HF models and will very likely
pick `sentence-transformers/all-MiniLM-L6-v2` - it is the standard choice for
short categorical text at this corpus size (384-dim, ~14k rows/sec on CPU,
strong MTEB retrieval score for its size) - but the harness decides empirically
rather than hard-coding that assumption.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from augmentation.embedding_backends import CANDIDATE_HF_MODELS, build_backend, timed_encode


def row_text(df: pd.DataFrame, cols: list[str]) -> list[str]:
    """Serialise a row's categorical context into one short text string."""
    return (df[cols].astype(str).agg(" | ".join, axis=1)).tolist()


def purity_at_k(embeddings: np.ndarray, labels: np.ndarray, k: int = 10) -> float:
    n = len(embeddings)
    k = min(k, n - 1)
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(embeddings)
    _, idx = nn.kneighbors(embeddings)
    neighbour_labels = labels[idx[:, 1:]]              # drop self-match at column 0
    same = (neighbour_labels == labels[:, None]).mean(axis=1)
    return float(same.mean())


def evaluate_candidates(df: pd.DataFrame, text_cols: list[str], label_col: str, sample_n: int = 3000,
                        k: int = 10, seed: int = 42) -> list[dict]:
    rng = np.random.default_rng(seed)
    sample = df.sample(n=min(sample_n, len(df)), random_state=seed) if len(df) > sample_n else df
    texts, labels = row_text(sample, text_cols), sample[label_col].astype(str).to_numpy()

    results = []
    for name in ["tfidf", *CANDIDATE_HF_MODELS]:
        backend = build_backend(name)
        try:
            emb, secs = timed_encode(backend, texts)
        except RuntimeError as exc:                    # expected here: no network to huggingface.co
            results.append({"model": name, "available": False, "reason": str(exc)})
            continue
        purity = purity_at_k(emb, labels, k=k)
        results.append({"model": name, "available": True, "purity_at_k": round(purity, 4),
                        "rows_per_sec": round(len(texts) / max(secs, 1e-6), 1), "embed_dim": emb.shape[1],
                        "seconds_for_sample": round(secs, 3)})
    return results


def pick_best(results: list[dict]) -> dict:
    available = [r for r in results if r["available"]]
    if not available:
        raise RuntimeError("no embedding backend available - not even the offline fallback")
    # rank by purity first, encoding speed as tiebreak (matters at 100k-1M row scale)
    return max(available, key=lambda r: (r["purity_at_k"], r["rows_per_sec"]))


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    fmcg = pd.read_csv(root / "data/processed/fmcg/train.csv")
    results = evaluate_candidates(fmcg, text_cols=["Category", "Brand", "City", "Store_Format", "Channel"],
                                  label_col="sku_code")
    best = pick_best(results)
    out = {"candidates": results, "selected_model": best["model"], "selection_reason":
           f"highest neighbour purity ({best['purity_at_k']}) among available backends at k=10; "
           f"{best['rows_per_sec']} rows/sec used as tiebreak",
           "note": "HF candidates show available=false in this sandbox (no network access to "
                   "huggingface.co); re-run on a machine with internet to get a real HF comparison - "
                   "the offline result below is what augmentation/synthesize.py was actually run with."}
    rep = root / "reports/augmentation"
    rep.mkdir(parents=True, exist_ok=True)
    (rep / "embedding_model_comparison.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
