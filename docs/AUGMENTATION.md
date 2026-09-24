# Data augmentation: what "RAG-style" means here, and its limits

## Why not literal RAG
Retrieval-Augmented Generation is an LLM pattern: embed a question, retrieve
relevant documents, generate a text answer grounded in them. There is no
question and no document corpus in an inventory dataset, so literal RAG does
not apply. What this pipeline borrows from it - and what it actually
implements (`augmentation/synthesize.py`) - is the **retrieval-grounded**
half of the pattern:

1. Embed each real row's categorical context with a HuggingFace sentence
   embedding model (`augmentation/embedding_backends.py`).
2. Retrieve its nearest real neighbours (scoped to the same SKU / category -
   see the module docstring for why a scoped search replaces a global O(n^2)
   k-NN at this scale).
3. Generate a new row only inside that neighbourhood: numeric fields are
   SMOTE-style interpolated between two real, similar rows; categorical and
   date fields are copied atomically from one of the two (never mixed
   field-by-field), so every synthetic categorical/date combination already
   existed in the real data.
4. Derived/business fields (Revenue, Total_Inventory_Value_USD, ...) are
   always recomputed from the synthesised inputs, never interpolated.

Every synthetic row is tagged `is_synthetic=True` with its seed/neighbour row
indices and the embedding backend used - synthetic data must stay
distinguishable from real evidence, not just statistically similar to it.

## What actually ran in this sandbox vs. what should run for the FYP
This sandbox has no network access to `huggingface.co` (organisation egress
policy), so `sentence-transformers` cannot be installed or a model
downloaded. `augmentation/model_selection.py` still runs a real, executed
comparison: it tries every candidate HuggingFace model
(`augmentation.embedding_backends.CANDIDATE_HF_MODELS`), records each as
"unavailable" here, and falls back to an offline TF-IDF+SVD embedding
(`TfidfEmbeddingBackend`) that implements the exact same interface. That
fallback is what generated the ~2,000,000 rows in `reports/augmentation/`.

**On your machine** (`pip install sentence-transformers`, with internet
access): re-run `python augmentation/model_selection.py` unmodified. It will
download and score the real candidates and very likely select
`sentence-transformers/all-MiniLM-L6-v2` (fast, 384-dim, strong retrieval
quality for its size - the standard default for short categorical text at
this corpus size), then `python augmentation/run_augmentation.py` will pick
that model up automatically from `reports/augmentation/embedding_model_comparison.json`
and use it for the real run. No other code changes.

## Volume and split
"~20 lacs" (2,000,000) is interpreted as the number of **new synthetic rows
generated**, split proportionally to each dataset's original size (FMCG
100,000 : grocery 1,000 -> ~99.0% / ~1.0%): FMCG gets ~1,980,000 synthetic
rows added to its 70,116 real training rows; grocery gets ~19,800 added to
its 704. This ratio, not an equal 50/50 split, is what "both, proportionally"
means in `reports/augmentation/augmentation_report.md`.

## Rules that keep the augmentation honest
* **Only the TRAIN split is ever augmented.** `augmentation/run_augmentation.py`
  reads exclusively from `data/processed/{fmcg,grocery}/train.csv`. Validation
  and test stay 100% real, or every metric downstream would be measuring the
  model's ability to recognise its own synthetic near-duplicates rather than
  genuine generalisation.
* Every augmented file is re-validated with the **same** `ml/pipelines/data_prep`
  checks the original real data went through (see the report) - a synthetic
  row is held to the same quality bar as a real one, not a lower one.
* Composed with SMOTE, not a replacement for it: this pipeline adds **volume
  and diversity** (more plausible rows); `ml/pipelines/smote.py` still handles
  **class balance** on top of it during model training. They solve different
  problems and both apply only to the training split.
* FMCG's `stock_risk` label is per-invoice random noise (see `docs/STATUS.md` -
  test PR-AUC = base rate on the *real* data already). Augmenting it does not
  create a learnable signal that was not there; `reports/model_comparison/`
  reports this honestly rather than hiding it behind a bigger dataset.
