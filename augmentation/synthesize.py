"""Retrieval-grounded synthetic row generation ("RAG-style" tabular augmentation).

Algorithm (vectorised with numpy so it scales to millions of rows):

  1. Embed each real row's categorical context (augmentation.embedding_backends).
  2. Retrieve its k nearest real neighbours by cosine similarity - this is the
     "retrieval" step: every synthetic row is grounded in a real neighbourhood,
     never sampled from an unconstrained distribution.
  3. For each synthetic row: pick a random real seed row and a random one of
     its k neighbours.
       - numeric fields are SMOTE-style interpolated between the two
         (single random gap per row, so joint numeric correlations survive -
         e.g. Units and Revenue move together instead of independently).
       - categorical/date fields are copied ATOMICALLY from whichever of the
         two rows a coin flip selects (never mixed field-by-field), which
         guarantees every synthetic categorical/date combination already
         existed in the real data - no impossible Category x Brand pairs.
  4. Derived fields (Revenue, Margin, Total_Inventory_Value_USD, ...) are
     recomputed from the synthesised inputs, never interpolated directly -
     business identities (Revenue = Units x Price) always hold exactly.
  5. Every synthetic row is tagged `is_synthetic=True` with its seed/neighbour
     row ids and the embedding backend used, for provenance/auditability -
     synthetic data must never be silently indistinguishable from real
     evidence (see docs/SECURITY.md).

ONLY the TRAIN split is ever passed in here. Augmenting validation or test
data would let synthetic near-duplicates of evaluation rows leak into
training and inflate every metric - see docs/AUGMENTATION.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from augmentation.embedding_backends import EmbeddingBackend
from augmentation.model_selection import row_text


@dataclass
class SynthesisRecipe:
    name: str
    numeric_cols: list[str]
    categorical_cols: list[str]     # copied atomically from seed OR neighbour
    inherit_cols: list[str]         # e.g. dates - copied atomically, same source as categoricals
    context_cols: list[str]         # serialised to text for embedding/retrieval
    group_col: str                  # retrieval is scoped to rows sharing this key (see module docstring)
    postprocess: Callable[[pd.DataFrame], pd.DataFrame]   # recompute derived fields + clip + re-validate


def build_neighbor_index(df: pd.DataFrame, backend: EmbeddingBackend, context_cols: list[str],
                         group_col: str, k: int = 10) -> np.ndarray:
    """Retrieval scoped per `group_col` (e.g. SKU code, or product category).

    A single global k-NN over the whole dataset is O(n^2) and pointless here besides
    slow: `context_cols` already includes the categorical fields that define the
    group, so a row's true nearest neighbours by that embedding are overwhelmingly
    inside its own group anyway. Searching within the group only turns an
    intractable O(n^2) search into O(sum of group_size^2), which is what actually
    ran at FMCG/grocery scale in reports/augmentation/. Singleton groups (no peer
    to interpolate with) fall back to self-index; synthesize_rows adds a small
    numeric jitter for those so they are not exact duplicates (see postprocess).
    """
    texts = row_text(df, context_cols)
    backend.fit(texts)
    emb = backend.encode(texts)
    n = len(df)
    neighbors = np.full((n, k), -1, dtype=np.int64)
    groups = df.reset_index(drop=True).groupby(group_col, sort=False).indices
    for _, local_idx in groups.items():
        local_idx = np.asarray(local_idx)
        g = len(local_idx)
        if g <= 1:
            continue                                    # left as -1 -> self-fallback in synthesize_rows
        kk = min(k, g - 1)
        nn = NearestNeighbors(n_neighbors=kk + 1, metric="euclidean", algorithm="auto")  # embeddings are
        nn.fit(emb[local_idx])                                                           # L2-normalised, so
        _, local_nbrs = nn.kneighbors(emb[local_idx])                                    # euclidean rank == cosine rank
        global_nbrs = local_idx[local_nbrs[:, 1:]]        # drop self-match, map back to global row ids
        neighbors[local_idx, :kk] = global_nbrs
        if kk < k:                                        # pad short rows with self so no -1 leaks through
            neighbors[local_idx, kk:] = local_idx[:, None]
    return neighbors


def synthesize_rows(df: pd.DataFrame, recipe: SynthesisRecipe, backend: EmbeddingBackend, target_n: int,
                    k: int = 10, seed: int = 42, batch_size: int = 500_000) -> pd.DataFrame:
    """Generate `target_n` synthetic rows grounded in `df` (must be a TRAIN split only)."""
    if target_n <= 0:
        return df.iloc[0:0].copy()
    n = len(df)
    neighbors = build_neighbor_index(df, backend, recipe.context_cols, recipe.group_col, k=k)
    rng = np.random.default_rng(seed)
    num = df[recipe.numeric_cols].to_numpy(dtype=float)
    num_std = num.std(axis=0, keepdims=True)
    df_reset = df.reset_index(drop=True)

    chunks = []
    remaining = target_n
    while remaining > 0:
        m = min(batch_size, remaining)
        seed_idx = rng.integers(0, n, size=m)
        nbr_col = rng.integers(0, neighbors.shape[1], size=m)
        nbr_idx = neighbors[seed_idx, nbr_col]
        is_singleton = nbr_idx == seed_idx               # group of 1 -> no real neighbour to interpolate with
        gap = rng.random(m)[:, None]
        num_synth = num[seed_idx] + gap * (num[nbr_idx] - num[seed_idx])
        jitter = rng.normal(0, 0.05, size=(m, num.shape[1])) * num_std   # small jitter so singleton groups
        num_synth = np.where(is_singleton[:, None], num[seed_idx] + jitter, num_synth)  # are not exact dup rows

        cat_source = np.where(rng.random(m) < 0.5, seed_idx, nbr_idx)
        out = pd.DataFrame(num_synth, columns=recipe.numeric_cols)
        for c in recipe.categorical_cols + recipe.inherit_cols:
            out[c] = df_reset[c].to_numpy()[cat_source]
        out["is_synthetic"] = True
        out["synthesis_backend"] = backend.name
        out["synthesis_seed_index"] = seed_idx
        out["synthesis_neighbor_index"] = nbr_idx
        chunks.append(out)
        remaining -= m

    synth = pd.concat(chunks, ignore_index=True)
    return recipe.postprocess(synth)
