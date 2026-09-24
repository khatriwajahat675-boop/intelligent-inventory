"""Dependency-free SMOTE (Chawla et al., 2002) for the stock-risk classifier.

Rules enforced here and by the callers:
  * SMOTE is applied to the TRAIN split only. Validation/test keep their real
    class balance, otherwise metrics would be measured on synthetic data.
  * Scaling is fitted on train before SMOTE (distance-based method).
  * Not used for demand forecasting (regression / time series) - synthetic
    interpolated rows would break temporal structure.
"""
from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors

# Above this many minority-class rows, an all-pairs distance search (even sklearn's
# batched one) gets slow; cap the neighbour search to a random subsample of the
# minority class per synthetic row so smote() stays fast at the ~20k-row scale
# this project's augmented+class-imbalanced training sets reach. See docs/STATUS.md.
_BRUTE_FORCE_SAFE_N = 8000


def smote(
    X: np.ndarray,
    y: np.ndarray,
    k_neighbors: int = 5,
    sampling_strategy: float = 1.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Oversample every minority class up to `sampling_strategy` x majority size.

    Returns (X_resampled, y_resampled) = originals followed by synthetic rows.
    Classes with a single sample are duplicated (no neighbour to interpolate).
    Neighbour search uses scikit-learn's NearestNeighbors (not a naive O(n^2)
    Python broadcast) so this scales to tens of thousands of minority rows
    without exhausting memory - a plain (n, n, d) broadcast at n=5,000, d=40
    would allocate ~8GB and was observed to OOM the process at that scale.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    if X.ndim != 2 or len(X) != len(y):
        raise ValueError("X must be 2-D and aligned with y")
    if not 0 < sampling_strategy <= 1:
        raise ValueError("sampling_strategy must be in (0, 1]")
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    target = int(round(counts.max() * sampling_strategy))
    new_X, new_y = [X], [y]
    for cls, n in zip(classes, counts):
        need = target - n
        if need <= 0:
            continue
        Xc = X[y == cls]
        if n == 1:
            synth = np.repeat(Xc, need, axis=0)
        elif n <= _BRUTE_FORCE_SAFE_N:
            k = min(k_neighbors, n - 1)
            nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(Xc)
            _, nbrs = nn.kneighbors(Xc)
            nbrs = nbrs[:, 1:]                                    # drop self-match
            base = rng.integers(0, n, size=need)
            pick = nbrs[base, rng.integers(0, k, size=need)]
            gap = rng.random((need, 1))
            synth = Xc[base] + gap * (Xc[pick] - Xc[base])
        else:
            # very large minority class: neighbours drawn from a random subsample
            # per synthetic row rather than one global (n, n) index - still SMOTE
            # in spirit (interpolate toward a nearby same-class point), bounded cost.
            k = min(k_neighbors, _BRUTE_FORCE_SAFE_N - 1)
            base = rng.integers(0, n, size=need)
            synth = np.empty((need, X.shape[1]), dtype=float)
            batch = 2000
            for i in range(0, need, batch):
                b_idx = base[i:i + batch]
                pool = rng.choice(n, size=min(_BRUTE_FORCE_SAFE_N, n), replace=False)
                nn = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(Xc[pool])
                _, local_nbrs = nn.kneighbors(Xc[b_idx])   # pool is a subsample, so no reliable self-match to drop
                pick = pool[local_nbrs[np.arange(len(b_idx)), rng.integers(0, k, len(b_idx))]]
                gap = rng.random((len(b_idx), 1))
                synth[i:i + batch] = Xc[b_idx] + gap * (Xc[pick] - Xc[b_idx])
        new_X.append(synth)
        new_y.append(np.full(need, cls, dtype=y.dtype))
    return np.vstack(new_X), np.concatenate(new_y)
