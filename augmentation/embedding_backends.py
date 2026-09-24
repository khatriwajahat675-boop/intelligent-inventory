"""Pluggable text-embedding backends for retrieval-grounded ("RAG-style") synthesis.

Why not literal RAG: retrieval-augmented *generation* is an LLM pattern for
answering questions from retrieved documents. There is no text corpus and no
question here. What we borrow from it - and what this module implements - is
the *retrieval-grounded* half of the pattern: embed each row's categorical
context, retrieve its nearest real neighbours, and generate new rows only
inside that neighbourhood so every synthetic row stays anchored to a
real, plausible combination instead of being sampled from thin air.

Two interchangeable backends implement the same interface:

  * HFSentenceTransformerBackend - the real model, downloaded from the
    HuggingFace Hub. Requires `pip install sentence-transformers` and
    network access to huggingface.co.
  * TfidfEmbeddingBackend - a scikit-learn-only offline fallback (TF-IDF +
    truncated SVD). It lets the entire retrieval + synthesis ALGORITHM be
    executed and unit-tested without internet access, which is what this
    sandbox is limited to (see docs/AUGMENTATION.md). Swapping to the HF
    backend on a machine with internet access changes zero downstream code.
"""
from __future__ import annotations

import abc
import time

import numpy as np

# Candidates evaluated by augmentation/model_selection.py - small, CPU-friendly
# sentence embedding models suitable for short categorical/text row descriptions.
# (Larger models like all-mpnet-base-v2 are included for completeness but are
# ~5x slower per row; on a CPU-only machine with 2M+ rows, MiniLM-class models
# are the practical choice, which the selection harness will confirm empirically.)
CANDIDATE_HF_MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",     # 384-dim, ~80MB, fast - default recommendation
    "sentence-transformers/all-MiniLM-L12-v2",    # 384-dim, deeper, ~2x slower than L6
    "sentence-transformers/paraphrase-MiniLM-L3-v2",  # 384-dim, smallest/fastest, lower quality
    "BAAI/bge-small-en-v1.5",                     # 384-dim, strong general retrieval benchmark
    "sentence-transformers/all-mpnet-base-v2",    # 768-dim, highest quality, slowest
]


class EmbeddingBackend(abc.ABC):
    name: str

    @abc.abstractmethod
    def fit(self, texts: list[str]) -> "EmbeddingBackend":
        """Prepare the backend (no-op for pretrained HF models)."""

    @abc.abstractmethod
    def encode(self, texts: list[str]) -> np.ndarray:
        """Return an (n, d) L2-normalised embedding matrix."""


class HFSentenceTransformerBackend(EmbeddingBackend):
    """Real backend. Lazily imported so its absence never breaks the offline path."""

    def __init__(self, model_name: str):
        self.name = model_name
        self._model = None

    def fit(self, texts: list[str]) -> "HFSentenceTransformerBackend":
        return self  # pretrained - nothing to fit

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers is not installed / huggingface.co is unreachable from this "
                    "environment. Run: pip install sentence-transformers  (needs internet access to "
                    "download the model once; it is then cached locally)."
                ) from exc
            self._model = SentenceTransformer(self.name)
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        vecs = self._load().encode(list(texts), batch_size=256, show_progress_bar=False,
                                   normalize_embeddings=True, convert_to_numpy=True)
        return np.asarray(vecs, dtype=np.float32)


class TfidfEmbeddingBackend(EmbeddingBackend):
    """Offline fallback: TF-IDF (char+word n-grams) -> truncated SVD -> L2 normalise.

    Not a HuggingFace model and not claimed to be one - it exists solely so the
    retrieval-grounded synthesis algorithm can be run and validated without
    network access. It is what actually executed the runs in reports/augmentation/.
    """

    def __init__(self, max_features: int = 512, n_components: int = 64, seed: int = 42):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.name = "tfidf-svd-offline-fallback"
        self._vec = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), min_df=1)
        self._svd = TruncatedSVD(n_components=n_components, random_state=seed)
        self._fitted = False

    def fit(self, texts: list[str]) -> "TfidfEmbeddingBackend":
        X = self._vec.fit_transform(texts)
        self._svd.fit(X)
        self._fitted = True
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("call fit() before encode()")
        X = self._vec.transform(texts)
        emb = self._svd.transform(X).astype(np.float32)
        norm = np.linalg.norm(emb, axis=1, keepdims=True)
        return emb / np.clip(norm, 1e-9, None)


def build_backend(name: str) -> EmbeddingBackend:
    """`name == "tfidf"` -> offline fallback; anything else is treated as a HF model id."""
    return TfidfEmbeddingBackend() if name == "tfidf" else HFSentenceTransformerBackend(name)


def timed_encode(backend: EmbeddingBackend, texts: list[str]) -> tuple[np.ndarray, float]:
    t0 = time.perf_counter()
    backend.fit(texts)
    emb = backend.encode(texts)
    return emb, time.perf_counter() - t0
