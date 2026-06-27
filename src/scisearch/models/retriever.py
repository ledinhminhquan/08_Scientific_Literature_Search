"""Dense retriever (the TRAINED core) + a TF-IDF fallback.

``DenseRetriever`` wraps a sentence-transformers bi-encoder (the fine-tuned model
when present, else the base). ``TfidfRetriever`` is the dependency-light fallback
(sklearn) used when sentence-transformers / torch are unavailable, so hybrid
search runs fully offline. ``load_dense_retriever`` picks the best available.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..config import ModelConfig
from ..logging_utils import get_logger
from .model_registry import resolve_latest
from .vector_store import VectorIndex

logger = get_logger(__name__)


class DenseRetriever:
    name = "dense"

    def __init__(self, model, cfg: ModelConfig, version: str = "dense-1.0"):
        self.model = model
        self.cfg = cfg
        self.version = version
        self.index = VectorIndex()

    @classmethod
    def from_pretrained(cls, model_path: str, cfg: ModelConfig, device: Optional[str] = None) -> "DenseRetriever":
        from sentence_transformers import SentenceTransformer  # lazy
        model = SentenceTransformer(model_path, device=device)
        model.max_seq_length = cfg.max_seq_length
        return cls(model, cfg, version=_read_version(model_path))

    def encode(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        if is_query and self.cfg.query_instruction:
            texts = [self.cfg.query_instruction + t for t in texts]
        return np.asarray(self.model.encode(texts, normalize_embeddings=self.cfg.normalize_embeddings,
                                            convert_to_numpy=True, show_progress_bar=False), dtype=np.float32)

    def build_index(self, papers: Dict[str, str]) -> "DenseRetriever":
        ids = list(papers.keys())
        emb = self.encode([papers[i] for i in ids], is_query=False)
        self.index.build(ids, emb)
        return self

    def search(self, query: str, k: int = 50) -> List[Tuple[str, float]]:
        return self.index.search(self.encode([query], is_query=True)[0], k)


class TfidfRetriever:
    name = "tfidf"
    version = "tfidf-1.0"

    def __init__(self, cfg: Optional[ModelConfig] = None):
        from sklearn.feature_extraction.text import TfidfVectorizer  # lazy (sklearn is core)
        self.vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=50000, sublinear_tf=True)
        self.ids: List[str] = []
        self.matrix = None

    def build_index(self, papers: Dict[str, str]) -> "TfidfRetriever":
        self.ids = list(papers.keys())
        self.matrix = self.vec.fit_transform([papers[i] for i in self.ids])
        return self

    def search(self, query: str, k: int = 50) -> List[Tuple[str, float]]:
        from sklearn.metrics.pairwise import linear_kernel
        q = self.vec.transform([query])
        sims = linear_kernel(q, self.matrix)[0]
        k = min(k, len(self.ids))
        if k == 0:
            return []
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        return [(self.ids[i], float(sims[i])) for i in top if sims[i] > 0]


def _read_version(model_path: str) -> str:
    meta = Path(model_path) / "model_meta.json"
    if meta.exists():
        try:
            import json
            return json.loads(meta.read_text(encoding="utf-8")).get("version", "dense-1.0")
        except Exception:
            pass
    return "dense-1.0"


def load_dense_retriever(cfg: ModelConfig, *, prefer: str = "dense", device: Optional[str] = None):
    """Return a dense retriever (fine-tuned > base ST), else the TF-IDF fallback."""
    if prefer != "tfidf":
        latest = resolve_latest(cfg.output_dir)
        target = str(latest) if latest is not None else cfg.base_model
        try:
            return DenseRetriever.from_pretrained(target, cfg, device=device)
        except Exception as exc:
            logger.info("Dense retriever unavailable (%s); using TF-IDF fallback.", exc)
    try:
        return TfidfRetriever(cfg)
    except Exception as exc:
        logger.warning("TF-IDF fallback unavailable (%s)", exc)
        raise


__all__ = ["DenseRetriever", "TfidfRetriever", "load_dense_retriever"]
