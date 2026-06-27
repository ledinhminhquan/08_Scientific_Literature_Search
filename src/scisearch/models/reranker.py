"""Cross-encoder reranker (re-scores top-k candidates) + identity fallback."""

from __future__ import annotations

from typing import List, Optional, Tuple

from ..config import RerankerConfig
from ..logging_utils import get_logger

logger = get_logger(__name__)


class CrossEncoderReranker:
    name = "cross-encoder"

    def __init__(self, model, top_k: int):
        self.model = model
        self.top_k = top_k

    @classmethod
    def load(cls, cfg: RerankerConfig, device: Optional[str] = None) -> "CrossEncoderReranker":
        from sentence_transformers import CrossEncoder  # lazy
        return cls(CrossEncoder(cfg.model, device=device), cfg.top_k_rerank)

    def rerank(self, query: str, candidates: List[Tuple[str, str]]) -> List[Tuple[str, float]]:
        """candidates = [(id, text)] -> [(id, score)] sorted desc."""
        cand = candidates[: self.top_k]
        if not cand:
            return []
        scores = self.model.predict([(query, text) for _, text in cand])
        ranked = sorted(zip([c[0] for c in cand], [float(s) for s in scores]), key=lambda x: -x[1])
        return ranked


class IdentityReranker:
    name = "identity"

    def __init__(self, top_k: int = 30):
        self.top_k = top_k

    def rerank(self, query: str, candidates: List[Tuple[str, str]]) -> List[Tuple[str, float]]:
        # keep the incoming order, assign descending pseudo-scores
        return [(cid, 1.0 - i / max(1, len(candidates))) for i, (cid, _) in enumerate(candidates[: self.top_k])]


def load_reranker(cfg: RerankerConfig, device: Optional[str] = None):
    if cfg.enabled:
        try:
            return CrossEncoderReranker.load(cfg, device=device)
        except Exception as exc:
            logger.info("Cross-encoder reranker unavailable (%s); using identity reranker.", exc)
    return IdentityReranker(cfg.top_k_rerank)


__all__ = ["CrossEncoderReranker", "IdentityReranker", "load_reranker"]
