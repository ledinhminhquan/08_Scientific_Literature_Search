"""Reciprocal Rank Fusion (RRF) for combining dense + BM25 rankings.

RRF score for a document d = sum over rankings r of 1 / (k + rank_r(d)).
It needs only ranks (not comparable raw scores), so it robustly fuses a dense
cosine ranking with a BM25 ranking.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple


def rrf_fuse(rankings: List[List[Tuple[str, float]]], k: int = 60,
            top: int = 100) -> List[Tuple[str, float]]:
    scores: Dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, (doc_id, _score) in enumerate(ranking):
            scores[doc_id] += 1.0 / (k + rank + 1)
    fused = sorted(scores.items(), key=lambda x: -x[1])
    return fused[:top]


def minmax_norm(hits: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
    if not hits:
        return hits
    vals = [s for _, s in hits]
    lo, hi = min(vals), max(vals)
    rng = (hi - lo) or 1.0
    return [(i, (s - lo) / rng) for i, s in hits]


__all__ = ["rrf_fuse", "minmax_norm"]
