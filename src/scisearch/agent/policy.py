"""Decision-point logic for the search agent (pure, testable, no model deps).

Four explicit decision points act on intermediate outputs:
* **D1** query-type routing (keyword vs conceptual vs hybrid; metadata filters),
* **D2** retrieval-coverage gate (expand the query when results are thin),
* **D3** rerank gate (rerank only when the head of the ranking is ambiguous),
* **D4** exploration / presentation strategy (cluster + suggest broaden/related).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from ..config import AgentConfig
from ..models.bm25 import tokenize


# ── D1 ───────────────────────────────────────────────────────────────────────
def route_query(query: str, filters: Dict, cfg: AgentConfig) -> str:
    n_words = len(tokenize(query))
    if n_words <= cfg.short_query_words and not any(k in query for k in ("how", "what", "why")):
        return "keyword"
    if n_words >= 8 or any(w in query.lower() for w in ("how", "what", "approach", "method", "using")):
        return "conceptual"
    return "hybrid"


# ── D2 ───────────────────────────────────────────────────────────────────────
def coverage_gate(n_candidates: int, top_raw_score: float, cfg: AgentConfig) -> Dict:
    """``top_raw_score`` = best raw retrieval similarity (cosine), NOT the RRF score."""
    needs = n_candidates < cfg.min_results or top_raw_score < cfg.min_top_score
    return {"needs_expansion": needs, "branch": "expand" if needs else "ok",
            "n": n_candidates, "top_score": round(float(top_raw_score), 4)}


# ── D3 ───────────────────────────────────────────────────────────────────────
def rerank_gate(candidates: List[Tuple[str, float]], cfg: AgentConfig) -> Dict:
    if len(candidates) < 3:
        return {"rerank": False, "branch": "skip", "margin": None}
    top = candidates[0][1]
    second = candidates[1][1]
    margin = abs(top - second) / (abs(top) or 1.0)
    rerank = margin <= cfg.rerank_margin            # head is close/ambiguous => rerank
    return {"rerank": rerank, "branch": "rerank" if rerank else "skip", "margin": round(margin, 4)}


# ── D4 ───────────────────────────────────────────────────────────────────────
def exploration_strategy(clusters: List[Dict], facets: List[Dict], cfg: AgentConfig) -> Dict:
    suggestions: List[str] = []
    n_clusters = len([c for c in clusters if c.get("paper_ids")])
    if n_clusters >= cfg.diversify_min_clusters:
        suggestions.append(f"narrow by topic ({n_clusters} sub-topics found)")
    if len(facets) >= 2:
        suggestions.append(f"filter by field ({', '.join(f['field'] for f in facets[:3])})")
    suggestions.append("see related papers for any result")
    branch = "diverse" if n_clusters >= cfg.diversify_min_clusters else "focused"
    return {"branch": branch, "suggestions": suggestions, "n_clusters": n_clusters}


__all__ = ["route_query", "coverage_gate", "rerank_gate", "exploration_strategy"]
