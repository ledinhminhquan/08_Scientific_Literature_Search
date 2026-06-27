"""Agent tools — each operates on the JobState and returns it.

Tools run against the SearchEngine primitives; they work with the TF-IDF fallback
retriever + identity reranker, so the whole pipeline runs offline for tests/CI.
The orchestrator wraps each call with timing/trace.
"""

from __future__ import annotations

from typing import List

from ..config import AppConfig
from ..logging_utils import get_logger
from ..search.expand import detect_filters, expand_query
from ..search.facets import cluster_results, field_facets
from ..search.hybrid import rrf_fuse
from . import policy
from .state import Decision, JobState

logger = get_logger(__name__)


def tool_understand(job: JobState, cfg: AppConfig, *, brain=None) -> JobState:
    _, filters = detect_filters(job.query)
    intent = policy.route_query(job.query, filters, cfg.agent)
    expanded = job.query
    llm_used = False
    if brain is not None and brain.available():
        adv = brain.understand_query(job.query)
        if adv is not None:
            intent, expanded, filters = adv["intent"], adv["expanded_query"], {**filters, **adv["filters"]}
            llm_used = True
    job.intent = intent
    job.filters = filters
    job.expanded_query = expanded
    job.add_decision(Decision("D1", "query_type_routing", intent,
                              detail=f"filters={filters or 'none'}", llm_used=llm_used))
    return job


def _retrieve_once(job: JobState, engine, cfg: AppConfig, query: str):
    k = cfg.search.top_k
    dense = engine.dense_search(query, k) if cfg.search.use_dense else []
    bm25 = engine.bm25_search(query, k) if cfg.search.use_bm25 else []
    # raw confidence signal = best dense (cosine) similarity, else "present" if bm25 hit
    top_raw = dense[0][1] if dense else (0.5 if bm25 else 0.0)
    rankings: List = []
    # bias the fusion by intent (duplicate the preferred ranking => more RRF weight)
    if job.intent == "keyword":
        rankings = [bm25, bm25, dense]
    elif job.intent == "conceptual":
        rankings = [dense, dense, bm25]
    else:
        rankings = [dense, bm25]
    fused = rrf_fuse([r for r in rankings if r], cfg.search.rrf_k, top=cfg.search.top_k)
    return fused, top_raw


def _apply_filters(engine, candidates, filters):
    field = filters.get("field")
    if not field:
        return candidates
    out = [(cid, s) for cid, s in candidates
           if field in (engine.papers[cid].categories if cid in engine.papers else [])]
    return out or candidates          # don't return empty just because of a filter


def tool_retrieve(job: JobState, cfg: AppConfig, *, engine) -> JobState:
    candidates, top_raw = _retrieve_once(job, engine, cfg, job.expanded_query)
    gate = policy.coverage_gate(len(candidates), top_raw, cfg.agent)
    attempts = 0
    while gate["needs_expansion"] and attempts < cfg.agent.max_expand_attempts:
        top_texts = [engine.texts.get(cid, "") for cid, _ in candidates[:3]]
        job.expanded_query = expand_query(job.query, top_texts)
        candidates, top_raw = _retrieve_once(job, engine, cfg, job.expanded_query)
        attempts += 1
        gate = policy.coverage_gate(len(candidates), top_raw, cfg.agent)
    candidates = _apply_filters(engine, candidates, job.filters)
    job._candidates = candidates
    job.n_candidates = len(candidates)
    job.add_decision(Decision("D2", "coverage_gate", "expanded" if attempts else gate["branch"],
                              score=gate["top_score"], detail=f"n={gate['n']}, expand_attempts={attempts}"))
    return job


def tool_rerank(job: JobState, cfg: AppConfig, *, engine) -> JobState:
    cands = job._candidates
    gate = policy.rerank_gate(cands, cfg.agent)
    if gate["rerank"] and cfg.reranker.enabled:
        ids = [cid for cid, _ in cands[: cfg.reranker.top_k_rerank]]
        reranked = engine.rerank(job.expanded_query, ids)
        rest = [(cid, s) for cid, s in cands if cid not in {i for i, _ in reranked}]
        job._candidates = reranked + rest
        job.model_versions["reranker"] = getattr(engine.reranker, "name", "?")
    job.add_decision(Decision("D3", "rerank_gate", gate["branch"], score=gate["margin"],
                              detail=f"reranked={gate['rerank'] and cfg.reranker.enabled}"))
    return job


def tool_explore(job: JobState, cfg: AppConfig, *, engine) -> JobState:
    top = job._candidates[: cfg.search.final_k]
    results = []
    for rank, (cid, score) in enumerate(top):
        p = engine.get(cid)
        if not p:
            continue
        related = engine.related(cid, cfg.search.related_k) if rank < 3 else []
        results.append({"id": cid, "rank": rank + 1, "score": round(float(score), 4),
                        "title": p.title, "abstract": p.abstract[:280], "categories": p.categories,
                        "related": related})
    job.results = results
    job.facets = field_facets(results)
    job.clusters = cluster_results(results, cfg.search.n_clusters, cfg.search.min_cluster_size)
    strat = policy.exploration_strategy(job.clusters, job.facets, cfg.agent)
    job.suggestions = strat["suggestions"]
    job.add_decision(Decision("D4", "exploration_strategy", strat["branch"],
                              detail=f"{strat['n_clusters']} clusters, {len(job.facets)} facets"))
    job.metrics["n_clusters"] = strat["n_clusters"]
    job.model_versions["retriever"] = getattr(engine.dense, "name", "?") + ":" + getattr(engine.dense, "version", "?")
    return job


__all__ = ["tool_understand", "tool_retrieve", "tool_rerank", "tool_explore"]
