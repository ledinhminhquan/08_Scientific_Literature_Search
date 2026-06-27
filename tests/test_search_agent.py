"""Search engine + the end-to-end agent (TF-IDF + BM25 + identity reranker, offline)."""

from __future__ import annotations

from scisearch.agent.policy import coverage_gate, exploration_strategy, rerank_gate, route_query
from scisearch.agent.search_agent import SearchAgent
from scisearch.config import AppConfig
from scisearch.search.engine import SearchEngine
from scisearch.search.facets import cluster_results, field_facets


def _cfg():
    c = AppConfig()
    c.data.use_hf_corpus = False
    return c


def test_engine_search_primitives():
    eng = SearchEngine.from_config(_cfg(), load_model=False)
    assert len(eng.papers) >= 30
    assert eng.dense.name == "tfidf"
    assert eng.bm25_search("transformer attention", 5)
    assert eng.dense_search("dense retrieval", 5)
    rel = eng.related(next(iter(eng.papers)), 3)
    assert isinstance(rel, list)


def test_agent_search_end_to_end():
    agent = SearchAgent(_cfg(), load_model=False)
    job = agent.search("dense passage retrieval for question answering", save=False)
    sd = job.to_dict()
    assert sd["status"] == "completed"
    assert sd["n_results"] >= 1
    assert {d["id"] for d in sd["decisions"]} == {"D1", "D2", "D3", "D4"}
    assert all(t["ok"] for t in sd["trace"])
    assert sd["facets"]
    # the top result should be relevant to the query
    assert any("retrieval" in r["title"].lower() or "question" in r["title"].lower()
               for r in sd["results"][:5])


def test_decision_helpers():
    cfg = AppConfig().agent
    assert route_query("BERT", {}, cfg) == "keyword"
    assert route_query("how do transformers learn long range dependencies", {}, cfg) == "conceptual"
    assert coverage_gate(1, 0.1, cfg)["needs_expansion"] is True
    assert coverage_gate(10, 0.6, cfg)["needs_expansion"] is False
    assert rerank_gate([("a", 0.10), ("b", 0.099), ("c", 0.05)], cfg)["rerank"] is True
    assert rerank_gate([("a", 0.9), ("b", 0.1), ("c", 0.05)], cfg)["rerank"] is False


def test_facets_and_clusters():
    papers = [{"id": "p1", "title": "A", "abstract": "neural machine translation attention", "categories": ["cs.CL"]},
              {"id": "p2", "title": "B", "abstract": "image classification convolutional network", "categories": ["cs.CV"]},
              {"id": "p3", "title": "C", "abstract": "machine translation sequence model", "categories": ["cs.CL"]}]
    facets = field_facets(papers)
    assert any(f["field"] == "cs.CL" and f["count"] == 2 for f in facets)
    clusters = cluster_results(papers, n_clusters=2, min_size=1)
    assert clusters and all("paper_ids" in c for c in clusters)
