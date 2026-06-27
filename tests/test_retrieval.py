"""BM25, corpus/pairs, RRF fusion and the IR metrics."""

from __future__ import annotations

from scisearch.data.corpus import load_corpus
from scisearch.data.pairs import build_pairs
from scisearch.models.bm25 import BM25Retriever, tokenize
from scisearch.search.hybrid import rrf_fuse
from scisearch.training.metrics import evaluate_ir


def test_tokenize_drops_stopwords():
    toks = tokenize("The Transformer is a model for machine translation")
    assert "the" not in toks and "is" not in toks
    assert "transformer" in toks and "translation" in toks


def test_bm25_finds_relevant(cfg):
    papers = load_corpus(cfg)
    corpus = {p.id: p.text for p in papers}
    bm25 = BM25Retriever().index(corpus)
    hits = bm25.search("attention transformer architecture", 5)
    assert hits
    titles = [papers[int(i[1:])].title for i, _ in hits if i.startswith("p")]
    assert any("Attention Is All You Need" in t or "Transformer" in t for t in titles)


def test_build_pairs_leakage_free(cfg):
    papers = load_corpus(cfg)
    pairs = build_pairs(papers, cfg.data)
    assert pairs["n_train_pairs"] > 0
    assert pairs["n_eval_queries"] > 0
    assert pairs["n_corpus"] == len(papers)
    # eval corpus contains every paper; each query maps to exactly one relevant id
    for qid, rel in pairs["relevant"].items():
        assert len(rel) == 1
        assert next(iter(rel)) in pairs["corpus"]


def test_rrf_fuse_combines_rankings():
    a = [("d1", 0.9), ("d2", 0.8), ("d3", 0.1)]
    b = [("d3", 0.95), ("d1", 0.5)]
    fused = rrf_fuse([a, b], k=60, top=5)
    ids = [i for i, _ in fused]
    assert "d1" in ids[:2]          # ranked high in both => top
    assert set(ids) == {"d1", "d2", "d3"}


def test_evaluate_ir_perfect_and_zero():
    queries = {"q1": "alpha", "q2": "beta"}
    relevant = {"q1": {"d1"}, "q2": {"d2"}}
    good = lambda q, k: [("d1", 1.0)] if q == "alpha" else [("d2", 1.0)]
    m = evaluate_ir(good, queries, relevant)
    assert m["recall@1"] == 1.0 and m["mrr@10"] == 1.0 and m["ndcg@10"] == 1.0
    bad = lambda q, k: [("x", 1.0)]
    assert evaluate_ir(bad, queries, relevant)["recall@10"] == 0.0
