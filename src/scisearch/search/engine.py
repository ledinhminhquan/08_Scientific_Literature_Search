"""The search engine: holds the corpus + dense retriever + BM25 + reranker and
exposes the primitives the agent orchestrates (dense/bm25/rerank/related)."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from ..config import AppConfig
from ..logging_utils import get_logger
from ..models.bm25 import BM25Retriever
from ..data.corpus import Paper, load_corpus

logger = get_logger(__name__)


class SearchEngine:
    def __init__(self, cfg: AppConfig, dense_retriever, reranker):
        self.cfg = cfg
        self.dense = dense_retriever
        self.reranker = reranker
        self.bm25 = BM25Retriever()
        self.papers: Dict[str, Paper] = {}
        self.texts: Dict[str, str] = {}

    @classmethod
    def from_config(cls, cfg: AppConfig, *, load_model: bool = True,
                    papers: Optional[List[Paper]] = None) -> "SearchEngine":
        from ..models.retriever import load_dense_retriever
        from ..models.reranker import load_reranker
        dense = load_dense_retriever(cfg.model, prefer="dense" if load_model else "tfidf")
        reranker = load_reranker(cfg.reranker) if load_model else __import__(
            "scisearch.models.reranker", fromlist=["IdentityReranker"]).IdentityReranker(cfg.reranker.top_k_rerank)
        eng = cls(cfg, dense, reranker)
        eng.build(papers if papers is not None else load_corpus(cfg))
        return eng

    def build(self, papers: List[Paper]) -> "SearchEngine":
        self.papers = {p.id: p for p in papers}
        self.texts = {p.id: p.text for p in papers}
        self.bm25.index(self.texts)
        try:
            self.dense.build_index(self.texts)
        except Exception as exc:
            logger.warning("dense index build failed (%s)", exc)
        logger.info("Search engine built: %d papers (dense=%s)", len(papers), getattr(self.dense, "name", "?"))
        return self

    # ---- primitives -------------------------------------------------------
    def dense_search(self, query: str, k: int) -> List[Tuple[str, float]]:
        try:
            return self.dense.search(query, k)
        except Exception as exc:
            logger.info("dense_search failed (%s)", exc)
            return []

    def bm25_search(self, query: str, k: int) -> List[Tuple[str, float]]:
        return self.bm25.search(query, k)

    def rerank(self, query: str, ids: List[str]) -> List[Tuple[str, float]]:
        return self.reranker.rerank(query, [(i, self.texts.get(i, "")) for i in ids])

    def related(self, paper_id: str, k: int) -> List[str]:
        p = self.papers.get(paper_id)
        if not p:
            return []
        hits = self.dense_search(p.text, k + 1) or self.bm25_search(p.text, k + 1)
        return [i for i, _ in hits if i != paper_id][:k]

    def meta(self, ids: List[str]) -> List[Dict]:
        return [self.papers[i].to_dict() for i in ids if i in self.papers]

    def get(self, paper_id: str) -> Optional[Paper]:
        return self.papers.get(paper_id)


__all__ = ["SearchEngine"]
