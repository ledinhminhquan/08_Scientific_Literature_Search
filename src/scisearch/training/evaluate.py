"""Evaluate the retriever vs baselines on the held-out IR set (Recall/MRR/nDCG).

Compares the main retriever (fine-tuned dense when available, else TF-IDF) against
the **BM25** baseline and, when a fine-tuned model exists, the **zero-shot base**
encoder — the comparison that shows fine-tuning helps.
"""

from __future__ import annotations

import json
from typing import Dict, Optional

from ..config import AppConfig, run_dir
from ..logging_utils import get_logger, utc_stamp
from ..models.bm25 import BM25Retriever
from ..models.model_registry import resolve_latest
from ..models.retriever import DenseRetriever, TfidfRetriever, load_dense_retriever
from ..data.corpus import load_corpus
from ..data.pairs import build_pairs
from . import metrics as M


def _limit_queries(queries: Dict, n: Optional[int]) -> Dict:
    if not n or n >= len(queries):
        return queries
    return dict(list(queries.items())[:n])


logger = get_logger(__name__)


def evaluate(cfg: AppConfig, limit: Optional[int] = None, save: bool = True) -> Dict:
    papers = load_corpus(cfg)
    pairs = build_pairs(papers, cfg.data)
    queries = _limit_queries(pairs["queries"], limit)
    corpus_d, relevant = pairs["corpus"], pairs["relevant"]

    bm25 = BM25Retriever().index(corpus_d)
    result: Dict = {"n_corpus": len(corpus_d), "n_queries": len(queries),
                    "bm25": M.evaluate_ir(lambda q, k: bm25.search(q, k), queries, relevant)}

    dense = load_dense_retriever(cfg.model, prefer="dense")
    dense.build_index(corpus_d)
    result["model_name"] = dense.name
    result["model"] = M.evaluate_ir(lambda q, k: dense.search(q, k), queries, relevant)
    trained = (resolve_latest(cfg.model.output_dir) is not None) and dense.name == "dense"
    result["trained_model"] = trained

    if trained:
        try:
            base = DenseRetriever.from_pretrained(cfg.model.base_model, cfg.model)
            base.build_index(corpus_d)
            result["zero_shot_base"] = M.evaluate_ir(lambda q, k: base.search(q, k), queries, relevant)
        except Exception as exc:
            logger.info("zero-shot base eval skipped (%s)", exc)

    m, b = result["model"], result["bm25"]
    result["summary"] = {
        "retriever": dense.name, "trained_model": trained,
        "model_ndcg@10": m.get("ndcg@10"), "bm25_ndcg@10": b.get("ndcg@10"),
        "model_recall@10": m.get("recall@10"), "bm25_recall@10": b.get("recall@10"),
        "zero_shot_ndcg@10": result.get("zero_shot_base", {}).get("ndcg@10"),
        "beats_bm25": (m.get("ndcg@10", 0) >= b.get("ndcg@10", 0)),
    }
    if save:
        out = run_dir() / "eval"
        out.mkdir(parents=True, exist_ok=True)
        (out / f"eval-{utc_stamp()}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        (out / "latest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        logger.info("Eval saved: %s ndcg@10=%s vs bm25 %s", dense.name, m.get("ndcg@10"), b.get("ndcg@10"))
    return result


__all__ = ["evaluate"]
