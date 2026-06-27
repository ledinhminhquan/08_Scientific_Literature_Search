"""Retrieval error analysis: per-query wins/losses of the retriever vs BM25.

For each held-out query, compute the rank of the relevant paper under the dense
retriever and under BM25 — surfacing where the (trained) dense model wins, where
BM25 wins, and the queries neither solves (failure cases).
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional

from ..config import AppConfig, run_dir
from ..logging_utils import get_logger, utc_stamp
from ..models.bm25 import BM25Retriever
from ..models.retriever import load_dense_retriever
from ..data.corpus import load_corpus
from ..data.pairs import build_pairs


def _rank_of(hits, relevant: set) -> int:
    for i, h in enumerate(hits, start=1):
        if (h[0] if isinstance(h, (tuple, list)) else h) in relevant:
            return i
    return 0  # not found in the returned list


logger = get_logger(__name__)


def error_analysis(cfg: AppConfig, limit: Optional[int] = None, save: bool = True) -> Dict:
    papers = load_corpus(cfg)
    pairs = build_pairs(papers, cfg.data)
    queries = dict(list(pairs["queries"].items())[:limit]) if limit else pairs["queries"]
    corpus_d, relevant = pairs["corpus"], pairs["relevant"]

    bm25 = BM25Retriever().index(corpus_d)
    dense = load_dense_retriever(cfg.model, prefer="dense")
    dense.build_index(corpus_d)

    dense_wins = bm25_wins = ties = both_fail = 0
    examples: List[Dict] = []
    for qid, q in queries.items():
        rel = relevant.get(qid, set())
        dr = _rank_of(dense.search(q, 10), rel)
        br = _rank_of(bm25.search(q, 10), rel)
        d_ok = dr if dr else 999
        b_ok = br if br else 999
        if d_ok == 999 and b_ok == 999:
            both_fail += 1
            if len(examples) < 10:
                examples.append({"query": q[:120], "dense_rank": dr, "bm25_rank": br, "case": "both_fail"})
        elif d_ok < b_ok:
            dense_wins += 1
        elif b_ok < d_ok:
            bm25_wins += 1
            if len(examples) < 10:
                examples.append({"query": q[:120], "dense_rank": dr, "bm25_rank": br, "case": "bm25_wins"})
        else:
            ties += 1

    result = {
        "retriever": dense.name, "n_queries": len(queries),
        "dense_wins": dense_wins, "bm25_wins": bm25_wins, "ties": ties, "both_fail": both_fail,
        "examples": examples,
    }
    if save:
        d = run_dir() / "error_analysis"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"errors-{utc_stamp()}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        (d / "latest.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("error analysis: dense_wins=%d bm25_wins=%d both_fail=%d", dense_wins, bm25_wins, both_fail)
    return result


__all__ = ["error_analysis"]
