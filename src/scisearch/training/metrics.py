"""Information-retrieval metrics — Recall@k, MRR@10, nDCG@10.

Computed over ``(queries, relevant)`` by running a ``search_fn(query, k)`` that
returns ranked document ids (or ``(id, score)`` tuples).
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Set


def _ids(hits) -> List[str]:
    return [h[0] if isinstance(h, (tuple, list)) else h for h in hits]


def evaluate_ir(search_fn: Callable, queries: Dict[str, str], relevant: Dict[str, Set[str]],
                ks: List[int] = (1, 5, 10)) -> Dict:
    max_k = max(max(ks), 10)
    recall = {k: 0.0 for k in ks}
    mrr = 0.0
    ndcg = 0.0
    n = 0
    for qid, query in queries.items():
        rel = relevant.get(qid, set())
        if not rel:
            continue
        n += 1
        ranked = _ids(search_fn(query, max_k))
        for k in ks:
            if any(d in rel for d in ranked[:k]):
                recall[k] += 1.0
        rr = 0.0
        for rank, d in enumerate(ranked[:10], start=1):
            if d in rel:
                rr = 1.0 / rank
                break
        mrr += rr
        dcg = 0.0
        for rank, d in enumerate(ranked[:10], start=1):
            if d in rel:
                dcg += 1.0 / math.log2(rank + 1)
        idcg = 1.0  # single relevant doc per query
        ndcg += dcg / idcg
    n = max(1, n)
    out = {f"recall@{k}": round(recall[k] / n, 4) for k in ks}
    out["mrr@10"] = round(mrr / n, 4)
    out["ndcg@10"] = round(ndcg / n, 4)
    out["n_queries"] = n
    return out


__all__ = ["evaluate_ir"]
