"""Build (query, positive paper) training pairs + an IR evaluation set.

For contrastive fine-tuning (MultipleNegativesRankingLoss) we need ``(query,
positive_document)`` pairs; in-batch negatives supply the rest. Queries are
generated from each paper: the **title** (lexical-leaning) and the **first
sentence of the abstract** (concept-leaning, where dense retrieval beats BM25).
A disjoint held-out slice forms the IR eval set ``{query -> relevant paper id}``.
"""

from __future__ import annotations

import random
import re
from typing import Dict, List, Tuple

from ..config import DataConfig
from .corpus import Paper

_SENT = re.compile(r"[^.!?]*[.!?]")


def _first_sentence(text: str) -> str:
    m = _SENT.match(text.strip())
    return (m.group(0) if m else text)[:200].strip()


def _concept_query(p: Paper) -> str:
    """A short conceptual query from the abstract's first sentence (paraphrase-ish)."""
    s = _first_sentence(p.abstract)
    s = re.sub(r"^(we|this paper|in this paper|we propose|we present|we introduce|we show)\b[ ,:]*", "",
               s, flags=re.I).strip()
    return s or p.title


def build_pairs(papers: List[Paper], dc: DataConfig) -> Dict:
    """Return train pairs + an IR eval set (queries/corpus/relevant) with no leakage."""
    rng = random.Random(dc.seed)
    docs = list(papers)
    rng.shuffle(docs)
    n_eval = min(dc.eval_queries, max(1, len(docs) // 5))
    eval_docs = docs[:n_eval]
    train_docs = docs[n_eval:] if len(docs) > n_eval else docs

    # training pairs (query, positive_text). positive = the paper's full text.
    train_pairs: List[Tuple[str, str]] = []
    for p in train_docs:
        queries = [p.title]
        if dc.pairs_per_paper >= 2 and len(p.abstract) > 40:
            queries.append(_concept_query(p))
        for q in queries[: dc.pairs_per_paper]:
            if q.strip():
                train_pairs.append((q.strip(), p.text))
        if len(train_pairs) >= dc.max_train_pairs:
            break

    # IR eval: corpus = ALL papers; query = concept query of held-out docs; relevant = that paper.
    corpus = {p.id: p.text for p in papers}
    queries: Dict[str, str] = {}
    relevant: Dict[str, set] = {}
    for i, p in enumerate(eval_docs):
        qid = f"q{i:05d}"
        queries[qid] = _concept_query(p)
        relevant[qid] = {p.id}

    return {"train_pairs": train_pairs, "corpus": corpus, "queries": queries, "relevant": relevant,
            "n_train_pairs": len(train_pairs), "n_eval_queries": len(queries), "n_corpus": len(corpus)}


__all__ = ["build_pairs"]
