"""Query expansion (for the D2 coverage gate) + related-paper helpers.

Expansion uses a small domain abbreviation map plus pseudo-relevance feedback
(adds high-signal terms from the top retrieved papers). Related-paper retrieval
is dense nearest-neighbours of a paper, optionally filtered by shared field.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List, Tuple

from ..models.bm25 import tokenize

_ABBR = {
    "nlp": "natural language processing", "llm": "large language model",
    "ml": "machine learning", "cv": "computer vision", "ir": "information retrieval",
    "rl": "reinforcement learning", "qa": "question answering", "ner": "named entity recognition",
    "mt": "machine translation", "asr": "automatic speech recognition", "gan": "generative adversarial network",
    "rag": "retrieval augmented generation", "kg": "knowledge graph", "cnn": "convolutional neural network",
    "rnn": "recurrent neural network", "gnn": "graph neural network",
}


def expand_query(query: str, top_texts: List[str] = None, n_terms: int = 4) -> str:
    """Return an expanded query string (original + abbreviation expansions + PRF terms)."""
    extra: List[str] = []
    for w in tokenize(query):
        if w in _ABBR:
            extra.append(_ABBR[w])
    if top_texts:
        qset = set(tokenize(query))
        counter: Counter = Counter()
        for t in top_texts[:3]:
            for tok in tokenize(t):
                if tok not in qset and len(tok) > 3:
                    counter[tok] += 1
        extra += [t for t, _ in counter.most_common(n_terms)]
    expanded = (query + " " + " ".join(dict.fromkeys(extra))).strip()
    return re.sub(r"\s+", " ", expanded)


def detect_filters(query: str) -> Tuple[str, Dict]:
    """Extract simple metadata filters (year, field) from the query; return (clean_query, filters)."""
    filters: Dict = {}
    m = re.search(r"\b(19|20)\d{2}\b", query)
    if m:
        filters["year"] = int(m.group(0))
    fm = re.search(r"\b(cs\.[A-Z]{2}|stat\.ML|eess\.[A-Z]{2})\b", query)
    if fm:
        filters["field"] = fm.group(0)
    after = re.search(r"\b(since|after)\s+((19|20)\d{2})\b", query, re.I)
    if after:
        filters["since"] = int(after.group(2))
    return query, filters


__all__ = ["expand_query", "detect_filters"]
