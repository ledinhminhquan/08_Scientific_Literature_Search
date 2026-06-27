"""Exploration: field-of-study facets + topic clustering of a result set.

Facets count primary arXiv categories (mapped to readable field names). Topic
clusters group results by content (KMeans over TF-IDF) and label each cluster with
its top distinctive terms — turning a flat ranked list into an explorable map.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List

from ..logging_utils import get_logger

logger = get_logger(__name__)

# readable names for common arXiv categories (facet labels)
ARXIV_CAT_NAMES = {
    "cs.CL": "Computation & Language (NLP)", "cs.LG": "Machine Learning", "cs.AI": "Artificial Intelligence",
    "cs.CV": "Computer Vision", "cs.IR": "Information Retrieval", "cs.NE": "Neural & Evolutionary Computing",
    "cs.DL": "Digital Libraries", "stat.ML": "Statistics — ML", "cs.SD": "Sound", "eess.AS": "Audio Processing",
    "cs.RO": "Robotics", "cs.HC": "Human-Computer Interaction", "math.OC": "Optimization",
}


def field_name(cat: str) -> str:
    return ARXIV_CAT_NAMES.get(cat, cat)


def field_facets(papers: List[Dict], top_n: int = 8) -> List[Dict]:
    counter: Counter = Counter()
    for p in papers:
        for c in (p.get("categories") or [])[:1] or ["unknown"]:
            counter[c] += 1
    return [{"field": c, "name": field_name(c), "count": n} for c, n in counter.most_common(top_n)]


def cluster_results(papers: List[Dict], n_clusters: int = 5, min_size: int = 2) -> List[Dict]:
    """KMeans over TF-IDF of result texts; label clusters by top terms."""
    texts = [f"{p.get('title','')}. {p.get('abstract','')}" for p in papers]
    if len(texts) < max(2, min_size):
        return [{"label": "all results", "terms": [], "paper_ids": [p["id"] for p in papers]}]
    try:
        import numpy as np
        from sklearn.cluster import KMeans
        from sklearn.feature_extraction.text import TfidfVectorizer
        k = max(1, min(n_clusters, len(texts) // min_size))
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=2000)
        X = vec.fit_transform(texts)
        km = KMeans(n_clusters=k, n_init=4, random_state=42).fit(X)
        terms = vec.get_feature_names_out()
        clusters: List[Dict] = []
        for ci in range(k):
            members = [i for i, lab in enumerate(km.labels_) if lab == ci]
            if not members:
                continue
            centroid = km.cluster_centers_[ci]
            top_terms = [terms[t] for t in centroid.argsort()[::-1][:4]]
            clusters.append({"label": ", ".join(top_terms[:3]), "terms": top_terms,
                             "paper_ids": [papers[i]["id"] for i in members]})
        clusters.sort(key=lambda c: -len(c["paper_ids"]))
        return clusters
    except Exception as exc:
        logger.info("clustering skipped (%s)", exc)
        return [{"label": "all results", "terms": [], "paper_ids": [p["id"] for p in papers]}]


__all__ = ["field_facets", "cluster_results", "field_name", "ARXIV_CAT_NAMES"]
