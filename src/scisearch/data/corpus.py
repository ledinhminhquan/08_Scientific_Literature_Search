"""Paper corpus loading + normalization.

Loads papers (title + abstract + categories) from a configurable HF dataset, or
falls back to the built-in mini-corpus when the dataset / network is unavailable.
Normalizes every record to a ``Paper`` with a combined ``text`` field for indexing.
``datasets`` is imported lazily.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List

from ..config import AppConfig, DataConfig
from ..logging_utils import get_logger
from .samples import papers as builtin_papers

logger = get_logger(__name__)


@dataclass
class Paper:
    id: str
    title: str
    abstract: str
    categories: List[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return f"{self.title}. {self.abstract}".strip()

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "title": self.title, "abstract": self.abstract,
                "categories": self.categories}


def _split_categories(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(c).strip() for c in value if str(c).strip()]
    return [c for c in re.split(r"[\s,;|]+", str(value)) if c.strip()]


def _from_records(records: List[Dict], dc: DataConfig) -> List[Paper]:
    out: List[Paper] = []
    for i, r in enumerate(records):
        title = str(r.get(dc.title_col, "") or "").strip()
        abstract = str(r.get(dc.abstract_col, "") or "").strip()
        if not title and not abstract:
            continue
        cats = _split_categories(r.get(dc.category_col))
        out.append(Paper(id=f"p{i:06d}", title=title, abstract=abstract[:2000], categories=cats))
    return out


def load_corpus(cfg: AppConfig) -> List[Paper]:
    dc = cfg.data
    if dc.use_hf_corpus:
        try:
            from datasets import load_dataset  # lazy
            ds = load_dataset(dc.hf_corpus, dc.hf_corpus_config or None, split=dc.hf_corpus_split)
            if dc.corpus_limit and len(ds) > dc.corpus_limit:
                ds = ds.select(range(dc.corpus_limit))
            papers = _from_records([dict(r) for r in ds], dc)
            if papers:
                logger.info("Loaded %d papers from %s", len(papers), dc.hf_corpus)
                return papers
        except Exception as exc:
            logger.warning("Could not load HF corpus %s (%s); using built-in mini-corpus.", dc.hf_corpus, exc)
    papers = _from_records(builtin_papers(), dc)
    logger.info("Using built-in mini-corpus (%d papers)", len(papers))
    return papers


def primary_field(categories: List[str]) -> str:
    return categories[0] if categories else "unknown"


__all__ = ["Paper", "load_corpus", "primary_field"]
