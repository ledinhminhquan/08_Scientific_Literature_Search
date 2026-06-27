"""Data preparation entrypoint.

Loads the paper corpus (HF dataset or built-in fallback), builds the (query,
paper) training pairs + IR eval set, and reports counts. The HF ``datasets``
library caches downloads, so subsequent runs are fast; everything degrades to the
built-in mini-corpus offline.
"""

from __future__ import annotations

import json
from typing import Dict

from ..config import AppConfig, data_dir
from ..logging_utils import get_logger
from .corpus import load_corpus
from .pairs import build_pairs

logger = get_logger(__name__)


def prepare(cfg: AppConfig) -> Dict:
    papers = load_corpus(cfg)
    pairs = build_pairs(papers, cfg.data)
    out = data_dir() / "corpus"
    out.mkdir(parents=True, exist_ok=True)
    sample = [p.to_dict() for p in papers[:5]]
    (out / "sample_papers.json").write_text(json.dumps(sample, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "signature.json").write_text(json.dumps(
        {"n_papers": len(papers), "hf_corpus": cfg.data.hf_corpus if cfg.data.use_hf_corpus else "built-in",
         "n_train_pairs": pairs["n_train_pairs"], "n_eval_queries": pairs["n_eval_queries"]}, indent=2), encoding="utf-8")
    return {"task": "corpus", "n_papers": len(papers), "n_train_pairs": pairs["n_train_pairs"],
            "n_eval_queries": pairs["n_eval_queries"], "source": cfg.data.hf_corpus if cfg.data.use_hf_corpus else "built-in"}


def download_all(cfg: AppConfig) -> Dict:
    return {"corpus": prepare(cfg)}


__all__ = ["prepare", "download_all"]
