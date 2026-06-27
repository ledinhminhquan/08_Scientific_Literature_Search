"""Latency benchmark: end-to-end agent search throughput."""

from __future__ import annotations

import json
import time
from typing import Dict, List

import numpy as np

from ..config import AppConfig, run_dir
from ..logging_utils import get_logger, utc_stamp

logger = get_logger(__name__)

_QUERIES = [
    "attention transformer", "dense passage retrieval", "named entity recognition",
    "machine translation attention", "contrastive sentence embeddings",
    "how to summarize long documents", "question answering reading comprehension",
    "word embeddings word2vec glove",
]


def _pct(xs: List[float]) -> Dict[str, float]:
    a = np.asarray(xs, dtype=np.float64)
    return {"p50": round(float(np.percentile(a, 50)), 2), "p95": round(float(np.percentile(a, 95)), 2),
            "p99": round(float(np.percentile(a, 99)), 2), "mean": round(float(a.mean()), 2)}


def benchmark(cfg: AppConfig, n: int = 30, warmup: int = 3, save: bool = True) -> Dict:
    from ..agent.search_agent import SearchAgent
    agent = SearchAgent(cfg, load_model=True)
    device = "cpu"
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        pass

    qs = (_QUERIES * ((n + warmup) // len(_QUERIES) + 1))[: n + warmup]
    for q in qs[:warmup]:
        agent.search(q, save=False)
    lat: List[float] = []
    for q in qs[warmup:]:
        t0 = time.perf_counter()
        agent.search(q, save=False)
        lat.append((time.perf_counter() - t0) * 1000.0)

    out = {"device": device, "retriever": getattr(agent.engine.dense, "name", "?"),
           "n_papers": len(agent.engine.papers), "query_ms": _pct(lat),
           "throughput_per_s": round(1000.0 / max(0.1, np.mean(lat)), 2), "n": n}
    if save:
        d = run_dir() / "benchmark"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"benchmark-{utc_stamp()}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        (d / "latest.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


__all__ = ["benchmark"]
