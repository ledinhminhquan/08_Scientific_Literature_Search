"""Lightweight hyperparameter search for the retriever (learning-rate grid)."""

from __future__ import annotations

import copy
import json
from typing import Dict, List, Optional

from ..config import AppConfig, run_dir
from ..logging_utils import get_logger, utc_stamp

logger = get_logger(__name__)

_DEFAULT_GRID: List[float] = [1.0e-5, 2.0e-5, 5.0e-5]


def tune_retriever(cfg: AppConfig, n_trials: int = 3, limit: int = 8000,
                   epochs: int = 1, grid: Optional[List[float]] = None) -> Dict:
    from .train_retriever import train_retriever

    lrs = (grid or _DEFAULT_GRID)[:n_trials]
    trials, best = [], None
    for lr in lrs:
        tcfg = copy.deepcopy(cfg)
        tcfg.model.learning_rate = lr
        tcfg.model.num_train_epochs = epochs
        tcfg.model.output_subdir = f"retriever_tune/lr_{lr:.0e}"
        try:
            res = train_retriever(tcfg, limit=limit, resume=False)
            ndcg = res["metrics"].get("papers_cosine_ndcg@10", res["metrics"].get("eval_papers_cosine_ndcg@10", 0.0))
        except Exception as exc:
            logger.warning("trial lr=%s failed: %s", lr, exc)
            ndcg = 0.0
        rec = {"learning_rate": lr, "ndcg@10": ndcg}
        trials.append(rec)
        if best is None or ndcg > best["ndcg@10"]:
            best = rec
        logger.info("trial lr=%.0e -> ndcg@10=%.4f", lr, ndcg)

    out = {"best": best, "trials": trials}
    d = run_dir() / "tune"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"tune-{utc_stamp()}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    (d / "best.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


__all__ = ["tune_retriever"]
