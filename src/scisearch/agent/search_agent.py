"""The search agent — a deterministic FSM that turns a query into ranked, faceted,
clustered, explorable results.

    understand (D1) -> retrieve+RRF (D2 coverage gate -> expand) -> rerank (D3 gate)
        -> explore: facets + clusters + related (D4 strategy)

Holds a SearchEngine (dense + BM25 + reranker over the corpus). Runs fully offline
(TF-IDF retriever + identity reranker) and upgrades when a fine-tuned retriever +
cross-encoder are present. Every step is timed and traced.
"""

from __future__ import annotations

import time
from typing import Callable, Optional

from ..config import AppConfig, ensure_dirs
from ..logging_utils import JsonlLogger, get_logger
from ..search.engine import SearchEngine
from . import tools
from .llm_orchestrator import LLMBrain
from .state import JobState, JobStatus, ToolTrace

logger = get_logger(__name__)


class SearchAgent:
    def __init__(self, cfg: Optional[AppConfig] = None, *, load_model: bool = True,
                 engine: Optional[SearchEngine] = None):
        self.cfg = cfg or AppConfig()
        self.engine = engine or SearchEngine.from_config(self.cfg, load_model=load_model)
        self.brain = LLMBrain(self.cfg.agent)
        ensure_dirs()
        self._log = JsonlLogger(self.cfg.serving.query_log_path) if self.cfg.serving.log_queries else None

    def _step(self, job: JobState, name: str, fn: Callable[[], JobState], summary: str = "") -> JobState:
        t0 = time.perf_counter()
        try:
            job = fn()
            ok, err = True, None
        except Exception as exc:
            logger.warning("tool %s failed: %s", name, exc)
            ok, err = False, str(exc)
        job.add_trace(ToolTrace(tool=name, ok=ok, latency_ms=round((time.perf_counter() - t0) * 1000, 2),
                                summary=summary or name, error=err))
        return job

    def search(self, query: str, save: bool = True) -> JobState:
        job = JobState(query=query.strip())
        t0 = time.perf_counter()
        if not job.query:
            job.status = JobStatus.FAILED
            return job
        job = self._step(job, "understand", lambda: tools.tool_understand(job, self.cfg, brain=self.brain),
                         summary="query understanding (D1)")
        job = self._step(job, "retrieve", lambda: tools.tool_retrieve(job, self.cfg, engine=self.engine),
                         summary="hybrid retrieve + RRF (D2)")
        job = self._step(job, "rerank", lambda: tools.tool_rerank(job, self.cfg, engine=self.engine),
                         summary="rerank gate (D3)")
        job = self._step(job, "explore", lambda: tools.tool_explore(job, self.cfg, engine=self.engine),
                         summary="facets + clusters + related (D4)")
        job.metrics["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        job.status = JobStatus.COMPLETED if job.results else JobStatus.NO_RESULTS
        if save and self._log is not None:
            try:
                self._log.log("query", query=job.query, intent=job.intent, status=job.status.value,
                              n_results=len(job.results), metrics=job.metrics)
            except Exception:
                pass
        return job


_AGENT: Optional[SearchAgent] = None


def get_agent(cfg: Optional[AppConfig] = None, **kwargs) -> SearchAgent:
    global _AGENT
    if _AGENT is None:
        _AGENT = SearchAgent(cfg, **kwargs)
    return _AGENT


__all__ = ["SearchAgent", "get_agent"]
