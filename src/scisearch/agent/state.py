"""Shared state types for the search agent (FSM context + audit records)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class JobStatus(str, Enum):
    PENDING = "pending"
    UNDERSTOOD = "understood"
    RETRIEVED = "retrieved"
    COMPLETED = "completed"
    NO_RESULTS = "no_results"
    FAILED = "failed"


@dataclass
class ToolTrace:
    tool: str
    ok: bool
    latency_ms: float
    summary: str = ""
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"tool": self.tool, "ok": self.ok, "latency_ms": self.latency_ms,
                "summary": self.summary, "error": self.error}


@dataclass
class Decision:
    id: str
    name: str
    branch: str
    score: Optional[float] = None
    detail: str = ""
    llm_used: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "name": self.name, "branch": self.branch,
                "score": self.score, "detail": self.detail, "llm_used": self.llm_used}


@dataclass
class JobState:
    query: str = ""
    status: JobStatus = JobStatus.PENDING
    intent: str = "hybrid"             # keyword | conceptual | hybrid
    filters: Dict[str, Any] = field(default_factory=dict)
    expanded_query: str = ""
    n_candidates: int = 0
    results: List[Dict[str, Any]] = field(default_factory=list)   # ranked papers
    facets: List[Dict[str, Any]] = field(default_factory=list)
    clusters: List[Dict[str, Any]] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    decisions: List[Decision] = field(default_factory=list)
    trace: List[ToolTrace] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    model_versions: Dict[str, str] = field(default_factory=dict)
    _candidates: List[Any] = field(default_factory=list, repr=False)  # [(id, rrf_score)]

    def add_trace(self, t: ToolTrace) -> None:
        self.trace.append(t)

    def add_decision(self, d: Decision) -> None:
        self.decisions.append(d)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query, "status": self.status.value, "intent": self.intent,
            "filters": self.filters, "expanded_query": self.expanded_query,
            "n_candidates": self.n_candidates, "n_results": len(self.results),
            "results": self.results, "facets": self.facets, "clusters": self.clusters,
            "suggestions": self.suggestions,
            "decisions": [d.to_dict() for d in self.decisions],
            "trace": [t.to_dict() for t in self.trace],
            "metrics": self.metrics, "model_versions": self.model_versions,
        }


__all__ = ["JobStatus", "ToolTrace", "Decision", "JobState"]
