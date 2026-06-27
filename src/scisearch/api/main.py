"""FastAPI service for the Scientific Literature Search system.

Endpoints
---------
* ``GET  /healthz`` / ``GET /readyz`` / ``GET /version``
* ``POST /search``        – query -> ranked + faceted + clustered results
* ``GET  /related/{id}``  – related papers for a result
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

from .. import __version__
from ..logging_utils import get_logger
from .dependencies import get_agent, get_config
from .schemas import HealthResponse, PaperResult, SearchRequest, SearchResponse

logger = get_logger(__name__)
cfg = get_config()
app = FastAPI(title=cfg.serving.api_title, version=cfg.serving.api_version)


@app.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    agent = get_agent()
    return HealthResponse(status="ok", retriever=getattr(agent.engine.dense, "name", "?"),
                          n_papers=len(agent.engine.papers), version=__version__)


@app.get("/readyz")
def readyz() -> dict:
    get_agent()
    return {"status": "ready"}


@app.get("/version")
def version() -> dict:
    agent = get_agent()
    return {"app": __version__, "retriever": getattr(agent.engine.dense, "version", "?"),
            "reranker": getattr(agent.engine.reranker, "name", "?")}


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    if not req.query.strip():
        raise HTTPException(status_code=422, detail="empty query")
    agent = get_agent()
    agent.cfg.search.final_k = max(1, min(req.k, 50))
    job = agent.search(req.query, save=True)
    sd = job.to_dict()
    return SearchResponse(
        query=sd["query"], intent=sd["intent"], n_results=sd["n_results"],
        results=[PaperResult(id=r["id"], rank=r["rank"], score=r["score"], title=r["title"],
                             abstract=r["abstract"], categories=r["categories"], related=r["related"])
                 for r in sd["results"]],
        facets=sd["facets"], clusters=sd["clusters"], suggestions=sd["suggestions"],
        decisions=sd["decisions"], metrics=sd["metrics"])


@app.get("/related/{paper_id}")
def related(paper_id: str, k: int = 5) -> dict:
    agent = get_agent()
    ids = agent.engine.related(paper_id, k)
    return {"paper_id": paper_id, "related": agent.engine.meta(ids)}


__all__ = ["app"]
