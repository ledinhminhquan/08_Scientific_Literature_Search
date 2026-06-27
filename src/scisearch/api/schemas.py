"""Pydantic request/response schemas for the Scientific Literature Search API."""

from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    k: int = Field(10, description="Number of results to return")


class PaperResult(BaseModel):
    id: str
    rank: int
    score: float
    title: str
    abstract: str
    categories: List[str]
    related: List[str] = []


class SearchResponse(BaseModel):
    query: str
    intent: str
    n_results: int
    results: List[PaperResult]
    facets: List[Dict[str, Any]]
    clusters: List[Dict[str, Any]]
    suggestions: List[str]
    decisions: List[Dict[str, Any]]
    metrics: Dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    retriever: str
    n_papers: int
    version: str


__all__ = ["SearchRequest", "PaperResult", "SearchResponse", "HealthResponse"]
