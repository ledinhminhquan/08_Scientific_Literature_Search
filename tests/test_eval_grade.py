"""Evaluation, error analysis, monitoring and grading."""

from __future__ import annotations

from pathlib import Path

from scisearch.analysis.error_analysis import error_analysis
from scisearch.grading.checklist import build_checklist
from scisearch.monitoring.drift_report import monitoring_report
from scisearch.training.evaluate import evaluate


def test_evaluate_structure(cfg):
    res = evaluate(cfg, save=False)
    assert "model" in res and "bm25" in res
    assert "ndcg@10" in res["model"]
    assert "summary" in res and "beats_bm25" in res["summary"]


def test_error_analysis_structure(cfg):
    res = error_analysis(cfg, save=False)
    assert set(("dense_wins", "bm25_wins", "ties", "both_fail")).issubset(res)
    assert res["n_queries"] > 0


def test_monitoring_handles_empty(cfg):
    res = monitoring_report(cfg, log_path="/nonexistent/queries.jsonl", save=False)
    assert res["n_queries"] == 0


def test_grade_repo():
    repo = Path(__file__).resolve().parents[1]
    res = build_checklist(repo)
    assert res["summary"]["FAIL"] == 0, [i for i in res["items"] if i["status"] == "FAIL"]
    assert res["ok"] is True
