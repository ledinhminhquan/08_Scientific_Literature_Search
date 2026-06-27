"""Monitoring & drift report from production query logs (JSONL).

Aggregates query outcomes (status mix, intent mix, zero-result rate, latency) and
a drift signal comparing a recent window vs an earlier baseline (rising
zero-result rate or shifting intent mix signals a corpus/query-distribution shift).
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional

from ..config import AppConfig, run_dir
from ..logging_utils import get_logger, utc_stamp

logger = get_logger(__name__)


def _read_logs(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def _window_stats(rows: List[Dict]) -> Dict:
    if not rows:
        return {"n": 0}
    statuses: Dict[str, int] = {}
    intents: Dict[str, int] = {}
    lats, zero = [], 0
    for r in rows:
        statuses[r.get("status", "?")] = statuses.get(r.get("status", "?"), 0) + 1
        intents[r.get("intent", "?")] = intents.get(r.get("intent", "?"), 0) + 1
        if (r.get("n_results", 0) or 0) == 0:
            zero += 1
        m = r.get("metrics", {}) or {}
        if isinstance(m.get("latency_ms"), (int, float)):
            lats.append(m["latency_ms"])
    return {"n": len(rows), "statuses": statuses, "intents": intents,
            "zero_result_rate": round(zero / len(rows), 4),
            "mean_latency_ms": round(mean(lats), 1) if lats else None}


def monitoring_report(cfg: AppConfig, log_path: Optional[str] = None, save: bool = True) -> Dict:
    path = Path(log_path) if log_path else cfg.serving.query_log_path
    rows = _read_logs(path)
    overall = _window_stats(rows)
    drift = {}
    if len(rows) >= 6:
        half = len(rows) // 2
        base, recent = _window_stats(rows[:half]), _window_stats(rows[half:])

        def delta(k):
            a, b = base.get(k), recent.get(k)
            return round(b - a, 4) if isinstance(a, (int, float)) and isinstance(b, (int, float)) else None
        drift = {"baseline_window": base, "recent_window": recent,
                 "delta_zero_result_rate": delta("zero_result_rate"),
                 "alert": bool((delta("zero_result_rate") or 0) > 0.1)}
    result = {"log_path": str(path), "n_queries": len(rows), "overall": overall, "drift": drift,
              "note": "no query logs found yet" if not rows else ""}
    if save:
        d = run_dir() / "monitoring"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"monitor-{utc_stamp()}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        (d / "latest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


__all__ = ["monitoring_report"]
