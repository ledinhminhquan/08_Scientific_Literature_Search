"""Matplotlib charts for the report/slides. Returns saved PNG paths; degrades to
``None`` when matplotlib is unavailable."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..logging_utils import get_logger

logger = get_logger(__name__)


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def ndcg_chart(eval_art: Dict[str, Any], out_path: Path) -> Optional[Path]:
    if not eval_art or "model" not in eval_art:
        return None
    try:
        plt = _mpl()
        labels, vals, colors = [], [], []
        labels.append("BM25"); vals.append(eval_art.get("bm25", {}).get("ndcg@10", 0)); colors.append("#9aa7b4")
        if eval_art.get("zero_shot_base"):
            labels.append("zero-shot"); vals.append(eval_art["zero_shot_base"].get("ndcg@10", 0)); colors.append("#d69e2e")
        labels.append("retriever"); vals.append(eval_art["model"].get("ndcg@10", 0)); colors.append("#2b6cb0")
        fig, ax = plt.subplots(figsize=(5.6, 3.4))
        ax.bar(labels, vals, color=colors)
        ax.set_ylabel("nDCG@10"); ax.set_ylim(0, 1); ax.set_title("Retrieval quality: model vs baselines")
        for i, v in enumerate(vals):
            ax.text(i, v + 0.01, f"{v:.3f}", ha="center")
        fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)
        return out_path
    except Exception as exc:
        logger.info("ndcg_chart skipped (%s)", exc)
        return None


def recall_chart(eval_art: Dict[str, Any], out_path: Path) -> Optional[Path]:
    if not eval_art or "model" not in eval_art:
        return None
    try:
        plt = _mpl()
        ks = ["recall@1", "recall@5", "recall@10"]
        m = [eval_art["model"].get(k, 0) for k in ks]
        b = [eval_art.get("bm25", {}).get(k, 0) for k in ks]
        x = range(len(ks))
        fig, ax = plt.subplots(figsize=(5.6, 3.4))
        ax.bar([i - 0.2 for i in x], b, width=0.4, label="BM25", color="#9aa7b4")
        ax.bar([i + 0.2 for i in x], m, width=0.4, label="retriever", color="#2b6cb0")
        ax.set_xticks(list(x)); ax.set_xticklabels(["R@1", "R@5", "R@10"]); ax.set_ylim(0, 1)
        ax.set_title("Recall@k"); ax.legend()
        fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)
        return out_path
    except Exception as exc:
        logger.info("recall_chart skipped (%s)", exc)
        return None


def wins_chart(err_art: Dict[str, Any], out_path: Path) -> Optional[Path]:
    if not err_art or "dense_wins" not in err_art:
        return None
    try:
        plt = _mpl()
        keys = ["dense_wins", "bm25_wins", "ties", "both_fail"]
        vals = [err_art.get(k, 0) for k in keys]
        fig, ax = plt.subplots(figsize=(5.6, 3.4))
        ax.bar(["dense wins", "bm25 wins", "ties", "both fail"], vals,
               color=["#2f855a", "#dd6b20", "#a0aec0", "#c53030"])
        ax.set_title("Per-query: dense retriever vs BM25")
        fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)
        return out_path
    except Exception as exc:
        logger.info("wins_chart skipped (%s)", exc)
        return None


def build_all(arts: Dict[str, Any], out_dir: Path) -> List[Tuple[str, Path]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    charts: List[Tuple[str, Path]] = []
    for name, fn, key in [("ndcg", ndcg_chart, "eval"), ("recall", recall_chart, "eval"),
                          ("wins", wins_chart, "error_analysis")]:
        p = fn(arts.get(key) or {}, out_dir / f"{name}.png")
        if p:
            charts.append((name, p))
    return charts


__all__ = ["ndcg_chart", "recall_chart", "wins_chart", "build_all"]
