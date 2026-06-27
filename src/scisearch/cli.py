"""Command-line interface — the single entrypoint for the Scientific Literature
Search system.

    scisearch <command> [options]

Commands: data, pairs, train, tune, evaluate, search, demo-agent, serve,
benchmark, error-analysis, monitor, generate-report, generate-slides, autopilot, grade.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from .config import AppConfig, ensure_dirs, load_config
from .logging_utils import get_logger

logger = get_logger(__name__)

TITLE = "Exploratory Scientific Literature Search System"
AUTHOR = "Le Dinh Minh Quan"


def _load(args) -> AppConfig:
    cfg = load_config(args.config) if getattr(args, "config", None) else AppConfig()
    ensure_dirs()
    return cfg


def cmd_data(args):
    from .data.download_dataset import download_all
    print(json.dumps(download_all(_load(args)), indent=2, ensure_ascii=False))


def cmd_pairs(args):
    from .data.corpus import load_corpus
    from .data.pairs import build_pairs
    cfg = _load(args)
    pairs = build_pairs(load_corpus(cfg), cfg.data)
    print(json.dumps({"n_train_pairs": pairs["n_train_pairs"], "n_eval_queries": pairs["n_eval_queries"],
                      "n_corpus": pairs["n_corpus"], "examples": pairs["train_pairs"][:4]}, indent=2, ensure_ascii=False))


def cmd_train(args):
    from .training.train_retriever import train_retriever
    print(json.dumps(train_retriever(_load(args), limit=args.limit, base_model=args.base_model), indent=2))


def cmd_tune(args):
    from .training.tune import tune_retriever
    print(json.dumps(tune_retriever(_load(args), n_trials=args.n_trials, limit=args.limit), indent=2))


def cmd_evaluate(args):
    from .training.evaluate import evaluate
    print(json.dumps(evaluate(_load(args), limit=args.limit).get("summary", {}), indent=2, ensure_ascii=False))


def cmd_search(args):
    from .agent.search_agent import SearchAgent
    job = SearchAgent(_load(args), load_model=not args.tfidf).search(args.query, save=False)
    print(json.dumps(job.to_dict(), indent=2, ensure_ascii=False))


def cmd_demo_agent(args):
    from .agent.search_agent import SearchAgent
    agent = SearchAgent(_load(args), load_model=not args.tfidf)
    for q in ["attention transformer", "how to retrieve relevant passages for question answering",
              "contrastive learning for sentence embeddings"]:
        sd = agent.search(q, save=False).to_dict()
        print(f"\nQUERY: {q!r}")
        print(f"  intent={sd['intent']} | results={sd['n_results']} | decisions={[(d['id'], d['branch']) for d in sd['decisions']]}")
        print(f"  facets={[(f['field'], f['count']) for f in sd['facets'][:4]]}")
        print(f"  clusters={[c['label'] for c in sd['clusters']][:4]}")
        for r in sd["results"][:3]:
            print(f"    [{r['score']:.3f}] {r['title'][:62]}")


def cmd_serve(args):
    import os
    import uvicorn
    if args.config:
        os.environ["SCISEARCH_INFER_CONFIG"] = str(args.config)
    target = "scisearch.api.app_combined:app" if args.ui else "scisearch.api.main:app"
    uvicorn.run(target, host=args.host, port=args.port, reload=False)


def cmd_benchmark(args):
    from .analysis.latency import benchmark
    print(json.dumps(benchmark(_load(args), n=args.n, warmup=args.warmup), indent=2))


def cmd_error_analysis(args):
    from .analysis.error_analysis import error_analysis
    print(json.dumps(error_analysis(_load(args), limit=args.limit), indent=2, ensure_ascii=False))


def cmd_monitor(args):
    from .monitoring.drift_report import monitoring_report
    print(json.dumps(monitoring_report(_load(args), log_path=args.log), indent=2))


def cmd_generate_report(args):
    from .autoreport.report_pdf import generate_report
    print("Report ->", generate_report(_load(args), title=args.title, author=args.author))


def cmd_generate_slides(args):
    from .autoreport.slides_pptx import generate_slides
    print("Slides ->", generate_slides(_load(args), title=args.title, author=args.author))


def cmd_autopilot(args):
    from .automation.autopilot import run_autopilot
    print(json.dumps(run_autopilot(_load(args), title=args.title, author=args.author,
                                   train=not args.no_train, limit=args.limit), indent=2))


def cmd_grade(args):
    from .grading.checklist import build_checklist
    repo = Path(args.repo) if args.repo else Path(__file__).resolve().parents[2]
    print(json.dumps(build_checklist(repo), indent=2))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="scisearch", description=TITLE)
    p.add_argument("--config", help="Path to a YAML config")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("data", help="prefetch corpus + build pairs/eval cache"); sp.set_defaults(func=cmd_data)
    sp = sub.add_parser("pairs", help="show generated (query, paper) training pairs"); sp.set_defaults(func=cmd_pairs)
    sp = sub.add_parser("train", help="fine-tune the dense retriever (MNRL)"); sp.add_argument("--limit", type=int, default=None); sp.add_argument("--base-model", default=None); sp.set_defaults(func=cmd_train)
    sp = sub.add_parser("tune", help="basic LR hyperparameter search"); sp.add_argument("--n-trials", type=int, default=3); sp.add_argument("--limit", type=int, default=8000); sp.set_defaults(func=cmd_tune)
    sp = sub.add_parser("evaluate", help="retriever vs BM25/zero-shot (Recall/MRR/nDCG)"); sp.add_argument("--limit", type=int, default=None); sp.set_defaults(func=cmd_evaluate)
    sp = sub.add_parser("search", help="run one query"); sp.add_argument("--query", required=True); sp.add_argument("--tfidf", action="store_true"); sp.set_defaults(func=cmd_search)
    sp = sub.add_parser("demo-agent", help="run the agent on sample queries"); sp.add_argument("--tfidf", action="store_true"); sp.set_defaults(func=cmd_demo_agent)
    sp = sub.add_parser("serve", help="start the FastAPI server"); sp.add_argument("--host", default="0.0.0.0"); sp.add_argument("--port", type=int, default=8000); sp.add_argument("--ui", action="store_true"); sp.set_defaults(func=cmd_serve)
    sp = sub.add_parser("benchmark", help="latency benchmark"); sp.add_argument("--n", type=int, default=30); sp.add_argument("--warmup", type=int, default=3); sp.set_defaults(func=cmd_benchmark)
    sp = sub.add_parser("error-analysis", help="per-query wins/losses vs BM25"); sp.add_argument("--limit", type=int, default=None); sp.set_defaults(func=cmd_error_analysis)
    sp = sub.add_parser("monitor", help="monitoring report from query logs"); sp.add_argument("--log", default=None); sp.set_defaults(func=cmd_monitor)
    sp = sub.add_parser("generate-report", help="generate the PDF report"); sp.add_argument("--title", default=TITLE); sp.add_argument("--author", default=AUTHOR); sp.set_defaults(func=cmd_generate_report)
    sp = sub.add_parser("generate-slides", help="generate the PPTX slides"); sp.add_argument("--title", default=TITLE); sp.add_argument("--author", default=AUTHOR); sp.set_defaults(func=cmd_generate_slides)
    sp = sub.add_parser("autopilot", help="one-button: train -> eval -> analysis -> report+slides"); sp.add_argument("--title", default=TITLE); sp.add_argument("--author", default=AUTHOR); sp.add_argument("--no-train", action="store_true"); sp.add_argument("--limit", type=int, default=None); sp.set_defaults(func=cmd_autopilot)
    sp = sub.add_parser("grade", help="rubric completeness self-check"); sp.add_argument("--repo", default=None); sp.set_defaults(func=cmd_grade)
    return p


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
