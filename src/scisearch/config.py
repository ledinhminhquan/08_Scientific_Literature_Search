"""Typed configuration + YAML loader for the Scientific Literature Search system.

Single source of truth for the paper corpus, the trainable dense retriever, the
reranker, the hybrid-search + exploration settings, agent thresholds and serving.
Paths come from environment variables so nothing is hard-coded.

Environment overrides
---------------------
* ``SCISEARCH_ARTIFACTS_DIR`` – base for data/models/index/runs (Drive on Colab)
* ``SCISEARCH_DATA_DIR``      – corpus / generated pairs cache
* ``SCISEARCH_MODEL_DIR``     – trained models
* ``SCISEARCH_INDEX_DIR``     – FAISS index + paper store
* ``SCISEARCH_RUN_DIR``       – eval/benchmark/analysis JSON
* ``HF_HOME``                 – HuggingFace cache
* ``SCISEARCH_LLM_API_KEY``   – optional key for the LLM query-understanding brain
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def _env(key: str, default: Optional[str] = None) -> Optional[str]:
    v = os.environ.get(key)
    return v if v not in (None, "") else default


def artifacts_dir() -> Path:
    return Path(_env("SCISEARCH_ARTIFACTS_DIR", "artifacts")).expanduser()


def data_dir() -> Path:
    return Path(_env("SCISEARCH_DATA_DIR", str(artifacts_dir() / "data"))).expanduser()


def model_dir() -> Path:
    return Path(_env("SCISEARCH_MODEL_DIR", str(artifacts_dir() / "models"))).expanduser()


def index_dir() -> Path:
    return Path(_env("SCISEARCH_INDEX_DIR", str(artifacts_dir() / "index"))).expanduser()


def run_dir() -> Path:
    return Path(_env("SCISEARCH_RUN_DIR", str(artifacts_dir() / "runs"))).expanduser()


# ─────────────────────────────────────────────────────────────────────────────
# Sub-configs
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DataConfig:
    """Paper corpus + training-pair generation (see docs/data_description.md)."""
    # configurable HF corpus (title + abstract + categories); loader falls back to
    # a built-in mini-corpus when the dataset / network is unavailable.
    # PRIMARY (VERIFIED): gfissore/arxiv-abstracts-2021 (CC0, title+abstract+categories(list)+authors).
    hf_corpus: str = "gfissore/arxiv-abstracts-2021"
    hf_corpus_config: str = ""
    hf_corpus_split: str = "train"
    title_col: str = "title"
    abstract_col: str = "abstract"
    category_col: str = "categories"   # a list in gfissore/arxiv-abstracts-2021
    corpus_limit: int = 20000          # max papers to index
    use_hf_corpus: bool = True
    # optional real IR eval set (VERIFIED): BeIR/scifact (corpus + queries + qrels)
    ir_eval_dataset: str = "BeIR/scifact"
    # synthetic (query, positive paper) training pairs built from the corpus
    pairs_per_paper: int = 2
    max_train_pairs: int = 80000
    eval_queries: int = 2000           # held-out query->paper IR eval set
    seed: int = 42


@dataclass
class ModelConfig:
    """Trainable dense retriever (sentence-transformers bi-encoder)."""
    base_model: str = "BAAI/bge-small-en-v1.5"             # MIT, 33M
    base_model_fallback: str = "sentence-transformers/all-MiniLM-L6-v2"  # Apache, 22M
    # domain options (scientific): allenai/specter2_base, malteos/scincl
    query_instruction: str = ""        # bge-v1.5 s2p instruction (optional; "" after fine-tune)
    max_seq_length: int = 256
    normalize_embeddings: bool = True
    # training (sentence-transformers SentenceTransformerTrainer / MNRL)
    num_train_epochs: int = 1
    learning_rate: float = 2.0e-5
    per_device_train_batch_size: int = 128   # large batch => more in-batch negatives
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    bf16: bool = True
    fp16: bool = False
    tf32: bool = True
    eval_steps: int = 500
    save_steps: int = 500
    seed: int = 42
    output_subdir: str = "retriever"

    @property
    def output_dir(self) -> Path:
        return model_dir() / self.output_subdir


@dataclass
class RerankerConfig:
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"   # Apache, reranker
    top_k_rerank: int = 30
    enabled: bool = True


@dataclass
class SearchConfig:
    top_k: int = 50                    # candidates retrieved per strategy
    final_k: int = 10                  # results returned
    rrf_k: int = 60                    # RRF constant
    use_bm25: bool = True
    use_dense: bool = True
    n_clusters: int = 5                # topic clusters over results (exploration)
    related_k: int = 5                 # related papers per result
    min_cluster_size: int = 2


@dataclass
class AgentConfig:
    """Agent thresholds (decision points) + optional LLM query-understanding brain."""
    # D1 — query-type routing (keyword-heavy vs conceptual)
    short_query_words: int = 3         # <= => keyword-leaning
    # D2 — retrieval-coverage gate
    min_results: int = 3               # below => expand the query and re-retrieve
    min_top_score: float = 0.30        # below => low-confidence => expand
    max_expand_attempts: int = 1
    # D3 — rerank gate
    rerank_margin: float = 0.05        # top results within this score margin => rerank
    # D4 — exploration/presentation strategy
    diversify_min_clusters: int = 2
    # optional cloud brain (off by default; the agent runs fully on rules)
    llm_fallback_enabled: bool = False
    llm_model: str = "claude-haiku-4-5-20251001"
    llm_api_key_env: str = "SCISEARCH_LLM_API_KEY"


@dataclass
class ServingConfig:
    model_version: str = "v1"
    api_title: str = "Scientific Literature Search API"
    api_version: str = "1.0.0"
    log_queries: bool = True
    query_log_subdir: str = "query_logs"
    max_query_len: int = 512

    @property
    def query_log_path(self) -> Path:
        return run_dir() / self.query_log_subdir / "queries.jsonl"


@dataclass
class AppConfig:
    project_title: str = "Exploratory Scientific Literature Search System"
    author: str = "Le Dinh Minh Quan"
    student_id: str = "23127460"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    serving: ServingConfig = field(default_factory=ServingConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


_SECTIONS = {"data": DataConfig, "model": ModelConfig, "reranker": RerankerConfig,
             "search": SearchConfig, "agent": AgentConfig, "serving": ServingConfig}


def _build(cls, raw: Optional[Dict[str, Any]]):
    raw = raw or {}
    known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    return cls(**{k: v for k, v in raw.items() if k in known})


def load_config(path: Optional[str | os.PathLike] = None) -> AppConfig:
    raw: Dict[str, Any] = {}
    if path is not None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Config not found: {p}")
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    top = {k: raw[k] for k in ("project_title", "author", "student_id") if k in raw}
    sections = {name: _build(cls, raw.get(name)) for name, cls in _SECTIONS.items()}
    return AppConfig(**top, **sections)


def save_config(cfg: AppConfig, path: str | os.PathLike) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False, allow_unicode=True), encoding="utf-8")


def ensure_dirs() -> Dict[str, Path]:
    dirs = {"artifacts": artifacts_dir(), "data": data_dir(), "models": model_dir(),
            "index": index_dir(), "runs": run_dir()}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


__all__ = ["DataConfig", "ModelConfig", "RerankerConfig", "SearchConfig", "AgentConfig",
           "ServingConfig", "AppConfig", "load_config", "save_config", "ensure_dirs",
           "artifacts_dir", "data_dir", "model_dir", "index_dir", "run_dir"]
