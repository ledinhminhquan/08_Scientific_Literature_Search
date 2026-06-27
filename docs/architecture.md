# System Architecture — P08 Exploratory Scientific Literature Search

**Project:** Exploratory Scientific Literature Search System
**Author:** Le Dinh Minh Quan (student 23127460)
**Course:** NLP in Industry — final assignment
**Package:** `src/scisearch/` · Reference: github.com/NLP-Knowledge-Graph/NLP-KG-WebApp

This document describes the technical architecture of the system: the end-to-end
component layout, the data flow through retrieval and exploration, the repository
module map, how configuration wires runtime artifacts, the lazy-import /
graceful-degradation design, the retriever fallback chain, the model-registry
linkage between training and serving, and how the API, UI, and CLI all share one
agent and one search engine.

---

## 1. Design goals that shape the architecture

The product is not "return a ranked list." A query returns a ranked list **plus**
facets (fields), topic clusters, related-paper rails, and broaden/narrow guidance,
all backed by a **fine-tuned dense retriever**. Three constraints drive the design:

1. **It is fundamentally an NLP/IR problem.** Keyword search (BM25) is precise on
   exact terms but blind to concept and paraphrase. A trained dense bi-encoder maps
   queries and papers into a shared embedding space so conceptual queries match.
2. **The demo must always run.** Heavy dependencies (torch, sentence-transformers,
   faiss) are *optional*; the system degrades gracefully to pure-Python / sklearn /
   numpy paths and a built-in mini-corpus.
3. **One brain, three faces.** The FastAPI service, the Gradio UI, and the CLI must
   all delegate to the *same* agent and the *same* search engine — no duplicated
   ranking logic.

---

## 2. End-to-end component diagram

```
                          ┌─────────────────────────────────────────────────────┐
   CLIENTS                │  FastAPI (api/main.py)   Gradio UI (api/ui.py)        │
                          │  /healthz /readyz        search box + facet/cluster   │
                          │  /version /search        sidebar + decision log       │
                          │  /related/{paper_id}     app_combined.py mounts /ui    │
                          │  CLI (cli.py: search, demo-agent, serve, benchmark…)   │
                          └───────────────────────────┬─────────────────────────────┘
                                                      │  (shared via api/dependencies.py)
                                                      ▼
                          ┌─────────────────────────────────────────────────────┐
   AGENT  (deterministic FSM + optional LLM brain)    │  agent/search_agent.py    │
                          │  state.py · policy.py · tools.py · llm_orchestrator.py │
                          │                                                       │
                          │  D1 query-type routing  ──┐  D2 coverage gate         │
                          │  (intent + filters)       │  (expand + re-retrieve)   │
                          │  D3 rerank gate ──────────┘  D4 exploration strategy  │
                          └───────────────────────────┬─────────────────────────────┘
                                                      │  calls engine + tools
                                                      ▼
                          ┌─────────────────────────────────────────────────────┐
   SEARCH ENGINE          │  search/engine.py  (orchestrates the hot path)        │
                          │                                                       │
                          │   ┌─────────────┐   ┌──────────────┐                  │
                          │   │ Dense        │   │ BM25-Okapi    │  search/hybrid │
                          │   │ retriever    │   │ (self-cont.)  │  RRF fusion    │
                          │   │ models/      │   │ models/bm25.py│  (k=60)        │
                          │   │ retriever.py │   └──────┬────────┘                │
                          │   │  + FAISS /   │          │                         │
                          │   │  numpy store │          ▼                         │
                          │   │ vector_store │   ┌──────────────┐                 │
                          │   └──────┬───────┘   │ Reranker      │ models/        │
                          │          └──────────▶│ cross-encoder │ reranker.py    │
                          │                       │ (D3 gated)   │ identity f/b   │
                          │                       └──────┬───────┘                │
                          │  EXPLORATION:  facets.py (arXiv cat → field names)    │
                          │  clustering (KMeans/TF-IDF) · related (dense NN)      │
                          │  expand.py (abbrev map + pseudo-relevance feedback)   │
                          └───────────────────────────┬─────────────────────────────┘
                                                      │  loads artifacts
                                                      ▼
                          ┌─────────────────────────────────────────────────────┐
   ARTIFACTS / DATA       │  model_registry.py (model_meta.json + latest pointer) │
                          │  FAISS index + cached embeddings (models/)            │
                          │  corpus: gfissore/arxiv-abstracts-2021 (or mini-corpus)│
                          │  eval: BeIR/scifact · mteb/scidocs-reranking          │
                          └─────────────────────────────────────────────────────┘
                                                      ▲
                          ┌───────────────────────────┴─────────────────────────────┐
   TRAINING (offline)     │  training/train_retriever.py  (SentenceTransformerTrainer│
                          │  + MultipleNegativesRankingLoss, in-batch negatives)     │
                          │  data/pairs.py (generated (query, positive) pairs)       │
                          │  training/evaluate.py · metrics.py · tune.py             │
                          │  writes fine-tuned model → registry → latest pointer     │
                          └──────────────────────────────────────────────────────────┘
```

---

## 3. Data flow (one `/search` request)

1. **Ingest query.** The client (API / UI / CLI) hands the raw query to the agent
   via `api/dependencies.py`, which constructs/caches the shared engine + agent.
2. **D1 — query-type routing.** `agent/policy.py` detects intent (keyword vs
   conceptual vs hybrid, by query length and wh-words) and extracts metadata
   filters (field e.g. `cs.CL`, year). Intent biases fusion weighting.
3. **Retrieve.** `search/engine.py` runs the **dense retriever** (over FAISS or the
   numpy fallback) and **BM25-Okapi** in parallel over the corpus.
4. **Fuse.** `search/hybrid.py` combines the two rankings with **Reciprocal Rank
   Fusion**. Intent biases the fusion by duplicating the preferred ranking
   (keyword → more BM25 weight; conceptual → more dense weight).
5. **D2 — coverage gate.** If too few results OR top *raw* similarity is below
   threshold, `search/expand.py` expands the query (abbreviation map +
   pseudo-relevance feedback from top docs) and re-retrieves. This backstops the
   zero-result rate (target < 2%).
6. **D3 — rerank gate.** Only when the head of the ranking is close/ambiguous
   (small score margin) does the cross-encoder re-score top-k; otherwise it is
   skipped to save latency.
7. **D4 — exploration strategy.** `search/facets.py` facets results by arXiv
   category → readable field names; KMeans-over-TF-IDF produces topic clusters
   labeled by top terms; dense nearest-neighbors produce related papers; the agent
   suggests broaden/narrow/related based on result diversity.
8. **Respond.** The endpoint returns
   `{results[], facets[], clusters[], suggestions, decisions, metrics}` — note
   `decisions` is the full agent audit trace of D1–D4.

**Latency profile.** Hot path (dense + BM25 + RRF) is ~tens of ms on a 100k–500k
slice via FAISS; the gated rerank adds ~150–400 ms; end-to-end stays sub-second.

---

## 4. Repository module map (`src/scisearch/*`)

| Module | Responsibility |
|---|---|
| `config.py` | Central configuration: artifact paths, model ids, RRF `k`, thresholds, env-var resolution. |
| `cli.py` | Console-script `scisearch`: `data, pairs, train, tune, evaluate, search, demo-agent, serve, benchmark, error-analysis, monitor, generate-report, generate-slides, autopilot, grade`. |
| `logging_utils.py` | Structured logging shared across components. |
| `data/samples.py` | Built-in mini-corpus (~40 real NLP papers) — offline fallback. |
| `data/corpus.py` | Corpus loading/normalization (title+abstract, categories, versions). |
| `data/pairs.py` | Generates `(query, positive_paper)` training pairs from the corpus. |
| `data/download_dataset.py` | Pulls `gfissore/arxiv-abstracts-2021` and eval sets. |
| `models/bm25.py` | Self-contained BM25-Okapi over tokenized title+abstract. |
| `models/retriever.py` | Dense bi-encoder retriever + the TF-IDF/fallback slot. |
| `models/vector_store.py` | FAISS index over L2-normalized embeddings; numpy brute-force fallback. |
| `models/reranker.py` | Cross-encoder reranker; identity fallback. |
| `models/model_registry.py` | `model_meta.json` + latest pointer linking training → serving. |
| `search/hybrid.py` | RRF fusion of dense + BM25 rankings. |
| `search/facets.py` | Faceting (arXiv cat → field), topic clustering, related papers. |
| `search/expand.py` | Query expansion: abbreviation map + pseudo-relevance feedback. |
| `search/engine.py` | Orchestrates retrieve → fuse → (rerank) → explore. |
| `training/train_retriever.py` | Fine-tuning via `SentenceTransformerTrainer` + MNRL. |
| `training/evaluate.py` | `InformationRetrievalEvaluator` / self-contained IR eval. |
| `training/tune.py` | Hyperparameter tuning. |
| `training/metrics.py` | nDCG@10, Recall@{1,5,10}, MRR@10. |
| `agent/state.py` | FSM state object (carries query, intent, filters, results, trace). |
| `agent/policy.py` | Deterministic rules for D1–D4 decision points. |
| `agent/tools.py` | Tool wrappers the agent invokes (retrieve, expand, rerank, cluster…). |
| `agent/llm_orchestrator.py` | Optional Anthropic LLM brain at D1; validates + falls back to rules. |
| `agent/search_agent.py` | Top-level agent loop binding policy + engine + tools. |
| `api/schemas.py` | Pydantic request/response schemas. |
| `api/dependencies.py` | Builds/caches the shared engine + agent for all surfaces. |
| `api/main.py` | FastAPI app: `/healthz /readyz /version`, `POST /search`, `GET /related/{paper_id}`. |
| `api/ui.py` | Gradio UI: search box → ranked results + facet/cluster sidebar + decision log. |
| `api/app_combined.py` | Combined app that mounts the UI at `/ui`. |
| `analysis/latency.py`, `analysis/error_analysis.py` | Latency profiling + error analysis. |
| `autoreport/`, `monitoring/`, `automation/`, `grading/` | Report/slide generation, monitoring, autopilot, grading. |

Supporting top-level dirs: `configs/ data/ models/ tests/ docs/ notebooks/ app/
deploy/`, plus `Dockerfile docker-compose.yml Makefile pyproject.toml
requirements*.txt README.md`.

---

## 5. Configuration & artifact wiring

`config.py` is the single place that resolves where artifacts live, so the same
code runs locally, in Docker, and on Colab/Drive:

- **Index & embeddings** — the FAISS index plus cached embeddings live under
  `models/`. On Colab the artifact root can point at a mounted Google Drive path so
  a trained model and its index survive runtime restarts. Env vars override the
  defaults; nothing is hard-coded into the engine.
- **Model ids** — primary `BAAI/bge-small-en-v1.5`; the registry's *latest pointer*
  (a `repo@revision`) is what the engine actually loads, so promoting a new model is
  a pointer write, not a code change.
- **Tunables** — RRF `k` (default 60), D2 similarity/coverage thresholds, D3 score
  margin, top-k, and the LLM-brain on/off flag (OFF by default ⇒ zero paid API
  calls) are all config-driven.

---

## 6. Lazy imports & graceful degradation

Every heavy dependency is imported **lazily, inside the function that needs it**, and
wrapped so a missing import flips the component to a pure-Python/sklearn/numpy
fallback instead of crashing import-time. This keeps the package importable — and the
demo runnable — with *no* torch and *no* network.

| Component | Preferred path | Fallback path |
|---|---|---|
| Dense embeddings | `torch` + `sentence-transformers` bi-encoder | sklearn **TF-IDF retriever** in the dense slot |
| Vector store | `faiss-cpu` (inner product on normalized vectors) | **numpy** brute-force cosine |
| Keyword search | self-contained **BM25-Okapi** (always available, no deps) | — |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | **identity** reranker (no-op) |
| Corpus | `gfissore/arxiv-abstracts-2021` via `datasets` | built-in **~40-paper mini-corpus** |
| Agent brain | optional Anthropic LLM at D1 | deterministic FSM rules |

Because BM25 is self-contained and TF-IDF stands in for the dense slot, **hybrid
search + RRF still runs with no torch** — only the *quality* of the dense leg
degrades, not the system's availability.

---

## 7. Retriever fallback chain

The dense leg resolves through a strict priority order; the first one that loads
wins:

```
   fine-tuned dense retriever   (registry latest pointer → models/)
            │  (model/registry/torch unavailable)
            ▼
   base / zero-shot dense       (BAAI/bge-small-en-v1.5, else all-MiniLM-L6-v2)
            │  (torch / sentence-transformers unavailable)
            ▼
   TF-IDF retriever (sklearn)   — offline, no-torch stand-in for the dense slot
```

The base encoder also doubles as a **baseline to beat**: the fine-tuned retriever
must outperform both the zero-shot base encoder and BM25, especially on conceptual
queries where dense ≫ BM25. Model facts (all VERIFIED): primary
`BAAI/bge-small-en-v1.5` (MIT, 33.4M params, 384-dim); fallback
`sentence-transformers/all-MiniLM-L6-v2` (Apache, 22.7M, 384-dim); domain options
`malteos/scincl` (MIT, 768-dim) and `allenai/specter2_base` (Apache, 768-dim).

---

## 8. RRF — hybrid fusion formula

Dense and BM25 each produce a ranking; they are fused by **Reciprocal Rank Fusion**:

$$
\text{score}(d) \;=\; \sum_{r \in \text{rankings}} \frac{1}{k + \text{rank}_r(d)}, \qquad k = 60
$$

```python
# search/hybrid.py — Reciprocal Rank Fusion, k = 60
def rrf(rankings, k=60):
    scores = {}
    for ranking in rankings:                 # e.g. [dense_ranked, bm25_ranked]
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)
```

RRF needs only rank positions (not comparable raw scores), which is why it cleanly
fuses a cosine-similarity dense ranking with a BM25 ranking. **Intent biases the
fusion by duplicating the preferred ranking** in the `rankings` list — a conceptual
query weights the dense ranking more, a keyword query weights BM25 more.

---

## 9. Model registry: training → serving linkage

Training and serving are decoupled through `models/model_registry.py`:

- **Training writes.** `training/train_retriever.py` fine-tunes the bi-encoder
  (`SentenceTransformerTrainer` + `MultipleNegativesRankingLoss`, in-batch
  negatives) on generated `(query, positive_paper)` pairs, evaluates with
  `InformationRetrievalEvaluator` on held-out data (headline **nDCG@10**, plus
  Recall@{1,5,10} and MRR@10), then registers the artifact. Each entry records
  metadata in `model_meta.json` (the `repo@revision` it derives from, metrics, etc.).
- **A latest pointer** names the current best model.
- **Serving reads.** `search/engine.py` (via `config.py`) loads whatever the latest
  pointer references. Promoting a model = updating the pointer; no engine code
  changes, which also gives reproducible **versioning** (model_meta.json + pointer +
  `repo@revision`) and supports rollback.

This is the only coupling between the offline training pipeline and the online
search engine — keeping the hot path free of training dependencies.

---

## 10. One agent + one engine across API, UI, and CLI

All three surfaces resolve the **same** `SearchAgent` and `SearchEngine` instances
through `api/dependencies.py`, so ranking, fusion, gating, and exploration logic
exist exactly once:

- **FastAPI** (`api/main.py`) — `POST /search` returns
  `{results, facets, clusters, suggestions, decisions, metrics}`;
  `GET /related/{paper_id}` reuses the dense nearest-neighbor path;
  `/healthz /readyz /version` for ops.
- **Gradio UI** (`api/ui.py`) — search box → ranked results + facet/cluster sidebar
  + the agent decision log; `api/app_combined.py` mounts it at `/ui` alongside the
  API.
- **CLI** (`cli.py`, console-script `scisearch`) — `search` and `demo-agent` drive
  the same agent for offline/local use; `serve` launches the API; `benchmark`,
  `evaluate`, `error-analysis`, `monitor` exercise the same engine.

Because every face calls one agent over one engine, the audit trace, fallback
behavior, and exploration outputs are identical no matter how the system is invoked.

---

## 11. Deployment & robustness notes

- **Packaging:** Docker (`python:3.11-slim`), `docker-compose`, and a Hugging Face
  Space (Gradio). Scalability comes from FAISS index sharding + cached embeddings;
  versioning from the model registry.
- **Validated offline:** the agent runs end-to-end on the built-in corpus, all four
  decision points fire, intent routing is correct (e.g. "dense passage retrieval" →
  the Dense Passage Retrieval paper ranks #1), facets + topic clusters are produced,
  and `report.pdf` + `slides.pptx` generate.
- **Robustness:** out-of-domain/garbled queries are caught by the D2 coverage gate +
  expansion; graceful degradation (TF-IDF retriever + identity reranker + built-in
  corpus) keeps the system available when torch/network are absent.
- **Known limits (documented, not hidden):** the primary corpus has a **2021 recency
  cutoff** (stale-index risk ⇒ incremental-update plan); raw queries can reveal
  research direction (minimize retention/logging; on-prem option); the UI shows
  scores + diversity + broaden suggestions so users never treat the top result as
  exhaustive. The system indexes only public arXiv papers.
