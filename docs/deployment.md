# Deployment — P08 Exploratory Scientific Literature Search

This document describes how the **Exploratory Scientific Literature Search System** is packaged, served, and operated. The system turns a query into not just a ranked list of papers but a structured exploration surface — facets (fields), topic clusters, related-paper rails, and broaden/narrow guidance — backed by a fine-tuned dense bi-encoder retriever.

Author: Le Dinh Minh Quan (23127460). Package: `src/scisearch/`.

---

## 1. Delivery formats

The same inference engine is exposed through five interchangeable surfaces, so the system can be consumed as a service, an interactive demo, a script, a container, or a hosted Space.

| Format | Entry point | Purpose |
|---|---|---|
| **FastAPI service** | `src/scisearch/api/main.py` | Programmatic JSON API for integration |
| **Gradio UI** | `src/scisearch/api/ui.py` | Interactive search + exploration demo |
| **Combined app** | `src/scisearch/api/app_combined.py` | FastAPI with the Gradio UI mounted at `/ui` |
| **CLI** | `src/scisearch/cli.py` (console-script `scisearch`) | Data, training, search, serving, automation |
| **Docker / Compose** | `Dockerfile`, `docker-compose.yml` | Reproducible containerized deployment |
| **Hugging Face Space** | `app/` (Gradio) | Public hosted demo |

All five share one engine (`src/scisearch/search/engine.py`) and one agent (`src/scisearch/agent/search_agent.py`), so behavior is identical across surfaces.

---

## 2. FastAPI service

Defined in `src/scisearch/api/main.py`. Schemas live in `api/schemas.py` and shared singletons (engine, agent, loaded models) in `api/dependencies.py`.

### Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/healthz` | Liveness — process is up |
| `GET` | `/readyz` | Readiness — index built and models loaded |
| `GET` | `/version` | Model + index version metadata |
| `POST` | `/search` | Run a full exploratory search for a query |
| `GET` | `/related/{paper_id}` | Dense nearest-neighbor rail for one paper |

### `/healthz` and `/readyz`

`/healthz` returns quickly to signal the process is alive and is suitable as a container/orchestrator liveness probe. `/readyz` is the **readiness** probe: it returns ready only once the FAISS index is built and the retriever/reranker models are loaded into memory, so traffic is not routed to a pod that is still warming up.

### `/version`

`/version` reports the active model and index versions from the registry — the bi-encoder `repo@revision`, the resolved model metadata, and the current index pointer. This makes every served response traceable to an exact model build (see [§7 Model versioning](#7-model-versioning)).

### `POST /search`

The core endpoint. Input is the query (and optional intent/filter hints); output is the full exploration payload.

**Request**

```json
{
  "query": "dense passage retrieval for open-domain QA",
  "top_k": 10
}
```

**Response** — a single JSON object with everything the UI needs to render:

```json
{
  "results":     [ { "paper_id": "...", "title": "...", "abstract": "...", "score": 0.0, "field": "cs.CL" } ],
  "facets":      [ { "field": "cs.CL", "count": 7 } ],
  "clusters":    [ { "label": "passage ranking, retrieval", "members": ["..."] } ],
  "suggestions": { "broaden": ["..."], "narrow": ["..."], "related": ["..."] },
  "decisions":   [ { "point": "D1", "choice": "conceptual", "reason": "..." } ],
  "metrics":     { "latency_ms": 0.0, "reranked": true }
}
```

- **`results`** — the fused, optionally reranked ranking with per-paper scores and field.
- **`facets`** — counts by arXiv category mapped to readable field names.
- **`clusters`** — KMeans topic clusters over TF-IDF, each labeled by its top terms.
- **`suggestions`** — broaden / narrow / related guidance derived from result diversity.
- **`decisions`** — the full audit trace of the agent's four decision points (see below).
- **`metrics`** — per-request latency and whether reranking fired.

### `GET /related/{paper_id}`

Returns the dense nearest-neighbors of a single paper from the FAISS index — the "related papers" rail used to keep exploring sideways from any result, independent of the original query.

---

## 3. Gradio UI

`src/scisearch/api/ui.py` provides the interactive demo: a search box that renders the ranked results, a sidebar of facets and topic clusters, the related-paper rails, and a **decision log** that surfaces the agent's reasoning (intent routing, coverage gate, rerank gate, exploration strategy) so the exploration is transparent rather than a black box.

The combined app (`api/app_combined.py`) mounts this UI at **`/ui`** on the FastAPI process, so a single deployment serves both the JSON API and the human-facing demo.

---

## 4. CLI

The console-script `scisearch` (`src/scisearch/cli.py`) is the operational entry point and covers the full lifecycle:

```
scisearch data              # build / download the corpus
scisearch pairs             # generate (query, positive_paper) training pairs
scisearch train             # fine-tune the bi-encoder
scisearch tune              # hyperparameter tuning
scisearch evaluate          # nDCG@10, Recall@{1,5,10}, MRR@10
scisearch search            # one-off query from the terminal
scisearch demo-agent        # run the agent end-to-end with decision trace
scisearch serve             # launch the FastAPI / combined app
scisearch benchmark         # latency benchmarking
scisearch error-analysis    # failure inspection
scisearch monitor           # runtime monitoring
scisearch generate-report   # build report.pdf
scisearch generate-slides   # build slides.pptx
scisearch autopilot         # end-to-end pipeline
scisearch grade             # self-grade against the rubric
```

`scisearch serve` is the supported way to start the service locally; `scisearch search` and `scisearch demo-agent` are convenient for smoke-testing inference without HTTP.

---

## 5. Inference pipeline (working hot path)

A `/search` request flows through the agent's deterministic FSM, which wraps the retrieval stack and fires four decision points (D1–D4), each recorded in the `decisions` trace:

1. **D1 — query-type routing.** Detect intent (keyword vs. conceptual vs. hybrid from query length / wh-words) and extract metadata filters (field e.g. `cs.CL`, year). Intent biases the fusion weighting. An optional LLM brain (`agent/llm_orchestrator.py`, `anthropic`) can assist query understanding here but is **off by default** (0 paid API calls) and always validates/falls back to rules.
2. **Retrieval.** Dense retrieval over the **FAISS** index (L2-normalized embeddings, inner product = cosine) plus **BM25-Okapi** over title+abstract, fused with **Reciprocal Rank Fusion** (`score(d) = Σ_r 1/(k + rank_r(d))`, `k=60`). Intent duplicates the preferred ranking to bias the fusion.
3. **D2 — coverage gate.** If too few results or the top raw similarity is below threshold, expand the query (abbreviation map + pseudo-relevance feedback from top docs) and re-retrieve. This backstops the zero-result rate.
4. **D3 — rerank gate.** Re-score the top-k with the cross-encoder **only** when the head of the ranking is close/ambiguous (small score margin), saving latency when the top is already unambiguous.
5. **D4 — exploration strategy.** Cluster results into topics, facet by field, fetch related papers, and choose broaden/narrow/related suggestions based on result diversity.

The engine assembles `{results, facets, clusters, suggestions, decisions, metrics}` and returns it as one JSON object.

### Input / output contract

- **Input:** a free-text query (plus optional `top_k`, intent, and filter hints).
- **Output:** the single JSON object described in [§2](#post-search) — ranked `results`, `facets`, `clusters`, `suggestions`, the `decisions` audit trace, and `metrics`.

### User interaction

A user types a query, sees ranked results with **visible scores** and field labels, then explores: click a facet to scope to a field, open a topic cluster, or follow a related-paper rail (`/related/{paper_id}`) to move sideways through the literature. The decision log explains why a given path was taken. Showing scores, diversity, and explicit "broaden" suggestions is deliberate — it signals that results are a ranked sample, not an exhaustive authority.

---

## 6. Latency, scalability, and graceful degradation

### Latency

| Stage | Cost |
|---|---|
| Hot path (dense + BM25 + RRF) | **~tens of ms** on a 100k–500k-paper slice with FAISS |
| Reranking (cross-encoder, **gated**) | **+150–400 ms**, only when D3 fires |
| End-to-end | **sub-second** |

The rerank gate (D3) is the key latency lever: most queries with an unambiguous top result skip the cross-encoder entirely and stay in the tens-of-ms range; only close calls pay the rerank cost.

### Scalability

- **FAISS index sharding** — the vector index is partitioned so the corpus can grow beyond a single shard's memory.
- **Cached embeddings** — paper embeddings are computed once and persisted; serving never re-embeds the corpus.
- **Models loaded once** — the retriever and reranker are loaded into memory at startup (via `api/dependencies.py` singletons) and reused across requests, not reloaded per call.

### Graceful degradation

The system is built to always run, even with no GPU, no network, and no `torch`:

- A self-contained **numpy brute-force** store substitutes when `faiss` is unavailable.
- A **TF-IDF retriever** (sklearn) stands in for the dense slot so hybrid search runs without `torch`.
- An **identity reranker** replaces the cross-encoder when it can't be loaded.
- A built-in **mini-corpus of ~40 real NLP papers** keeps the demo functional offline.

This was validated end-to-end offline: the agent runs on the built-in corpus, all four decisions fire, intent routing is correct (e.g. the query "dense passage retrieval" ranks the Dense Passage Retrieval paper **#1**), and facets + topic clusters are produced.

---

## 7. Model versioning

Versioning is handled by the model registry (`src/scisearch/models/model_registry.py`):

- **`model_meta.json`** records each trained build's metadata.
- A **`latest` pointer** selects the active build.
- Models are pinned by **`repo@revision`**, so a served response is reproducible against an exact base-model revision.

The base retriever is **`BAAI/bge-small-en-v1.5`** (MIT, 33.4M params, 384-dim), fine-tuned with sentence-transformers; the fallback is **`sentence-transformers/all-MiniLM-L6-v2`** (Apache, 22.7M, 384-dim). The reranker is **`cross-encoder/ms-marco-MiniLM-L-6-v2`** (Apache). The `/version` endpoint reports the active model and index versions so any deployment's exact lineage is queryable at runtime.

---

## 8. Containerization

- **`Dockerfile`** — based on `python:3.11-slim`.
- **`docker-compose.yml`** — orchestrates the service for local/single-host deployment.
- **Hugging Face Space** — the Gradio app under `app/` provides a public hosted demo.

A typical flow:

```bash
docker compose up --build       # build image + start the service
# service exposes the FastAPI API and the Gradio UI at /ui
```

Health/readiness probes map cleanly onto container orchestration: `/healthz` for liveness, `/readyz` to hold traffic until the index and models are ready.

---

## 9. Deployment challenges & limitations

| Challenge | Detail | Mitigation |
|---|---|---|
| **Index build time & memory** | Embedding and indexing millions of papers (the corpus `gfissore/arxiv-abstracts-2021` is ~2.0M rows) is time- and memory-intensive. | FAISS index sharding; cached embeddings computed once; the served hot path operates over a 100k–500k slice. |
| **Embedding refresh** | When the retriever is re-fine-tuned, the entire corpus must be re-embedded and re-indexed. | Embeddings cached/persisted; registry `repo@revision` lets old and new index builds coexist and the `latest` pointer cut over atomically. |
| **Rerank cost** | The cross-encoder adds 150–400 ms per query. | D3 rerank gate fires only on ambiguous heads, keeping the common case in tens of ms. |
| **Stale index** | The primary corpus has a **2021 recency cutoff** — papers after the snapshot are missing. | Document the cutoff; continual/incremental ingestion plan needed to add newer papers. |
| **Corpus coverage/bias** | arXiv-only, English. | Communicate scope; report Recall@k so users understand coverage, not just precision. |
| **Query privacy** | Search queries can reveal a research direction. | Minimize retention/logging of raw queries; on-prem deployment option. |
| **Over-trust in ranking** | Users may treat the top result as exhaustive/authoritative. | Always show scores + diversity + explicit "broaden" suggestions; never hide that results are a ranked sample. |

---

## 10. Summary

The system ships as a FastAPI service (`/healthz`, `/readyz`, `/version`, `POST /search`, `GET /related/{paper_id}`), a Gradio exploration UI (mountable at `/ui`), a full-lifecycle `scisearch` CLI, a `python:3.11-slim` Docker image with Compose, and a Hugging Face Space. One query in returns one `{results, facets, clusters, suggestions, decisions, metrics}` object out. The hot path (dense + BM25 + RRF over FAISS) runs in tens of milliseconds, the gated cross-encoder rerank keeps end-to-end latency sub-second, and the stack scales via index sharding, cached embeddings, and one-time model loading. Every served response is traceable to an exact model build through the registry's `repo@revision` pointer.
