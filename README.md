# 🔎 Exploratory Scientific Literature Search System

> Find **and explore** scientific papers: a query returns not just a ranked list but
> **field facets, topic clusters, related-paper rails and broaden/narrow guidance** — backed by
> a **trainable dense retriever**, hybrid (dense + BM25) search, and an **agentic** pipeline.

**NLP in Industry — Final Assignment.** Author: **Le Dinh Minh Quan** (Student `23127460`).
Reference inspiration: [NLP-Knowledge-Graph/NLP-KG-WebApp](https://github.com/NLP-Knowledge-Graph/NLP-KG-WebApp).

Keyword search (BM25) is precise on exact terms but blind to paraphrase and concept, and returns a
flat list with no way to *explore* a literature. P08 fixes both: a **fine-tuned bi-encoder** understands
scientific phrasing (semantic recall), and an **agent** turns one query into a guided exploration session.

---

## ✅ How this repo meets every assignment requirement

| Requirement | Where it is delivered |
|---|---|
| **Business problem** | [`docs/problem_definition.md`](docs/problem_definition.md) |
| **Dev infra & tooling** | `src/` package, `pyproject.toml`, `requirements*.txt`, `Makefile`, Docker, CI |
| **Data management** | corpus loader + synthetic pair generation ([`data/pairs.py`](src/scisearch/data/pairs.py)); [`docs/data_description.md`](docs/data_description.md), [`docs/data_card.md`](docs/data_card.md) |
| **Model selection & optimization** | fine-tuned dense retriever + **BM25/zero-shot baselines**; IR metrics; [`docs/model_selection.md`](docs/model_selection.md) |
| **Deployment** | FastAPI `/search` + `/related` + Gradio + CLI + Docker + HF Space; [`docs/deployment.md`](docs/deployment.md) |
| **Agentic AI** | deterministic FSM with **4 decision points** + optional LLM brain; [`docs/agent_architecture.md`](docs/agent_architecture.md) |
| **Continual learning & monitoring** | [`docs/continual_learning_monitoring.md`](docs/continual_learning_monitoring.md) + [`monitoring/drift_report.py`](src/scisearch/monitoring/drift_report.py) |
| **Privacy & robustness** | [`docs/privacy_robustness.md`](docs/privacy_robustness.md) |
| **Project management** | [`docs/project_plan.md`](docs/project_plan.md) |
| **Ethics** | [`docs/ethics_statement.md`](docs/ethics_statement.md) |
| **Report + slides** | auto-generated `report.pdf` + `slides.pptx` (`autopilot`) |

---

## 🏗️ Pipeline

```
query
  │  understand: intent + filters + expansion          ── D1 query-type routing
  ▼
dense (fine-tuned bi-encoder) + BM25  →  RRF fusion     ── D2 coverage gate (expand if thin)
  │  cross-encoder rerank (gated)                       ── D3 rerank gate
  ▼
explore: field facets · topic clusters · related papers ── D4 exploration strategy
  ▼
ranked, faceted, clustered, explorable results
```

## 📦 Models & data (ids VERIFIED on the HF Hub)

| Role | Id | License |
|---|---|---|
| **Retriever (trained)** | `BAAI/bge-small-en-v1.5` (fallback `all-MiniLM-L6-v2`) | MIT / Apache |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Apache-2.0 |
| **Baselines** | BM25 (self-contained) · zero-shot base encoder · TF-IDF (offline) | — |
| Corpus | `gfissore/arxiv-abstracts-2021` (title+abstract+categories) | CC0-1.0 |
| Real IR eval | `BeIR/scifact` | CC-BY-SA |
| Domain options | `malteos/scincl`, `allenai/specter2_base` | MIT / Apache |

## 🗂️ Repository layout

```
src/scisearch/
├── config.py  cli.py  logging_utils.py
├── data/         samples.py · corpus.py · pairs.py · download_dataset.py
├── models/       bm25.py · retriever.py · vector_store.py · reranker.py · model_registry.py
├── search/       hybrid.py (RRF) · facets.py · expand.py · engine.py
├── training/     train_retriever.py · evaluate.py · tune.py · metrics.py
├── agent/        state.py · policy.py · tools.py · llm_orchestrator.py · search_agent.py
├── api/          schemas.py · dependencies.py · main.py · ui.py · app_combined.py
├── analysis/ autoreport/ monitoring/ automation/ grading/
configs/ · data/ · models/ · tests/ · docs/ · notebooks/ · app/ · deploy/ · sample_data/
```

---

## 🚀 Quickstart

```bash
pip install -e ".[ml,api,report]"

scisearch data                          # load corpus + build (query, paper) pairs
scisearch demo-agent --tfidf            # run the agent on sample queries (offline)
scisearch search --query "retrieval augmented generation for question answering"
```

### Train
```bash
scisearch --config configs/train.yaml train     # fine-tune the bi-encoder (MNRL, auto-resumes)
scisearch evaluate                              # retriever vs BM25/zero-shot, Recall/MRR/nDCG
```
On Colab/GPU use the notebook (below) — it auto-profiles H100/A100/L4/T4 (batch = #in-batch negatives).

### Serve
```bash
scisearch serve --ui --port 7860        # FastAPI /search + /related + Gradio UI at /ui
```

### One-button report + slides + self-grade
```bash
scisearch autopilot --no-train          # eval → analysis → report.pdf + slides.pptx + bundle
scisearch grade
```

---

## 🤖 The agent (mandatory agentic component)

A **deterministic FSM** with **four decision points** acting on intermediate outputs, plus an
*optional* LLM brain (`anthropic`) that validates its output and **falls back to rules**:

- **D1** query-type routing (keyword / conceptual / hybrid; metadata filters)
- **D2** retrieval-coverage gate (expand the query when results are thin)
- **D3** rerank gate (rerank only when the head of the ranking is ambiguous)
- **D4** exploration strategy (cluster + facet + related + suggest broaden/narrow)

Every step is timed + traced. See [`docs/agent_architecture.md`](docs/agent_architecture.md).

## ☁️ Colab / H100 training

Open [`notebooks/SciSearch_Colab_Training_H100_AUTOPILOT.ipynb`](notebooks/SciSearch_Colab_Training_H100_AUTOPILOT.ipynb)
— mounts Drive, installs Colab-safe deps, auto-profiles the GPU, fine-tunes resume-safely, evaluates
vs BM25/zero-shot, runs the agent, and generates the report/slides.
Step-by-step: [`notebooks/COLAB_GUIDE.md`](notebooks/COLAB_GUIDE.md).

## 🧪 Tests

```bash
pytest -q        # CPU-only, no model/network downloads (built-in corpus + BM25/TF-IDF + identity reranker)
```

## 📚 Docs index

`docs/`: problem_definition · data_description · data_card · model_selection · evaluation ·
agent_architecture · deployment · continual_learning_monitoring · privacy_robustness ·
project_plan · ethics_statement · architecture · model_card · slide_deck_outline · DESIGN_BRIEF.

## 📝 License

MIT — see [`LICENSE`](LICENSE). Pretrained models keep their own licenses (table above). The corpus
snapshot ends in 2021 — ingest newer papers incrementally for production.
