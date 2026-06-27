# P08 — Exploratory Scientific Literature Search: Project Plan

**Project:** Exploratory Scientific Literature Search System
**Author:** Le Dinh Minh Quan (student 23127460)
**Course:** NLP in Industry — Final Assignment
**Package:** `src/scisearch/` (mirrors P02–P07; reuses classic RAG-retrieval patterns)
**Reference:** github.com/NLP-Knowledge-Graph/NLP-KG-WebApp

---

## 1. Scope & Objective

The deliverable is a search engine that does not just **find** scientific papers but lets a
researcher **explore** them. A query returns a ranked list **plus** facets (arXiv fields),
topic clusters (KMeans over TF-IDF), related-paper rails (dense nearest neighbours) and
broaden/narrow guidance — all backed by a **fine-tuned dense bi-encoder**.

The search engine **is** the NLP/IR problem: semantic matching of scientific phrasing
(paraphrase, concept), query understanding and topic clustering. BM25 keyword search is
precise on exact terms but blind to concept; a trained dense retriever closes that gap,
especially on conceptual queries.

This document covers **project management and teamwork**: the phased plan and milestone
timeline, the task breakdown across simulated roles for a solo build, and a reflection on how
the same work scales to a real team.

---

## 2. Project Plan — Phases & Milestones

The project is sequenced into eight phases. Each phase has a concrete exit artifact so progress
is verifiable rather than aspirational. The headline technical goal — the fine-tuned retriever
beating **both** the BM25 baseline and the **zero-shot base encoder** on **nDCG@10** — is the
gate between Phase 3 and the rest.

### 2.1 Phase summary

| # | Phase | Key activities | Exit artifact / milestone |
|---|-------|----------------|---------------------------|
| 0 | **Research & design** | Scope the exploratory-search problem; pick models & datasets; lock the architecture (retriever → hybrid → rerank → explore → agent) | This plan + verified model/dataset IDs in `config.py` |
| 1 | **Corpus + pairs + BM25 baseline** | Load `gfissore/arxiv-abstracts-2021`; build the field/year/author facet metadata; generate `(query, positive_paper)` training pairs; stand up self-contained BM25-Okapi | Indexed corpus slice + pair set + **BM25 baseline nDCG@10** |
| 2 | **Retriever fine-tune** | Fine-tune `BAAI/bge-small-en-v1.5` with `sentence-transformers` MultipleNegativesRankingLoss (in-batch negatives) via `SentenceTransformerTrainer` | Fine-tuned checkpoint + `model_meta.json` in the registry |
| 3 | **Hybrid + rerank + facets** | FAISS dense index; **RRF fusion** (k=60) of dense + BM25; cross-encoder rerank; faceting, KMeans topic clustering, related-paper rails, query expansion | Hybrid engine beats BM25 **and** zero-shot on **nDCG@10** (gate) |
| 4 | **Agent** | Deterministic FSM with 4 decision points (D1 routing, D2 coverage gate, D3 rerank gate, D4 exploration); optional Anthropic LLM brain at D1, OFF by default | Agent runs end-to-end with full audit trace; all 4 decisions fire |
| 5 | **Deployment** | FastAPI service (`/healthz` `/readyz` `/version` `/search` `/related/{paper_id}`); Gradio UI; combined app; CLI; Docker + compose; HF Space | Sub-second hot path; `serve` + UI live; image builds |
| 6 | **Evaluation** | `InformationRetrievalEvaluator` on `BeIR/scifact`; `mteb/scidocs-reranking` secondary; latency profiling; error analysis | nDCG@10 / Recall@{1,5,10} / MRR@10 report + latency numbers |
| 7 | **Docs & report** | README, this plan, model card, risk/ethics notes; auto-generated `report.pdf` + `slides.pptx` | Submission bundle complete |

### 2.2 Indicative timeline (solo, part-time)

Durations are working-day estimates for a single developer; later phases overlap because
deployment and evaluation can proceed against an early checkpoint.

| Phase | Days | Cumulative | Dependency |
|-------|------|------------|------------|
| 0 Research & design | 2 | D2 | — |
| 1 Corpus + pairs + BM25 | 3 | D5 | Phase 0 |
| 2 Retriever fine-tune | 2 | D7 | Phase 1 (pairs) |
| 3 Hybrid + rerank + facets | 3 | D10 | Phase 2 (checkpoint), Phase 1 (BM25) |
| 4 Agent | 2 | D12 | Phase 3 (engine) |
| 5 Deployment | 2 | D14 | Phase 4 (agent) |
| 6 Evaluation | 2 | D16 | Phase 2+3 (can start on early checkpoint) |
| 7 Docs & report | 2 | D18 | All |

> **Training cost note:** fine-tuning `bge-small` is roughly **minutes to ~1h** depending on
> the pair count, so Phase 2 is wall-clock-cheap; the time risk lives in pair generation
> (Phase 1) and the exploration features (Phase 3), not the GPU run. GPU profile scales the
> batch size for more in-batch negatives: **H100 bs256 · A100-40 bs128 · L4 bs64 · T4 bs32 (fp16)**.

### 2.3 Definition of done (the gate that matters)

The project is "working" when, on the held-out `(query → relevant paper)` set:

- **nDCG@10** (headline) of the fine-tuned retriever **>** BM25 **and >** zero-shot base encoder;
- **Recall@{1,5,10}** and **MRR@10** reported alongside (recall, not precision-only, so users see coverage);
- the agent fires all four decisions with a full audit trace (validated offline: e.g. the query *"dense passage retrieval"* ranks the Dense Passage Retrieval paper **#1**, with facets and topic clusters produced);
- the hot path (dense + BM25 + RRF) is **sub-second** end-to-end on a 100k–500k slice.

---

## 3. Task Breakdown — Simulated Roles (Solo Build)

Although this is a solo submission, the work is organised as if staffed by a cross-functional
team. Each "role" is a hat the author wears; the columns map ownership to the concrete modules
under `src/scisearch/`. This makes the solo plan directly translatable to a real team (Section 4).

| Simulated role | Owns (modules) | Core responsibilities | Primary phases |
|----------------|----------------|-----------------------|----------------|
| **ML / IR Engineer** | `models/retriever.py`, `training/` (`train_retriever.py`, `evaluate.py`, `tune.py`, `metrics.py`), `models/reranker.py` | Bi-encoder fine-tune (MNRL, in-batch negatives), eval harness (`InformationRetrievalEvaluator`), nDCG@10/Recall/MRR, cross-encoder rerank, anti-overfitting (dedup, 1–2 epochs) | 2, 3, 6 |
| **Data Engineer** | `data/` (`corpus.py`, `pairs.py`, `download_dataset.py`, `samples.py`) | Load & clean `gfissore/arxiv-abstracts-2021`; facet metadata (category/year/author); generate `(query, positive_paper)` pairs (title→abstract, first-sentence→paper); built-in mini-corpus (~40 NLP papers) offline fallback | 1 |
| **Search / Backend Engineer** | `models/` (`bm25.py`, `vector_store.py`, `model_registry.py`), `search/` (`hybrid.py`, `facets.py`, `expand.py`, `engine.py`), `api/` | FAISS index + numpy fallback; BM25-Okapi; **RRF fusion (k=60)**; faceting, KMeans clustering, related-NN, query expansion; FastAPI endpoints; model registry (`model_meta.json` + latest pointer) | 1, 3, 5 |
| **Agent Engineer** | `agent/` (`state.py`, `policy.py`, `tools.py`, `llm_orchestrator.py`, `search_agent.py`) | Deterministic FSM with D1–D4; intent routing & metadata filters; coverage gate + PRF re-retrieval; margin-gated rerank; exploration strategy; optional LLM brain with rule fallback; audit trace | 4 |
| **Frontend Engineer** | `api/ui.py`, `api/app_combined.py`, `app/` | Gradio UI: search box → ranked results + facet/cluster sidebar + decision log; combined app mounting UI at `/ui` | 5 |
| **QA Engineer** | `tests/`, `analysis/` (`latency.py`, `error_analysis.py`) | Unit/integration tests; offline-mode validation (TF-IDF + identity reranker + built-in corpus, no torch/network); latency profiling; error analysis; zero-result-rate checks | 3, 5, 6 |
| **Project Manager** | `docs/`, `autoreport/`, `automation/`, `grading/`, `Makefile`, `cli.py` | Milestone tracking, this plan, success-metric scorecard, `generate-report`/`generate-slides`/`autopilot`/`grade` orchestration, risk register | 0, 7 |

**CLI as the integration seam.** The console-script `scisearch` exposes every workstream as a
subcommand — `data`, `pairs`, `train`, `tune`, `evaluate`, `search`, `demo-agent`, `serve`,
`benchmark`, `error-analysis`, `monitor`, `generate-report`, `generate-slides`, `autopilot`,
`grade`. In solo work this is how one role hands off to the next; in a team it is the contract
each role builds against.

---

## 4. Reflection — Scaling to a Real Team

The solo plan is deliberately modular so it maps onto a real engineering org with minimal
rework. The role table in Section 3 becomes a set of **parallel workstreams**, each owning a
package boundary that already exists.

### 4.1 Parallel workstreams by module

The dependency graph allows genuine parallelism once Phase 1 lands:

- **Data Eng** and **ML/IR** can run concurrently — the data team grows and curates pair sets
  while the modelling team iterates on loss/batch-size/checkpoints against the current snapshot.
- **Search/Backend** builds the hybrid engine and API against a frozen early checkpoint, so the
  retriever can improve underneath a stable interface (model registry `repo@revision` pin).
- **Agent** and **Frontend** consume the `engine.py` contract and the FastAPI schemas, decoupled
  from how retrieval is implemented.
- **QA** runs cross-cutting against all of the above, gated on the offline-mode invariant.

### 4.2 What a real team would add beyond the solo build

| Concern | Solo build today | Real-team extension |
|---------|------------------|---------------------|
| **CI** | Local `make` + `tests/` | Pipeline runs tests + an **eval gate**: a PR cannot merge if nDCG@10 regresses vs the registered baseline |
| **Model registry** | `model_meta.json` + latest pointer, `repo@revision` | Promote/rollback workflow, staging vs prod aliases, lineage from pair-set version → checkpoint |
| **Indexing pipeline** | FAISS over a 100k–500k slice | **Sharded FAISS index** + cached embeddings to scale to **millions of papers**; distributed embedding jobs; incremental shard rebuilds |
| **Relevance-feedback loop** | PRF inside D2; offline qrels | Capture click/dwell signals → continuously mined hard negatives → periodic re-fine-tune (closing the loop the eval harness only approximates) |
| **On-call / monitoring** | `monitoring/`, `analysis/latency.py` | Latency/error SLOs, dashboards, alerting on zero-result-rate and tail latency; on-call rotation; canary the rerank gate |
| **Ingestion / freshness** | 2021 snapshot, documented cutoff | **Continual ingestion** of new arXiv papers to kill the stale-index risk; scheduled re-index + backfill |

### 4.3 Cross-cutting practices

- **Latency budget as a shared contract.** The hot path (dense + BM25 + RRF) is tens of ms on a
  FAISS slice; rerank is **gated** (D3) and adds +150–400ms only when the head of the ranking is
  ambiguous. In a team this budget is a tracked SLO, not a one-off measurement.
- **Scalability levers already designed in.** FAISS index sharding and cached embeddings are part
  of the deployment story now, so scaling to millions of papers is an ops project, not a rewrite.
- **Graceful degradation as a team safety net.** The offline path (TF-IDF retriever + identity
  reranker + built-in corpus) means the demo and CI always run even when torch/network/GPU are
  absent — valuable when many engineers run the stack locally.
- **Privacy and ethics owned, not bolted on.** Search queries can reveal research direction, so
  raw-query retention is minimized and an on-prem option exists; the UI shows scores, diversity
  and broaden/narrow suggestions to counter over-trust in the top result. These are PM-owned
  acceptance criteria, not afterthoughts.

---

## 5. Success Metrics (tracked across the plan)

| Type | Metric | Target / direction |
|------|--------|--------------------|
| Business | Time-to-first-relevant-result | Down vs keyword search |
| Business | Exploration depth | ≥1 facet/cluster/related interaction per session |
| Business | Zero-result rate | < 2% (D2 coverage gate backstops) |
| Business | Corpus coverage | All arXiv fields |
| Technical | **nDCG@10** (headline) | Fine-tuned retriever beats BM25 **and** zero-shot |
| Technical | Recall@{1,5,10}, MRR@10 | Reported (coverage, not precision-only) |
| Technical | Latency | Sub-second end-to-end hot path |

These metrics close the loop with Section 2.3: the milestone gate, the eval phase and the
team-scale relevance-feedback loop all measure the same headline number, **nDCG@10**.
