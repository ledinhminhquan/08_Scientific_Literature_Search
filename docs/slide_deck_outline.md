# P08 — Exploratory Scientific Literature Search · Slide-Deck Outline

**Presentation outline for the final-assignment submission.** 13 slides, aligned to the report and the assignment's required slide list. Each slide lists a title, 3–6 concise bullets, and a note on the intended visual/diagram.

> Style guidance: keep one idea per slide; let the architecture diagram (Slide 4) and results table (Slide 6) carry the technical weight. Speaker can demo the live Gradio UI between Slides 6 and 7.

---

## Slide 1 — Title & Author

- **Exploratory Scientific Literature Search System** — find *and* explore scientific papers.
- Le Dinh Minh Quan — Student ID **23127460**.
- Course: NLP in Industry — Final Assignment.
- One-line pitch: a query returns not just a ranked list, but **facets, topic clusters, related-paper rails, and broaden/narrow guidance**, backed by a fine-tuned dense retriever.
- Reference: `github.com/NLP-Knowledge-Graph/NLP-KG-WebApp`; package `src/scisearch/`.

> **Visual:** Title slide with system name, author, ID, course; subtle background montage of a ranked result list + facet sidebar mock.

---

## Slide 2 — Business Problem & Motivation

- Researchers need to **find AND explore** literature, not just match keywords.
- Keyword search (BM25) is precise on exact terms but **blind to concept** — paraphrase and conceptual queries fail.
- The search engine **is** an NLP/IR problem: semantic matching of scientific phrasing, query understanding, topic clustering.
- Success targets: lower **time-to-first-relevant-result**, **exploration depth** (≥1 facet/cluster/related interaction per session), **zero-result rate < 2%**, full arXiv-field coverage.
- A trained **dense retriever** fixes the concept gap that keyword search leaves open.

> **Visual:** Two-column "Keyword vs. Concept" contrast — a conceptual query that BM25 misses and dense retrieval catches.

---

## Slide 3 — Proposed NLP Solution

- **Fine-tuned bi-encoder retriever** maps query and paper into a shared embedding space (dense passage retrieval).
- **Hybrid search**: dense + BM25 fused with **Reciprocal Rank Fusion (RRF)**, plus a gated cross-encoder reranker.
- **Exploration layer**: faceting by field, topic clustering, related-paper rails, query expansion.
- **Agentic controller**: a deterministic FSM with 4 decision points routes intent, gates coverage/rerank, and picks the presentation strategy.
- Designed to **beat two baselines**: BM25 and the zero-shot (un-fine-tuned) base encoder.

> **Visual:** Four labeled pillars — Retriever · Hybrid Fusion · Facets/Exploration · Agent — feeding into a single result panel.

---

## Slide 4 — System Architecture Diagram

- **Query → Agent (D1 routing)** detects intent + extracts metadata filters (field, year).
- **Retrieval**: FAISS dense (L2-normalized, inner product = cosine) ∥ self-contained BM25-Okapi → **RRF fusion (k=60)**.
- **Gated reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2` re-scores top-k only when the head is ambiguous (D3).
- **Exploration**: facets (arXiv category → field names), KMeans/TF-IDF topic clusters, dense-NN related papers, query expansion (D2/D4).
- **Serving**: FastAPI + Gradio UI + CLI, FAISS-backed, with full decision/audit trace returned alongside results.

> **Visual:** Left-to-right block diagram: Query → Agent FSM → [Dense ∥ BM25 → RRF → Reranker] → Exploration (facets/clusters/related) → Response `{results, facets, clusters, suggestions, decisions, metrics}`.

---

## Slide 5 — Data Overview

- **Corpus:** `gfissore/arxiv-abstracts-2021` (CC0-1.0, **2.0M rows**; fields: id, authors, title, abstract, categories (list), versions → enables field/year/author facets).
- **Training pairs (generated):** (query, positive_paper) from the corpus — *title → abstract* (lexical) and *first-sentence-of-abstract → paper* (conceptual); **in-batch negatives** supply the rest.
- **Real evaluation:** `BeIR/scifact` (corpus + queries + qrels) as a drop-in for `InformationRetrievalEvaluator`; `mteb/scidocs-reranking` secondary.
- **Offline fallback:** a built-in mini-corpus of ~40 real NLP papers so the demo always runs without network/torch.

| Asset | Source | License | Role |
|---|---|---|---|
| Corpus | `gfissore/arxiv-abstracts-2021` | CC0-1.0 | Index + facets (2.0M rows) |
| Train pairs | Generated from corpus | — | Contrastive fine-tuning |
| Eval set | `BeIR/scifact` | — | nDCG/Recall/MRR |

> **Visual:** Data-flow strip — arXiv corpus → generated pairs → fine-tune → BeIR/scifact held-out eval.

---

## Slide 6 — Model & Evaluation Results

- **Retriever:** `BAAI/bge-small-en-v1.5` (MIT, 33.4M params, 384-dim) primary; fallback `sentence-transformers/all-MiniLM-L6-v2` (Apache, 22.7M, 384-dim).
- **Training:** `sentence-transformers` `MultipleNegativesRankingLoss`, large batch (128–256 → more in-batch negatives), lr ~2e-5, 1–3 epochs, warmup 0.1, `batch_sampler="no_duplicates"`.
- **Metrics:** **nDCG@10 (headline)**, Recall@{1,5,10}, MRR@10 via `InformationRetrievalEvaluator`.
- **Baselines beaten:** BM25 (Okapi) and the **zero-shot base encoder** — gap largest on **conceptual queries** (dense ≫ BM25).
- Anti-overfitting: large, deduped, diverse pair set + 1–2 epochs + in-batch negatives.

| System | nDCG@10 | Recall@{1,5,10} | MRR@10 |
|---|---|---|---|
| BM25 (Okapi) | baseline | baseline | baseline |
| Zero-shot encoder | baseline | baseline | baseline |
| **Fine-tuned bi-encoder** | **beats both (headline)** | higher coverage | higher |

> **Visual:** Grouped bar chart — nDCG@10 for BM25 vs zero-shot vs fine-tuned; annotate the conceptual-query lift. (Use measured numbers from the evaluation run.)

---

## Slide 7 — Agentic AI Component

- Deterministic **finite-state machine** with an **optional LLM brain** (off by default → **$0 paid API**); 4 decision points, full audit trace.
- **D1 — Query routing:** detect intent (keyword vs conceptual vs hybrid by length/wh-words) + extract filters (e.g. `cs.CL`, year) → bias retrieval weighting.
- **D2 — Coverage gate:** too few results or top raw similarity < threshold → expand query (abbreviation map + pseudo-relevance feedback) and re-retrieve; backstops zero-result rate.
- **D3 — Rerank gate:** run the cross-encoder **only** when the head margin is small/ambiguous → saves latency when the top is clear.
- **D4 — Exploration strategy:** cluster into topics, facet by field, fetch related, suggest broaden/narrow/related from result diversity.

> **Visual:** FSM state diagram with the four decision nodes D1→D2→D3→D4 and their branch conditions; sidebar showing a sample decision log.

---

## Slide 8 — Live Demo (Validated Offline)

- Agent runs **end-to-end on the built-in corpus** — no network or torch required.
- **All 4 decisions fire**; intent routing is correct (e.g. *"dense passage retrieval"* → the Dense Passage Retrieval paper ranks **#1**).
- Facets + topic clusters are produced and shown in the UI sidebar alongside ranked results.
- Decision log is surfaced to the user for transparency.
- Artifacts auto-generate: `report.pdf` and `slides.pptx`.

> **Visual:** Annotated Gradio UI screenshot — search box, ranked results, facet/cluster sidebar, decision log. (Optionally a live demo here.)

---

## Slide 9 — Deployment Overview

- **FastAPI** (`api/main.py`): `GET /healthz /readyz /version`, `POST /search` → `{results, facets, clusters, suggestions, decisions, metrics}`, `GET /related/{paper_id}`.
- **Gradio UI** (`api/ui.py`) + combined app mounting the UI at `/ui`; **CLI** (`scisearch`: data, pairs, train, evaluate, search, demo-agent, serve, benchmark, generate-report/slides, …).
- **Containerized:** Docker (`python:3.11-slim`), docker-compose, HF Space (Gradio).
- **Latency:** hot path (dense + BM25 + RRF) **~tens of ms** on a 100k–500k slice via FAISS; gated rerank +150–400 ms; **end-to-end sub-second**.
- **Scale & versioning:** FAISS index sharding + cached embeddings; model registry (`model_meta.json` + latest pointer, `repo@revision`).

> **Visual:** Deployment topology — Docker container → FastAPI + Gradio; latency budget callout (hot path vs gated rerank vs total).

---

## Slide 10 — Ethics, Privacy & Risks

- **Query privacy:** search queries can reveal research direction → minimize raw-query retention/logging; offer on-prem deployment.
- **Over-trust in ranking:** users may treat top results as exhaustive/authoritative → show scores + diversity + "broaden" suggestions; never hide that it's a ranked sample.
- **Stale index:** primary corpus snapshot is **2021** — newer papers are missing → document the cutoff and a continual-ingestion / incremental-update plan.
- **Corpus bias/coverage:** arXiv-only, English; report **Recall@k** so users understand coverage, not just precision.
- **Dual-use:** system indexes **only public papers**; robustness via the D2 coverage gate and graceful degradation (TF-IDF retriever + identity reranker + built-in corpus when torch/network absent).

> **Visual:** Risk matrix (likelihood × impact) tagging each risk with its mitigation; highlight privacy, over-trust, and stale-index.

---

## Slide 11 — Key Takeaways

- A fine-tuned dense retriever **closes the concept gap** that keyword search leaves — and beats both BM25 and the zero-shot encoder on **nDCG@10**.
- **Hybrid (RRF) + gated reranker** balances quality and latency; sub-second end-to-end.
- The **4-decision agent** makes retrieval adaptive: route, recover, rerank-when-needed, explore.
- Exploration (facets, clusters, related, broaden/narrow) turns search into **discovery**.
- Fully runnable offline; deployable via FastAPI / Gradio / CLI / Docker with a model registry.

> **Visual:** Five-icon takeaway strip mirroring the bullets.

---

## Slide 12 — Future Work

- **Continual ingestion** to fix the 2021 cutoff — incremental index updates beyond the snapshot.
- **Domain-specialized encoders** (e.g. `malteos/scincl`, `allenai/specter2_base`, 768-dim) for scientific text.
- Enable the **optional LLM brain** at D1 for richer query understanding, with rule-based validation/fallback.
- Broaden evaluation beyond `BeIR/scifact` (e.g. `mteb/scidocs-reranking`) and add per-intent error analysis.
- Harden privacy posture: configurable retention, on-prem packaging, query-logging controls.

> **Visual:** Roadmap timeline — ingestion → domain encoders → LLM-assisted routing → broader eval.

---

## Slide 13 — Q&A / Backup

- Architecture deep-dive (D1–D4 internals, RRF fusion math `1/(k + rank)`, k=60).
- Training profile by GPU: H100 bs256 · A100-40 bs128 · L4 bs64 · T4 bs32 (fp16).
- Repo layout (`src/scisearch/`: data, models, search, training, agent, api, analysis, autoreport, monitoring, automation, grading).
- Full metric definitions and offline-fallback behavior.

> **Visual:** "Thank you / Questions" slide with contact + repo reference; backup slides hidden behind it.
