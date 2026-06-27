# Continual Learning & Monitoring

**Project:** Exploratory Scientific Literature Search System (P08)
**Author:** Le Dinh Minh Quan (23127460) — NLP in Industry, final assignment
**Package:** `src/scisearch/`

This document describes how the system stays fresh and accurate **after** the first model ships: how new data is collected, how the retriever is periodically retrained and versioned, how the index is kept current despite a hard data cutoff, and how we detect degradation through monitoring metrics and drift signals.

The core trainable component is a **fine-tuned bi-encoder** (`BAAI/bge-small-en-v1.5`, 33.4M params, 384-dim; fallback `sentence-transformers/all-MiniLM-L6-v2`) trained with `sentence-transformers` `MultipleNegativesRankingLoss`. The corpus is `gfissore/arxiv-abstracts-2021` (2.0M rows, **recency cutoff 2021**). Both facts drive everything below: the model can stale (terminology drift) and the index can stale (missing post-2021 papers).

---

## 1. New-Data Collection

Two independent streams feed the continual-learning loop. They are deliberately decoupled: one keeps the **index** current, the other keeps the **ranker** current.

### 1.1 Incremental paper ingestion (index freshness)

The primary corpus is a static 2021 snapshot. To cover papers published after the cutoff, we ingest new arXiv records incrementally rather than rebuilding from scratch.

| Step | Action |
|------|--------|
| Harvest | Pull new arXiv records (same schema as the corpus: `id, authors, title, abstract, categories` (LIST), `versions`). |
| Dedup | Drop records whose `id` already exists in the index (arXiv ids are stable); reconcile `versions` so revised papers update in place. |
| Embed | Encode `title + abstract` with the **current registered retriever** (so new vectors share the live embedding space) and L2-normalize. |
| Append | Add vectors to the FAISS index (`faiss-cpu`, inner product = cosine) and the BM25-Okapi token store; refresh facet metadata (category → field name, year, author). |

Because faceting, topic clustering, and related-paper rails all read from the same store, an append automatically extends exploration coverage — no separate pipeline. Appending vectors with the live model avoids embedding-space mismatch; a **full re-embed** is only required when the retriever itself is upgraded (see §2.3).

### 1.2 Feedback harvesting (training pairs)

The deployed search loop is also a data source. Each `POST /search` returns `results[]`, `facets[]`, `clusters[]`, `suggestions`, `decisions`, and `metrics`; the UI logs which result the user opens or marks relevant. This yields implicit **`(query, clicked_paper)` training pairs** that complement the originally **generated** pairs:

- Generated (cold-start) pairs: `title → abstract` (lexical) and `first-sentence-of-abstract → paper` (conceptual), with in-batch negatives.
- Harvested (warm) pairs: real `(query, clicked_paper_id)` — true positives reflecting actual user intent and phrasing, which the synthetic pairs cannot capture.

**Privacy constraint (carried from the risk register):** search queries can reveal a researcher's direction and are treated as sensitive. We minimize retention/logging of raw queries — aggregate or hash where possible, support an on-prem deployment, and prefer storing the `(query_id, paper_id, rank, position)` tuple needed for training over verbatim query text wherever feasible.

---

## 2. Retraining

### 2.1 Trigger policy

Retraining is **periodic** (e.g. scheduled cadence) and **event-driven** (fired early when monitoring crosses a threshold — see §4). A run begins only when there is a meaningful volume of fresh feedback or a degradation signal, to avoid churning the registry.

### 2.2 Training recipe (unchanged from initial training)

We re-fine-tune on **harvested feedback pairs + fresh synthetic pairs** generated from newly ingested papers, keeping the same regime that produced the shipped model:

- Loss: `MultipleNegativesRankingLoss` (in-batch negatives) via `SentenceTransformerTrainer` / `SentenceTransformerTrainingArguments`.
- Large `per_device_train_batch_size` (128–256) → more in-batch negatives → stronger contrastive signal. GPU profile: H100 bs256 (bf16+tf32), A100-40 bs128, L4 bs64, T4 bs32 (fp16).
- `lr ~2e-5`, **1–3 epochs**, `warmup_ratio 0.1`, `batch_sampler "no_duplicates"`, resume via `get_last_checkpoint`.
- **Anti-overfitting:** large diverse pair set, dedup, 1–2 epochs, in-batch negatives — especially important as harvested pairs can be skewed toward popular queries.

### 2.3 Re-embedding after a model upgrade

A new retriever produces a **new embedding space**, so the FAISS index must be re-embedded with the candidate model before it can serve. This is the expensive path (full corpus re-encode) and is gated behind the registry/canary process so it only runs for a model that has already passed offline evaluation.

### 2.4 Registry versioning

Every model is versioned through the existing model registry: `model_meta.json` + a **`latest` pointer**, identified by `repo@revision`. A new fine-tune is registered as a new revision; the `latest` pointer is only advanced after canary/AB validation. This makes rollback a pointer move and keeps the served model id auditable (also surfaced at `GET /version`).

### 2.5 Canary / AB rollout

```
new fine-tune ──► offline eval (held-out qrels)
                       │  must beat current on nDCG@10
                       ▼
                  register revision (model_meta.json)
                       │
                       ▼
                  CANARY: small % of /search traffic
                       │  compare live nDCG (golden set), CTR, latency, zero-result rate
                       ▼
            ┌──────────┴───────────┐
       regression?              improvement?
            │                        │
        rollback                 advance `latest`
     (move pointer back)        (move pointer fwd)
```

A candidate must beat the current production model **offline** (see §2.6) before any live traffic, then prove out on a **canary** slice / **AB** test against the monitoring metrics in §4 before the `latest` pointer advances. Either way the decision is reversible by moving the registry pointer.

### 2.6 Offline gate — must beat the baselines

Every candidate is evaluated with the same self-contained evaluator / `InformationRetrievalEvaluator` used originally, on a held-out `(query → relevant paper)` set:

- **Headline:** `nDCG@10`. Also `Recall@{1,5,10}`, `MRR@10`.
- Real eval set: `BeIR/scifact` (corpus + queries + qrels); `mteb/scidocs-reranking` secondary.
- A candidate must beat **both** baselines it was built to beat: **BM25** (self-contained Okapi) and the **zero-shot base encoder** (un-fine-tuned bge/MiniLM) — and must not regress on **conceptual** queries, where dense retrieval is the differentiator over BM25.

---

## 3. Index Freshness (the 2021 cutoff)

The primary corpus stops at **2021**, so papers published after the snapshot are simply absent — a documented stale-index risk. Mitigation is the incremental ingestion of §1.1 run on a **schedule** (scheduled re-indexing): new arXiv records are harvested, deduped by `id`, embedded with the live model, and appended to FAISS + BM25.

Operational notes:

- **Hot path is unaffected:** appends extend the existing index; the dense + BM25 + RRF (`k=60`) fusion and the `cross-encoder/ms-marco-MiniLM-L-6-v2` reranker are untouched.
- **Scalability:** the index grows over time, handled via FAISS index sharding + cached embeddings.
- **Offline fallback unchanged:** the built-in ~40-paper NLP mini-corpus (and the TF-IDF retriever standing in for the dense slot) keep the demo runnable with no torch/network — useful when validating an ingestion run in isolation.

---

## 4. Monitoring Metrics

We track a small dashboard of signals on live traffic and a golden set. Each maps to a concrete failure mode.

| Metric | What it measures | Why it matters / target |
|--------|------------------|--------------------------|
| **Zero-result rate** | Share of queries returning nothing useful | Business target **< 2%**; the **D2 coverage gate** (expand + re-retrieve) is the backstop. A rise means the index or expansion is failing. |
| **Intent mix** | Distribution of D1 routing (keyword / conceptual / hybrid) | Detects **query-distribution shift**. A sudden swing changes which retriever (BM25 vs dense) carries the load. |
| **Latency** | End-to-end and per-stage time | Hot path (dense+BM25+RRF) ~tens of ms on a 100k–500k FAISS slice; reranker gated (D3) adds +150–400ms; target **sub-second** end-to-end. |
| **nDCG@10 on a golden query set** | Ranking quality on a fixed, labeled set re-scored every run | Headline quality signal; a drop is the primary **degradation** trigger. Reuses the offline evaluator. |
| **Click-through (CTR)** | Whether users click returned results (and at what rank) | Live proxy for real relevance; feeds the feedback harvester (§1.2) and the AB comparison. |

**Degradation detection.** The golden query set gives a stable yardstick: re-score `nDCG@10` on it on each monitoring cycle and on every canary. A sustained drop there — or a rising zero-result rate, a shifting intent mix, or falling CTR — fires an early retraining trigger (§2.1) and/or a canary rollback (§2.5). Latency regressions are caught against the sub-second budget. Because the golden set is fixed, a quality drop on it isolates **model/index** regressions from genuine changes in user behaviour.

---

## 5. Drift Risks & Mitigation

| Drift risk | Symptom in monitoring | Mitigation |
|------------|----------------------|------------|
| **New research areas / terminology** | Rising zero-result rate; queries with novel terms route oddly; nDCG dip on newer topics | Incremental ingestion brings the papers in (§3); periodic retraining on fresh synthetic + harvested pairs re-aligns the embedding space (§2); query expansion (abbreviation map + pseudo-relevance feedback) bridges unseen terms at D2. |
| **Query-distribution shift** | Intent-mix swing; CTR change at fixed ranking quality | D1 intent routing adapts BM25/dense weighting per query; the **harvested feedback** retrains the ranker toward the new query mix; AB-validate before promoting. |
| **Stale index** (post-2021 gap) | Recent papers never appear; zero-result on current-events queries | Scheduled re-indexing (§3); recency cutoff documented for users; "broaden"/diversity suggestions so the top-k is never presented as exhaustive. |
| **Over-trust in ranking** | (UX risk, not a metric drift) | Show scores + result diversity + broaden/narrow suggestions (D4); report **Recall@k** so users see coverage, not precision alone. |

### 5.1 Drift report — `src/scisearch/monitoring/drift_report.py`

Drift surveillance is implemented in **`src/scisearch/monitoring/drift_report.py`** (under the `monitoring/` package, exposed via the `monitor` / `generate-report` CLI commands). It periodically computes and persists the signals above so trends — not just point values — are visible:

- **Quality drift:** `nDCG@10` (+ `Recall@{1,5,10}`, `MRR@10`) on the fixed golden query set, compared against the last known-good baseline.
- **Behaviour drift:** zero-result rate, intent-mix distribution, click-through — compared period-over-period to flag query-distribution shift.
- **Performance drift:** end-to-end and per-stage latency against the sub-second budget (including how often the D3 rerank gate fires).
- **Output:** a structured drift report that flags threshold breaches and, when a breach is sustained, signals the retraining trigger (§2.1) / canary rollback (§2.5).

---

## 6. End-to-End Lifecycle

```
new arXiv papers ──► incremental ingest ──► FAISS + BM25 append (live model)   [index freshness]
                                                   │
user search (/search) ──► clicks ──► (query, clicked_paper) pairs              [feedback]
                                                   │
              + fresh synthetic pairs from new papers
                                                   ▼
                          periodic / triggered re-fine-tune (MNRL)             [retraining]
                                                   ▼
                       register revision (model_meta.json, repo@revision)
                                                   ▼
                offline gate (nDCG@10 > BM25 & zero-shot) ─► canary/AB
                                                   ▼
                          advance `latest`  ◄──┐
                                               │ rollback on regression
        drift_report.py monitors nDCG@10 (golden set), zero-result rate,
        intent mix, latency, CTR ──► triggers retraining / rollback
```

This loop keeps **both** the index current (continual ingestion past the 2021 cutoff) and the retriever current (feedback-driven retraining), with the registry + canary process ensuring every change is measured against `nDCG@10` and the live monitoring metrics before it reaches users, and reversible if it regresses.
