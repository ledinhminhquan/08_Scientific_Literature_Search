# Agent Architecture — Exploratory Scientific Literature Search (P08)

**Project:** Exploratory Scientific Literature Search System
**Author:** Le Dinh Minh Quan (student 23127460)
**Course:** NLP in Industry — final assignment
**Package:** `src/scisearch/agent/`
**Reference:** github.com/NLP-Knowledge-Graph/NLP-KG-WebApp

---

## 1. Overview

The agent is the mandatory agentic component that turns a flat retrieval pipeline into an *exploratory* search experience. It is implemented as a **deterministic finite-state machine (FSM)** with **four decision points (D1–D4)**, an optional **LLM query-understanding brain** at D1, and a **full audit trace** of every decision.

The agent does not "free-roam." Every transition is governed by explicit thresholds over a shared `JobState`, so the same query always produces the same path (reproducible, debuggable, gradeable). The optional Anthropic LLM only advises at D1 and is **OFF by default** — the agent therefore incurs **zero paid API calls** unless explicitly enabled, and even then it *validates and falls back to rules* if the LLM output is malformed.

The pipeline encodes the classic exploratory-search loop:

> **understand** (D1) → **retrieve + RRF fuse** (D2 coverage gate → expand) → **rerank** (D3 gate) → **explore** (D4: facets + clusters + related).

| File | Role |
| --- | --- |
| `agent/state.py` | `JobState` — the shared context object threaded through every stage |
| `agent/policy.py` | The four decision functions D1–D4 (pure thresholds, no I/O) |
| `agent/tools.py` | Tool contracts: `tool_understand`, `tool_retrieve`, `tool_rerank`, `tool_explore` |
| `agent/llm_orchestrator.py` | Optional Anthropic brain for D1 query understanding (validate + fallback) |
| `agent/search_agent.py` | The FSM driver that wires state → policy → tools → audit trace |

---

## 2. JobState — the shared context

`JobState` is the single object passed between FSM states. Each tool reads the fields it needs and writes its outputs back, so the state accumulates a complete record of the run. Conceptually it holds:

| Group | Fields (conceptual) | Written by |
| --- | --- | --- |
| **Query** | `raw_query`, `expanded_query`, `intent` (`keyword` / `conceptual` / `hybrid`), `filters` (e.g. field `cs.CL`, year) | D1 / `tool_understand` |
| **Retrieval** | `dense_ranking`, `bm25_ranking`, `fused_ranking` (RRF), `top_raw_score`, `n_results` | `tool_retrieve` |
| **Rerank** | `head_margin`, `reranked`, `final_ranking` | D3 / `tool_rerank` |
| **Exploration** | `facets`, `clusters`, `related`, `suggestions` (broaden / narrow / related) | `tool_explore` |
| **Audit** | `decisions[]` (one record per D1–D4: inputs, threshold, branch taken, fallback flag), `metrics` | every stage |

The audit trace is not a side-channel log — it is part of the returned payload (`decisions` in the `/search` response), so a grader or user can inspect *why* each branch fired.

---

## 3. The four decision points (D1–D4)

Each decision is a pure function in `policy.py`: it takes fields from `JobState`, compares them against a threshold, and returns a branch. This keeps the policy testable in isolation and the thresholds tunable in one place.

### D1 — Query-type routing (intent + filters)

| | |
| --- | --- |
| **Inputs** | `raw_query` (length, presence of wh-words), optional LLM hint |
| **Logic** | Classify intent: short / exact-term queries → **keyword**; long / wh-word / concept-phrased queries → **conceptual**; mixed → **hybrid**. Extract metadata **filters** (arXiv field e.g. `cs.CL`, year). |
| **Output / effect** | Sets `intent` + `filters`, which **bias the RRF fusion**: keyword → up-weight the BM25 ranking; conceptual → up-weight the dense ranking. (Fusion bias is applied by duplicating the preferred ranking before RRF.) |
| **Branches** | `keyword` \| `conceptual` \| `hybrid` |
| **Brain / fallback** | If the LLM brain is enabled, it proposes intent + filters; the result is **schema-validated**, and on any failure the agent **falls back to the deterministic rules**. LLM OFF by default. |

### D2 — Retrieval-coverage gate (expand?)

| | |
| --- | --- |
| **Inputs** | `n_results`, `top_raw_score` (raw dense/BM25 similarity of the head doc) |
| **Threshold** | Fire if **`n_results` is too few** OR **`top_raw_score` < coverage threshold** |
| **Action on fire** | **Expand the query** — abbreviation map (e.g. `DPR` → "dense passage retrieval") **+** pseudo-relevance feedback (PRF) terms harvested from the current top docs — then **re-retrieve** once |
| **Branches** | `coverage_ok` (proceed) \| `expand_and_retry` (expand, re-retrieve, continue) |
| **Why** | Backstops the **zero-result rate (<2%)** and rescues out-of-domain / garbled queries |

### D3 — Rerank gate (call the cross-encoder?)

| | |
| --- | --- |
| **Inputs** | `head_margin` = score gap among the top of `fused_ranking` |
| **Threshold** | Rerank **only when the head is close / ambiguous** (small margin) |
| **Action on fire** | Run `cross-encoder/ms-marco-MiniLM-L-6-v2` over the top-k to re-score; otherwise skip |
| **Branches** | `rerank` (ambiguous head) \| `skip_rerank` (unambiguous head) |
| **Why** | Reranking costs **+150–400 ms**; skipping it when the top is already decisive keeps the hot path at **tens of ms** |

### D4 — Exploration / presentation strategy

| | |
| --- | --- |
| **Inputs** | `final_ranking`, result **diversity** (spread of fields / cluster count) |
| **Action** | **Facet** by arXiv category → readable field names; **cluster** results into topics (KMeans over TF-IDF, labeled by top terms); **fetch related** (dense nearest-neighbors of a paper) |
| **Output** | `facets`, `clusters`, `related`, and **broaden / narrow / related suggestions** chosen by diversity (e.g. low diversity → "broaden"; high diversity → "narrow" into a cluster) |
| **Branches** | suggestion strategy = `broaden` \| `narrow` \| `related` |

---

## 4. Tool contracts

The FSM driver calls four tools. Each has a narrow contract: it reads named fields from `JobState` and writes named fields back. This is what lets the optional LLM brain (and the deterministic policy) reason over a stable interface.

```text
tool_understand(raw_query)
    -> intent ∈ {keyword, conceptual, hybrid}, filters{field?, year?}
    (consumed by D1; may be advised by the LLM brain, always validated)

tool_retrieve(query, intent, filters)
    -> dense_ranking, bm25_ranking
    -> fused_ranking via Reciprocal Rank Fusion:  score(d) = Σ_r 1/(k + rank_r(d)), k=60
       (intent biases fusion by duplicating the preferred ranking)
    -> top_raw_score, n_results      (consumed by D2)

tool_rerank(fused_ranking, query)        # only invoked when D3 fires
    -> reranked top-k via cross-encoder/ms-marco-MiniLM-L-6-v2
    -> final_ranking                     (identity fallback if reranker absent)

tool_explore(final_ranking)
    -> facets (arXiv category -> field name)
    -> clusters (KMeans / TF-IDF, top-term labels)
    -> related (dense nearest-neighbors)
    -> suggestions (broaden / narrow / related)   (driven by D4)
```

**Graceful degradation is built into the contracts.** When `torch`/network are absent, the retriever slot is filled by a **TF-IDF retriever** (sklearn), the reranker degrades to **identity**, and retrieval runs over the **built-in ~40-paper NLP mini-corpus** — so the agent always completes end-to-end offline.

---

## 5. Optional LLM brain (Anthropic) at D1

The LLM brain is a thin advisor wired only into **D1 query understanding** (`llm_orchestrator.py`):

1. The brain is asked to produce a structured understanding of the query: intent + metadata filters.
2. The output is **schema-validated**.
3. On **any** validation failure (or when the brain is disabled), the agent **falls back to the deterministic rule-based D1**.

It is **OFF by default**, guaranteeing **0 paid API usage** in the default configuration and full offline operation. The brain can only *refine* D1; it never replaces the deterministic D2–D4 gates, so latency budgets and reproducibility for the rest of the pipeline are unaffected.

---

## 6. ASCII flow diagram

```text
                      ┌──────────────────────────────┐
                      │   raw_query  →  JobState      │
                      └───────────────┬──────────────┘
                                      │
                         ┌────────────▼────────────┐
                         │  D1  UNDERSTAND          │
                         │  tool_understand         │
                         │  intent? + filters?      │
                         │  (LLM brain → validate   │
                         │   → fallback to rules)    │
                         └────────────┬─────────────┘
                  keyword / conceptual / hybrid  (biases RRF)
                                      │
                         ┌────────────▼─────────────┐
                         │  RETRIEVE + FUSE          │
                         │  tool_retrieve            │
                         │  dense ⊕ BM25 → RRF(k=60) │
                         └────────────┬─────────────┘
                                      │
                         ┌────────────▼─────────────┐
                         │  D2  COVERAGE GATE        │
                         │  n_results too few? OR    │
                         │  top_raw_score < thresh?  │
                         └───────┬───────────┬───────┘
                            yes  │           │  no (coverage_ok)
                                 ▼           │
                    ┌────────────────────┐   │
                    │ EXPAND query        │  │
                    │ abbrev + PRF        │  │
                    │ → re-retrieve once  │  │
                    └─────────┬───────────┘   │
                              └───────┬────────┘
                                      │
                         ┌────────────▼─────────────┐
                         │  D3  RERANK GATE          │
                         │  head_margin small?       │
                         └───────┬───────────┬───────┘
                            yes  │           │  no (skip_rerank)
                                 ▼           │
                    ┌────────────────────┐   │
                    │ tool_rerank         │  │
                    │ cross-encoder       │  │
                    │ ms-marco-MiniLM-L-6 │  │
                    └─────────┬───────────┘   │
                              └───────┬────────┘
                                      │
                         ┌────────────▼─────────────┐
                         │  D4  EXPLORE / PRESENT    │
                         │  tool_explore             │
                         │  facets + clusters +      │
                         │  related + suggestions    │
                         │  (broaden/narrow/related  │
                         │   by diversity)           │
                         └────────────┬─────────────┘
                                      │
                      ┌───────────────▼──────────────┐
                      │ {results, facets, clusters,   │
                      │  suggestions, decisions[],    │
                      │  metrics}   ← audit trace      │
                      └───────────────────────────────┘
```

---

## 7. Worked example — a conceptual query

**Query:** `"dense passage retrieval"`

| Stage | What happens | State written |
| --- | --- | --- |
| **D1 understand** | The query is a concept phrase (no exact-term keyword signal, no field filter). Routed **conceptual** → fusion will up-weight the **dense** ranking. No metadata filters extracted. | `intent = conceptual` |
| **Retrieve + RRF** | Dense bi-encoder (`BAAI/bge-small-en-v1.5`, fine-tuned) and BM25 each return a ranking; they are fused with RRF (`k=60`), with the dense ranking duplicated to honor the conceptual bias. | `fused_ranking`, `top_raw_score`, `n_results` |
| **D2 coverage gate** | Plenty of results and a strong head similarity → **coverage_ok**. No expansion needed (the abbreviation/PRF path is *not* taken). | branch = `coverage_ok` |
| **D3 rerank gate** | The head of the fused ranking is **close/ambiguous** (small `head_margin`) → **rerank**. The cross-encoder `cross-encoder/ms-marco-MiniLM-L-6-v2` re-scores the top-k, sharpening the head. | branch = `rerank`; `final_ranking` |
| **D4 explore** | Results clustered into sub-topics (KMeans over TF-IDF, top-term labels), faceted by arXiv field, and **related** papers fetched via dense nearest-neighbors. Diversity drives a **related**-style suggestion. | `facets`, `clusters`, `related`, `suggestions` |

**Outcome (validated offline):** correct intent routing makes the **Dense Passage Retrieval paper rank #1**, all four decisions fire and are recorded in `decisions[]`, and the response carries facets + topic clusters + related rails plus the suggestion strip.

---

## 8. Why this design

- **Determinism first.** Thresholded D1–D4 over a shared `JobState` make the pipeline reproducible and gradeable; the optional LLM only advises D1 and always validates + falls back.
- **Latency-aware.** D3 gates the expensive cross-encoder (**+150–400 ms**) so the hot path (dense + BM25 + RRF over a FAISS slice) stays at **tens of ms**; end-to-end is **sub-second**.
- **Robust by construction.** D2 backstops the **zero-result rate (<2%)** and rescues out-of-domain/garbled queries; the tool contracts degrade gracefully (TF-IDF retriever + identity reranker + built-in mini-corpus) when `torch`/network are unavailable.
- **Exploratory, not just ranked.** D4 is what makes the system "exploratory" — facets, clusters, related rails, and broaden/narrow guidance counter over-trust in a single ranked list and surface coverage, not just precision.
