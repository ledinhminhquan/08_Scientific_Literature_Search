# P08 — Exploratory Scientific Literature Search: Problem Definition

**Author:** Le Dinh Minh Quan (student 23127460)
**Course:** NLP in Industry — final assignment
**Reference design:** [NLP-KG-WebApp](https://github.com/NLP-Knowledge-Graph/NLP-KG-WebApp)
**Package:** `src/scisearch/` (mirrors the P02–P07 project family; reuses classic RAG-retrieval patterns)

---

## 1. Business Context & Motivation

Scientific output grows far faster than any individual can read. A researcher entering a new sub-field, or an analyst scoping a technology, faces a corpus of millions of papers (this project indexes **~2.0M arXiv abstracts**) and no realistic way to keep up by browsing. Two failure modes dominate the tools they currently use:

1. **Keyword search misses the concept.** Classic lexical search (BM25 / Okapi) is precise when the user already knows the exact terminology, but it is **blind to concept**: it cannot connect a query to a paper that expresses the same idea with different words (paraphrase, synonym, abbreviation, or a newer name for the same method). A query for "dense passage retrieval" should surface the right paper even when the abstract phrases it differently — lexical matching alone does not guarantee this.

2. **A flat ranked list offers no exploration.** Even when retrieval works, returning a single flat list of titles gives the user no sense of the *shape* of the result set — which fields it spans, what sub-topics exist within it, which papers are neighbours of a paper they like, or whether they should broaden or narrow. Exploratory search (literature discovery) is fundamentally different from known-item lookup: the user often does **not** yet know the right query, and the system must help them refine it.

This project addresses both gaps. A query returns not only a **ranked list** but also:

- **Facets** (the arXiv fields the results span, mapped to readable names),
- **Topic clusters** (sub-topics within the results, labelled by their top terms),
- **Related-paper rails** (nearest neighbours of any paper), and
- **Broaden / narrow / related guidance**,

all backed by a **fine-tuned dense retriever** that performs semantic — not just lexical — matching.

---

## 2. Target Users & Jobs-to-be-Done

| User | Context | Job-to-be-done |
|------|---------|----------------|
| **Researchers** | Entering or tracking a sub-field | Find conceptually relevant prior work even without the exact keywords |
| **Analysts** | Technology / landscape scoping | Map a field: which sub-topics exist, how results are distributed across areas |
| **R&D teams** | Applied investigation | Discover methods and adjacent approaches; expand from a seed paper |
| **Literature-review authors** | Systematic / scoping review | Achieve coverage (Recall) over a topic, not just a few top hits |
| **Citation expansion** | Building a reference set | From a known paper, fan out to related work via nearest-neighbour rails |

The recurring jobs-to-be-done are **literature review**, **scoping**, and **citation expansion** — all of which are *exploration* tasks, not single-shot lookups. The system is designed so that one search produces enough structure (facets, clusters, related, suggestions) to support an iterative discovery loop rather than a dead-end list.

---

## 3. The Problem

> **Given a natural-language query over a large scientific corpus, return a semantically ranked set of papers AND the exploration structure needed to navigate and refine the result set.**

Concretely, the system must:

- **Retrieve semantically** — match the *meaning* of the query against paper text (title + abstract), not just shared tokens.
- **Understand the query** — detect intent (keyword vs. conceptual vs. hybrid) and extract metadata filters (field, year) to bias retrieval.
- **Support exploration** — produce facets, topic clusters, related papers, and broaden/narrow suggestions over the result set.
- **Backstop empty results** — when coverage is poor, expand the query and re-retrieve so users rarely hit a zero-result wall.

This decomposes into a retrieval core (the trainable ML/IR model) plus a search-and-exploration stack and a deterministic decision agent that routes, gates, and presents results.

---

## 4. Why NLP / IR

**The search engine itself is an NLP/IR problem.** The hard part is not plumbing — it is the *semantic matching of scientific text*:

- **Dense retrieval is semantic matching.** The core trainable component is a **fine-tuned bi-encoder** that maps both the query and each paper into a shared embedding space, so relevance is measured by vector similarity (cosine over L2-normalized embeddings) rather than token overlap. This directly handles paraphrase and concept-level matching that keyword search cannot. The primary model is **`BAAI/bge-small-en-v1.5`** (MIT, 33.4M params, 384-dim), fine-tuned with `sentence-transformers` using `MultipleNegativesRankingLoss` (in-batch negatives).

- **Query understanding** is an NLP task: classifying intent and extracting field/year filters from free text, which then biases the retrieval fusion (keyword-leaning queries get more BM25 weight; conceptual queries get more dense weight).

- **Topic clustering** is an unsupervised NLP task: grouping the retrieved papers into sub-topics (KMeans over TF-IDF, labelled by top terms) so the user can see structure.

Keyword search (BM25) remains a strong, precise baseline on exact terms — and is fused into the system via Reciprocal Rank Fusion — but it cannot bridge the concept gap. **A trained dense retriever is what fixes that**, and demonstrating that improvement is the technical heart of the project.

---

## 5. Success Metrics

Success is measured on two axes: **business outcomes** for users and **technical IR quality** of the retriever.

### 5.1 Business Metrics

| Metric | Target / Direction | How it is supported |
|--------|--------------------|---------------------|
| **Time-to-first-relevant-result** | Down vs. keyword search | Semantic retrieval surfaces concept matches without query reformulation |
| **Exploration depth** | ≥ 1 facet / cluster / related interaction per session | Facets, topic clusters, related rails make exploration the default |
| **Zero-result rate** | < 2% | Agent D2 coverage gate expands the query and re-retrieves when coverage is poor |
| **Corpus coverage** | All arXiv fields represented | ~2.0M-row corpus with category-based faceting |

### 5.2 Technical Metrics (IR)

| Metric | Role | Notes |
|--------|------|-------|
| **nDCG@10** | **Headline** | The fine-tuned retriever must beat **both** BM25 **and** the zero-shot (un-fine-tuned) base encoder, especially on conceptual queries |
| **Recall@{1,5,10}** | Coverage | Recall (not precision-only) is reported so users understand how exhaustive results are — critical for literature-review coverage |
| **MRR@10** | Rank of first relevant hit | Measures how quickly a relevant paper appears |
| **Latency** | Serving cost | Hot path (dense + BM25 + RRF) on the order of tens of ms over a FAISS slice; cross-encoder rerank is *gated* and adds +150–400ms only when needed; end-to-end sub-second |

Technical metrics are computed with the `sentence-transformers` `InformationRetrievalEvaluator` (or a self-contained equivalent) on a held-out (query → relevant paper) set, with **`BeIR/scifact`** as the real drop-in evaluation set and `mteb/scidocs-reranking` as a secondary signal.

### 5.3 Baselines to Beat

The fine-tuned retriever is explicitly benchmarked against:

1. **BM25** — a self-contained Okapi keyword baseline (precise on exact terms, blind to concept).
2. **Zero-shot base encoder** — the un-fine-tuned `bge-small` / `MiniLM`.

Beating both — particularly on **conceptual** queries where dense retrieval should dominate BM25 — is the definition of technical success for this project.

---

## 6. Scope, Constraints & Robustness Notes

- **Corpus & cutoff.** The primary corpus (`gfissore/arxiv-abstracts-2021`, CC0-1.0) is arXiv-only, English, with a **2021 recency cutoff**. This implies a stale-index risk; the cutoff is documented and an incremental-update / continual-ingestion plan is part of the design.
- **Query privacy.** Search queries can reveal a research direction, so raw-query retention/logging is minimized and an on-prem option is offered.
- **Over-trust mitigation.** Results show scores, diversity, and explicit "broaden" suggestions so the top of the list is never presented as exhaustive or authoritative — it is a ranked *sample*, and Recall@k is reported to make coverage visible.
- **Graceful degradation.** When `torch` or network access is unavailable, the system falls back to a TF-IDF retriever, an identity reranker, and a built-in mini-corpus (~40 real NLP papers) so the demo always runs and all agent decision points still fire.
