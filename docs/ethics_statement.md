# Ethics & Responsible AI Statement

**Project:** Exploratory Scientific Literature Search System (P08)
**Author:** Le Dinh Minh Quan (student 23127460)
**Course:** NLP in Industry — final assignment

This document describes the ethical considerations, fairness limitations, explainability design, and misuse safeguards of the P08 system. The system finds *and* explores scientific papers: a query returns not just a ranked list, but **facets** (research fields), **topic clusters**, **related-paper** rails, and **broaden/narrow** guidance, backed by a fine-tuned dense retriever (a bi-encoder over `BAAI/bge-small-en-v1.5`) fused with BM25 via Reciprocal Rank Fusion and an optionally-gated cross-encoder reranker.

The guiding principle below is simple and load-bearing: **the system presents a ranked, explorable *sample* of the literature — never an exhaustive or authoritative verdict.**

---

## 1. Who benefits

| Stakeholder | Benefit |
|---|---|
| **Researchers** | Faster time-to-first-relevant-result versus keyword search; semantic matching surfaces conceptually-related work that exact-term BM25 search is blind to (e.g., paraphrased or differently-named concepts). |
| **Students / newcomers to a field** | Faceting by arXiv field and topic clustering give a map of an unfamiliar area; "broaden" suggestions and related-paper rails support *exploration*, not just lookup. |
| **R&D teams** | Broader literature discovery across all arXiv fields, with the option of on-prem deployment to keep search direction private. |

The exploration features (facets, clusters, related, query expansion) are the core of the value proposition: they help a user understand *the shape of the results*, not just consume a top-10 list.

---

## 2. Who could be harmed

- **Users who over-rely on the ranking.** Treating the top results as exhaustive or authoritative risks **missed work** and **citation bias** — important papers that rank just below the cutoff, or sit in an under-represented field, may never be seen. This is the system's central ethical risk and is addressed directly in Sections 4 and 6.
- **Authors whose papers are systematically under-ranked.** A learned retriever distributes visibility unevenly. Papers in niche fields, with unusual phrasing, or outside the well-trodden topics of the training pairs can be consistently down-ranked, reducing their reach through no fault of their quality.
- **Society, via dual-use surfacing.** Better semantic search makes *all* indexed methods easier to find, including research with dual-use potential. The system only indexes **public, already-published papers** (see Section 5), but improved discoverability is itself a responsibility.

---

## 3. Bias & fairness

The system inherits and can amplify several biases. We document them rather than hide them.

### 3.1 Corpus coverage bias

The primary corpus is `gfissore/arxiv-abstracts-2021` (CC0-1.0, ~2.0M rows). This bounds what the system can *ever* return:

- **arXiv-only** — fields, venues, and communities that do not post to arXiv are absent.
- **English** — non-English scientific literature is effectively out of scope for the retriever, which is built on English encoders (`bge-small-en-v1.5` / `all-MiniLM-L6-v2`).
- **Recency cutoff** — the primary corpus is a **2021 snapshot**. Papers published after the snapshot are simply missing (the "stale index" risk). This is documented to users, and an incremental / continual-ingestion update plan is part of the design.

### 3.2 Ranking bias

Beyond coverage, the *ranking* can skew toward:

- **Well-cited and arXiv-indexed work**, because such papers dominate the conceptual and lexical patterns the retriever learns from generated `(query, positive_paper)` pairs (title→abstract, first-sentence→paper).
- **Mainstream phrasing**, since the dense bi-encoder and in-batch-negative contrastive training reward queries and papers that look like the bulk of the corpus. **Niche fields may be systematically under-ranked** by the retriever.

### 3.3 Mitigations

- **Hybrid retrieval (dense + BM25 via RRF, k=60)** keeps exact-term recall: a niche paper that uses a rare exact term can still surface through the BM25 ranking even if the dense encoder under-ranks it.
- **Recall@{1,5,10} is reported alongside nDCG@10 and MRR@10**, so coverage — not just top-of-list precision — is measurable and visible. A precision-only headline would hide exactly the kind of under-ranking described above.
- **D2 coverage gate + query expansion** (abbreviation map + pseudo-relevance feedback) re-retrieves when too few results appear or top similarity is low, backstopping the zero-result rate (target < 2%) and helping out-of-domain or garbled queries.
- **Faceting and clustering** make the field/topic distribution of results explicit, so a user can *see* when a field is thin and act on it.

We make no claim that these eliminate bias. They make it observable and partially correctable.

---

## 4. Explainability for non-technical stakeholders

The system is designed so that a non-technical user can understand *why* they are seeing what they see, and *what they are not* seeing. The `/search` response and Gradio UI expose four explanation surfaces:

| Surface | What it tells the user |
|---|---|
| **Relevance scores** | Each result carries a score, signaling that results are *ranked estimates*, not equal-confidence facts. |
| **Facets** | The field distribution (arXiv category → readable field name) shows the breadth — or narrowness — of the result set. |
| **Topic clusters** | KMeans-over-TF-IDF clusters, labeled by top terms, summarize *what kinds* of papers matched. |
| **Decision log** | The agent's four decision points (D1 intent routing, D2 coverage gate, D3 rerank gate, D4 exploration strategy) are recorded as a full audit trace, so the behavior is inspectable rather than opaque. |
| **Broaden / narrow / related suggestions** | Explicit prompts (based on result diversity) tell the user the ranking is *one slice* and offer concrete next moves. |

Together these make the ranked output **transparent, not authoritative**. The intent is that a stakeholder reads the results as "here is a useful sample and here is its shape," not "here is the answer."

---

## 5. Misuse & safeguards

### 5.1 Misuse vectors

- **Mass-scraping** — using the search API to bulk-harvest the corpus or downstream content.
- **Surfacing dual-use methods** — using semantic search to efficiently locate methods with potential for harm.

### 5.2 Safeguards

| Safeguard | Detail |
|---|---|
| **Public-papers-only** | The system indexes only public, already-published arXiv papers. It exposes no private, paywalled, or non-public content; it does not create new capability, only easier discovery of what is already public. |
| **Query privacy** | Search queries can reveal a user's research direction and are treated as sensitive. The design **minimizes retention and logging of raw queries**, and offers an **on-prem deployment** option so queries never leave the user's environment. |
| **Transparency about coverage & recency** | The arXiv-only scope, English limitation, and **2021 recency cutoff** are documented to users, so results are never mistaken for a complete or current view of the field. |
| **No paid LLM by default** | The optional LLM "brain" (used only at D1 for query understanding) is **OFF by default** and validates against / falls back to deterministic rules, so no query text is sent to a third-party API unless explicitly enabled. |

---

## 6. The over-trust-in-ranking risk

This is the risk we weight most heavily. A ranked list invites users to treat the top results as **exhaustive and authoritative**, which directly produces the missed-work and citation-bias harms in Section 2.

Our mitigation is **diversity and coverage transparency** rather than a claim of correctness:

1. **Scores are always shown**, framing every result as a ranked estimate.
2. **Recall@{1,5,10} is a first-class reported metric**, so coverage is measured, not assumed.
3. **Facets and topic clusters expose the diversity (or lack of it)** in the result set.
4. **"Broaden" suggestions are surfaced based on result diversity**, actively nudging the user away from over-narrow trust.
5. **The product copy never hides that the output is a ranked sample.** Related-paper rails and broaden/narrow guidance reinforce that more relevant work exists beyond the visible top-k.

In short: the system is built to *help a user explore the literature and understand the limits of its own ranking* — not to issue a definitive list. Every design choice above exists to keep the user in the loop and aware of what the ranking does not show.

---

## 7. Robustness & graceful degradation (responsibility-relevant)

Responsible behavior also means failing safely:

- **Out-of-domain / garbled queries** are caught by the D2 coverage gate and query expansion rather than returning silent or empty results.
- **Graceful degradation**: when `torch` or network is unavailable, the system falls back to a TF-IDF retriever, an identity reranker, and a built-in ~40-paper mini-corpus, so the demo always runs and never silently fails.
- **Stale-index awareness**: because the primary corpus is a 2021 snapshot, continual ingestion is documented as required for production use; users are told the index has a cutoff.

---

*This statement is grounded entirely in the verified P08 design facts. Dataset and model identifiers (e.g., `gfissore/arxiv-abstracts-2021`, `BAAI/bge-small-en-v1.5`) are exact and are not to be altered.*
