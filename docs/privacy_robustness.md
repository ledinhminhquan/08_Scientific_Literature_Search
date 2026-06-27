# Data Privacy & Model Robustness

P08 — Exploratory Scientific Literature Search System
Author: Le Dinh Minh Quan (23127460) · Course: NLP in Industry, final assignment

This document explains how the system treats **search queries as the sensitive
asset** (the corpus itself is public) and how the agent stays **robust** under
out-of-domain, garbled, very short, or otherwise hard inputs — including how it
degrades gracefully when GPU/network/heavy dependencies are unavailable. The two
load-bearing code references are `src/scisearch/agent/policy.py` (the
deterministic decision FSM) and `src/scisearch/search/expand.py` (query
expansion).

---

## 1. Threat model in one line

| Asset | Sensitivity | Where the risk lives |
|---|---|---|
| The **corpus** (papers) | Public — `gfissore/arxiv-abstracts-2021` (CC0-1.0, ~2.0M arXiv abstracts) | No PII; the only corpus risks are *bias/coverage* and *staleness*, not privacy |
| The **query** | Potentially sensitive | A query reveals what someone is researching — research direction, an unpublished idea, competitive/IP intent |

The asymmetry drives the whole design: we spend our privacy budget on **queries
and query-derived artifacts**, and our robustness budget on **degrading
gracefully** rather than failing or silently returning garbage.

---

## 2. Data Privacy

### 2.1 The corpus carries no PII

The index is built from `gfissore/arxiv-abstracts-2021` — public arXiv metadata
(columns `id`, `authors`, `title`, `abstract`, `categories`, `versions`). Author
names are already published bibliographic data, not private personal data. There
is therefore **no PII-scrubbing requirement on the corpus side**. The corpus
concerns we *do* document live in the bias/ethics analysis, not here:

- **Coverage bias** — arXiv-only, English-dominant.
- **Recency cutoff** — the primary snapshot ends at **2021**; papers after the
  snapshot are missing (stale-index risk), addressed by an incremental-update /
  continual-ingestion plan, not by privacy controls.

### 2.2 The query is the sensitive surface

A query such as *"membrane-less neuromorphic accelerator for X"* can leak a
research direction or unfiled IP. The system minimizes exposure of that surface
with the following controls.

**Minimize retention & logging of raw queries.**
The audit trace the agent emits (decisions D1–D4, intent routing, score margins,
metrics) is what we need for debugging and grading — it does **not** need to
durably store the raw query string. Operationally:

- Keep the raw query only for the lifetime of the request where possible.
- Where a trace must reference the query (e.g. error analysis), prefer a hash or
  a redacted/truncated form over the verbatim string.
- Put a **TTL** on any query log so retention is bounded, not indefinite.

**On-prem / no-egress option.**
The entire hot path — dense retriever (or its TF-IDF stand-in), self-contained
BM25-Okapi, RRF fusion, cross-encoder rerank, faceting and clustering — runs
**locally** with `faiss-cpu`, `scikit-learn`, and `numpy`. The **optional LLM
brain** (`anthropic`, used only at D1 for query understanding) is **OFF by
default**, which means **zero paid API calls and zero query egress** in the
default configuration. An organization with confidential research can run the
full system fully on-prem with no third-party network calls; the query never
leaves the box.

**Anonymization & aggregation for any telemetry.**
If usage telemetry is collected (e.g. for the success metrics — exploration
depth, zero-result rate), it should be aggregated/anonymized: count *that* a
search happened and whether it found something, not *what* was searched. Success
metrics like zero-result rate (target < 2%) and exploration depth are
aggregate counters and do not require storing query text.

**Summary of privacy posture**

| Control | Mechanism |
|---|---|
| Minimize raw-query retention | Request-scoped use; hash/redact in traces; bounded **TTL** on logs |
| No egress | Local FAISS + BM25 + sklearn; LLM brain OFF by default ⇒ 0 paid API, no query leaves host |
| On-prem deployment | Docker / docker-compose; all heavy deps run CPU-local |
| Telemetry | Aggregate/anonymized counters, not query text |
| Corpus | Public papers, CC0 ⇒ no PII to scrub |

---

## 3. Model Robustness

Robustness is enforced primarily by the **deterministic FSM in
`agent/policy.py`**, whose D2 gate is the safety net for bad queries, and by the
**expansion logic in `search/expand.py`**. The principle: never crash, never
silently return junk, and never let the user mistake a ranked sample for ground
truth.

### 3.1 Hard inputs: out-of-domain, garbled, very short

These inputs share a symptom — **weak retrieval signal** (too few hits, or a low
top similarity). The **D2 retrieval-coverage gate** in `policy.py` catches
exactly that:

> **D2 — coverage gate:** if too few results **OR** top *raw* similarity falls
> below threshold ⇒ **expand the query** (abbreviation map + pseudo-relevance
> feedback from the top docs) and **re-retrieve**.

```
query ──▶ D1 route (keyword / conceptual / hybrid, + field/year filters)
      ──▶ retrieve (dense + BM25, fused via RRF, k=60)
      ──▶ D2 coverage gate
             ├─ enough results & top-sim ≥ τ ──▶ proceed
             └─ weak signal ──▶ search/expand.py
                                  • abbreviation expansion (e.g. "DPR" → ...)
                                  • pseudo-relevance feedback (terms from top docs)
                                ──▶ re-retrieve
      ──▶ D3 rerank gate (cross-encoder only if head margin is small/ambiguous)
      ──▶ D4 exploration: facets + topic clusters + related + broaden/narrow
```

- **Very short / underspecified queries** → expansion adds context so the dense
  retriever has more to match on.
- **Garbled / typo'd queries** → BM25's exact-term blindness is offset by the
  dense leg; if both legs come back weak, D2 fires and broadens before giving up.
- **Out-of-domain queries** → low top-similarity trips D2; even after expansion,
  D4's **broaden** suggestions tell the user the corpus is thin here rather than
  pretending the top hit is relevant.

D2 is also the mechanism behind the **zero-result rate < 2%** target: it is the
explicit backstop against returning an empty list.

### 3.2 Known failure cases (and honest handling)

| Failure case | What happens | Mitigation |
|---|---|---|
| **Topic absent from the corpus** | Best matches are weak/irrelevant | D2 expands; if still weak, D4 surfaces **broaden** suggestions and the user sees low scores — we do not fabricate relevance |
| **Ambiguous acronym** (e.g. a TLA with several meanings) | Expansion may pick the wrong sense | Abbreviation map in `expand.py` is best-effort; D4 **topic clustering** splits the result set so competing senses appear as distinct clusters the user can choose between |
| **Non-English query** | Retriever is English-centric (`bge-small-en-v1.5` / `all-MiniLM-L6-v2`, corpus English-dominant) | Recall degrades; documented limitation, not silently masked — scores will read low, signaling poor coverage |
| **Recency** (post-2021 topic) | Snapshot cutoff means the paper isn't indexed | Documented cutoff + incremental-update plan; out of scope for the static index |

### 3.3 Graceful degradation (no GPU, no network, no torch)

The system is built to **always run the demo**, even with none of the heavy
stack present. Each component has a self-contained fallback:

| Capability | Primary | Fallback when torch/network absent |
|---|---|---|
| Dense retriever | Fine-tuned bi-encoder (`BAAI/bge-small-en-v1.5`) | **TF-IDF retriever** (sklearn) fills the dense slot so hybrid search still runs |
| Vector store | FAISS (`faiss-cpu`) | numpy brute-force inner product |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | **identity reranker** (keeps fused order) |
| Corpus | `gfissore/arxiv-abstracts-2021` | **built-in mini-corpus** (~40 real NLP papers), offline |
| LLM brain (D1) | `anthropic` (optional) | rule-based intent routing (default) |

Because the **TF-IDF retriever + identity reranker + built-in corpus** path has
no external dependency, the agent runs end-to-end offline: all four decisions
(D1–D4) fire, intent routing is correct (e.g. *"dense passage retrieval"* ranks
the Dense Passage Retrieval paper #1), and facets + topic clusters are still
produced. Degradation is in *quality*, never in *availability*.

### 3.4 Over-trust in ranking — the UX-level robustness risk

The subtlest failure is human: users treat the top result as **exhaustive and
authoritative**. A ranked list is a *sample*, and we never hide that. Mitigations
baked into the presentation layer (D4) and metrics:

- **Show scores.** Relevance/similarity scores are surfaced, so a weak top hit
  *looks* weak. We never present a low-confidence result as if it were certain.
- **Diversity, not just rank.** D4 **topic clustering** (KMeans over TF-IDF,
  labeled by top terms) and **faceting** by arXiv field expose the breadth of
  the result set instead of collapsing it to a single "best" answer.
- **Broaden / narrow / related suggestions.** D4 actively proposes broadening
  when results are thin or homogeneous, countering the "the top-3 is all there
  is" illusion.
- **Recall@k transparency.** We report **Recall@{1,5,10}** alongside the
  headline **nDCG@10** (and MRR@10) — recall, not precision alone, so users can
  see *coverage*, i.e. how much of the relevant set the ranking actually
  surfaced. Reporting recall is itself an anti-over-trust control.

---

## 4. Where this lives in the code

| Concern | Module | Role |
|---|---|---|
| Robustness FSM, D2 coverage gate | `src/scisearch/agent/policy.py` | Decides when retrieval is too weak and must expand/re-retrieve; gates rerank and exploration |
| Query expansion | `src/scisearch/search/expand.py` | Abbreviation map + pseudo-relevance feedback used by D2 |
| Hybrid retrieval / fusion | `src/scisearch/search/hybrid.py` | Dense + BM25 fused via RRF (k=60) |
| Faceting & exploration | `src/scisearch/search/facets.py` | Field facets + topic clusters + broaden/narrow (D4) |
| Offline fallbacks | `models/retriever.py`, `models/reranker.py`, `data/samples.py` | TF-IDF retriever, identity reranker, built-in mini-corpus |
| Eval / transparency metrics | `training/metrics.py`, `training/evaluate.py` | nDCG@10, Recall@{1,5,10}, MRR@10 |

---

## 5. TL;DR

- **Privacy:** the corpus is public (no PII); the **query** is the sensitive
  asset. Minimize raw-query retention, hash/redact in traces, bound logs with a
  **TTL**, aggregate any telemetry, and run **fully on-prem with no egress** —
  the optional LLM brain is OFF by default, so the default config makes **zero
  external calls**.
- **Robustness:** the **D2 coverage gate** (`policy.py`) + **query expansion**
  (`expand.py`) catch out-of-domain/garbled/short queries and keep the
  zero-result rate < 2%; documented failure modes (absent topics, ambiguous
  acronyms, non-English, post-2021) are handled honestly via low scores,
  clustering, and broaden suggestions; and the **TF-IDF retriever + identity
  reranker + built-in corpus** path guarantees the system always runs offline.
- **Over-trust mitigation:** show scores, surface diversity (facets + clusters),
  offer broaden suggestions, and report **Recall@k** so a ranked sample is never
  mistaken for the whole truth.
