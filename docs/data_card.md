# Data Card — P08 Exploratory Scientific Literature Search

This card documents the data used to train and evaluate the dense retriever at the
core of the **Exploratory Scientific Literature Search System** (P08). It follows
the Hugging Face dataset-card style and covers two artifacts:

1. The **(query, paper) training set** — synthesized contrastive pairs used to
   fine-tune a bi-encoder retriever.
2. The **corpus** — the arXiv abstract collection that is both the source of the
   training pairs and the searchable index at serving time.

- **Author / maintainer:** Le Dinh Minh Quan (student 23127460)
- **Course:** NLP in Industry — final assignment
- **Package:** `src/scisearch/` (data tooling in `src/scisearch/data/`)
- **Reference design:** [NLP-Knowledge-Graph/NLP-KG-WebApp](https://github.com/NLP-Knowledge-Graph/NLP-KG-WebApp)

---

## 1. Dataset summary

The retriever is trained on **synthetically generated `(query, positive_paper)`
pairs** mined from a large corpus of arXiv abstracts. Each pair frames retrieval
as a contrastive learning problem: an **anchor** query should embed close to its
**positive** paper text and far from every other paper in the batch. Negatives are
not stored — they are supplied **in-batch** at training time (one example's
positive is every other example's negative) via
`MultipleNegativesRankingLoss`.

The data is used to fine-tune a sentence-transformers bi-encoder so that semantic
matching of scientific phrasing (paraphrase / concept-level queries) outperforms
keyword search. Evaluation uses a **real, human-annotated IR set** (BeIR/scifact)
rather than the synthetic pairs, so reported quality reflects genuine relevance
judgments.

| Property | Value |
|---|---|
| Language | English (`en`) |
| Task | Dense passage retrieval (bi-encoder, contrastive) |
| Anchor | Synthesized query |
| Positive | Paper text (title + abstract) |
| Negatives | In-batch (not materialized) |
| Training pair source | `gfissore/arxiv-abstracts-2021` |
| Corpus size | 2.0M rows |
| Eval set | `BeIR/scifact` (corpus + queries + qrels) |
| Secondary eval | `mteb/scidocs-reranking` |
| Offline fallback corpus | ~40 real NLP papers (built-in) |

---

## 2. Languages

All text — corpus abstracts, synthesized queries, and the evaluation set — is in
**English (`en`)**. Non-English papers and queries are explicitly out of scope (see
[Out-of-scope](#7-out-of-scope-uses)).

---

## 3. Training set: pair schema and generation

### 3.1 Pair schema

Each training example is an anchor/positive pair consumed by the
`SentenceTransformerTrainer`:

| Field | Role | Description |
|---|---|---|
| `anchor` | Query | A short query string standing in for what a user would type |
| `positive` | Paper text | The relevant paper's text (title and/or abstract) |

There is no explicit `negative` column. With `MultipleNegativesRankingLoss` and the
`"no_duplicates"` batch sampler, the other positives in a batch act as negatives,
so a **large batch size produces more in-batch negatives** and a stronger
contrastive signal.

### 3.2 How queries are synthesized

Queries are **generated from the corpus itself** using two complementary
strategies, each targeting a different query style:

| Strategy | Anchor (query) | Positive (paper) | Captures |
|---|---|---|---|
| Lexical | Paper **title** | Paper **abstract** | Exact-term / keyword-style matching |
| Conceptual | **First sentence of the abstract** | The **paper** | Concept-/paraphrase-level matching |

- The **lexical** pairs (`title -> abstract`) teach the model to bridge a concise,
  keyword-dense query to the fuller abstract that answers it.
- The **conceptual** pairs (`first-sentence -> paper`) teach concept-level matching
  where dense retrieval is expected to beat BM25.
- Generation, deduplication, and pairing live in `src/scisearch/data/pairs.py`;
  the corpus loader is `src/scisearch/data/corpus.py` (download helper
  `download_dataset.py`).

Diversity and dedup of the pair set are deliberate anti-overfitting measures (see
[Biases & limitations](#9-known-biases-and-limitations)).

### 3.3 Optional query prompt

When the primary base model `BAAI/bge-small-en-v1.5` is used, an optional query
prompt may be prepended **to queries only**:

```
Represent this sentence for searching relevant passages:
```

This matches bge's intended asymmetric query/passage usage. It is optional and not
applied to the positive paper text.

---

## 4. Corpus: source and fields

The corpus is **`gfissore/arxiv-abstracts-2021`** (2.0M rows). It is the source of
the training pairs and the searchable index at serving time.

| Column | Type | Use in P08 |
|---|---|---|
| `id` | string | Stable paper identifier; used by `GET /related/{paper_id}` |
| `title` | string | Lexical anchor; part of indexed text |
| `abstract` | string | Positive text; primary searchable content |
| `authors` | string | Author facet |
| `categories` | list | arXiv category facet → readable field names |
| `versions` | — | Enables year facet (version date) |

The `categories`, `authors`, and `versions` columns directly enable the system's
**exploration** features — faceting by field/author/year — that make the search
"exploratory" rather than a flat ranked list.

### Built-in mini-corpus (offline fallback)

A built-in corpus of **~40 real NLP papers** (`src/scisearch/data/samples.py`)
ships with the package so the demo and the agent always run with no network and no
torch. Offline, a **TF-IDF retriever** (scikit-learn) stands in for the dense slot
and an identity reranker stands in for the cross-encoder, so hybrid search still
functions end-to-end.

---

## 5. Splits and sizes

| Split | Source | Size | Purpose |
|---|---|---|---|
| Train pairs | Generated from `gfissore/arxiv-abstracts-2021` | Derived from up to 2.0M corpus rows (configurable slice) | Fine-tune the bi-encoder |
| Serving corpus | `gfissore/arxiv-abstracts-2021` (or a slice) | up to 2.0M rows; latency profiled on a 100k–500k slice | Index searched at inference |
| IR eval | `BeIR/scifact` | corpus + queries + qrels | Headline metrics |
| Secondary eval | `mteb/scidocs-reranking` | reranking set | Reranker sanity check |
| Offline fallback | Built-in mini-corpus | ~40 papers | Network-free demo |

The number of generated training pairs is configurable (driven by how much of the
2.0M-row corpus is sampled). Training runs 1–3 epochs (1–2 recommended for
anti-overfitting). Indexing the full 2.0M corpus is supported; the **hot path
latency is profiled on a 100k–500k slice**.

---

## 6. IR evaluation set

Reported retrieval quality uses **`BeIR/scifact`**, a standard scientific
fact-verification IR benchmark, as a **drop-in for `InformationRetrievalEvaluator`**.
It provides the three components a retrieval evaluator needs:

| Component | Role |
|---|---|
| `corpus` | Documents to retrieve over |
| `queries` | Held-out evaluation queries |
| `qrels` | Human relevance judgments (query → relevant document) |

`mteb/scidocs-reranking` is used as a secondary set to sanity-check the
cross-encoder reranker.

### Metrics

Computed on the held-out (query → relevant paper) judgments via the
`InformationRetrievalEvaluator` / a self-contained evaluator
(`src/scisearch/training/metrics.py`, `evaluate.py`):

- **nDCG@10** — headline metric
- **Recall@{1, 5, 10}** — coverage (reported so users understand recall, not just precision)
- **MRR@10**

The fine-tuned retriever is required to beat two baselines: **BM25** (self-contained
Okapi) and the **zero-shot base encoder** (un-fine-tuned bge/MiniLM), especially on
**conceptual queries** where dense retrieval is expected to dominate BM25.

---

## 7. Intended uses

**Intended use.** Training a dense bi-encoder for semantic retrieval over English
arXiv abstracts, and serving exploratory scientific literature search: ranked
results plus facets (field/year/author), topic clusters, related-paper rails, and
broaden/narrow guidance. Appropriate for research-discovery and exploratory
literature-survey scenarios over a public arXiv index.

The intended consumers are the P08 training pipeline
(`src/scisearch/training/train_retriever.py`) and the search/agent stack
(`src/scisearch/search/`, `src/scisearch/agent/`).

---

## 7. Out-of-scope uses

- **Non-arXiv corpora.** The data and category/field mappings are arXiv-specific;
  retrieval quality and facet labels do not transfer to non-arXiv sources without
  re-derivation.
- **Full-text retrieval.** Only **titles and abstracts** are indexed — not paper
  full text. Queries answerable only from a paper's body (methods, results tables,
  appendices) are out of scope.
- **Non-English** queries and documents — all text is English; multilingual use is
  unsupported.
- Treating the index as authoritative or exhaustive (see ranking over-trust under
  [Biases & limitations](#9-known-biases-and-limitations)).

---

## 8. Licensing

| Artifact | License | Notes |
|---|---|---|
| Corpus `gfissore/arxiv-abstracts-2021` | **CC0-1.0** | Public domain dedication |
| Eval `BeIR/scifact` | **CC-BY-SA** | Attribution + share-alike |
| Generator / system code (`src/scisearch/`) | **MIT** | Pair-generation and tooling code |

Base models referenced by the pipeline carry their own licenses (e.g.
`BAAI/bge-small-en-v1.5` MIT, `all-MiniLM-L6-v2` Apache-2.0,
`cross-encoder/ms-marco-MiniLM-L-6-v2` Apache-2.0); see the model card. Downstream
use of the evaluation set must honor CC-BY-SA's attribution and share-alike terms.

---

## 9. Known biases and limitations

- **Corpus coverage / domain bias.** The corpus is **arXiv-only and English-only**.
  Fields, venues, and communities under-represented on arXiv are under-represented
  in retrieval.
- **Recency cutoff (2021).** The primary corpus is a **2021 snapshot**. Papers
  published after the snapshot are missing → **stale-index risk**. Continual
  ingestion / incremental updates are required to keep the index current; the
  cutoff and update plan are documented as a known risk.
- **Synthetic-query bias.** Training anchors are derived from titles and first
  sentences, not real user queries. They may under-represent genuine search
  phrasing, typos, and multi-intent questions. The held-out **BeIR/scifact**
  evaluation (real queries + human qrels) mitigates over-optimistic
  self-evaluation.
- **Title/abstract only.** No full-text signal; concept matches that depend on a
  paper's body are not learnable from this data.
- **Over-trust in ranking.** Users may treat top results as exhaustive or
  authoritative. The system reports scores and diversity and surfaces "broaden"
  suggestions to counteract this; the data card stresses results are a **ranked
  sample**, not a complete answer.
- **Query privacy.** Search queries can reveal a research direction; raw-query
  retention/logging should be minimized (on-prem option available). This is a
  property of how the data is *used*, documented here for completeness.

Anti-overfitting is addressed at the data level via a **large, diverse, deduplicated
pair set** and at the training level via **1–2 epochs** and **in-batch negatives**.

---

## 10. Example pairs

Synthesized training pairs (anchor = query, positive = paper text):

```json
{
  "anchor": "Dense Passage Retrieval for Open-Domain Question Answering",
  "positive": "Open-domain question answering relies on efficient passage retrieval to select candidate contexts... we show that retrieval can be practically implemented using dense representations alone..."
}
```

Conceptual pair (first sentence → paper):

```json
{
  "anchor": "We introduce a method that maps queries and passages into a shared embedding space for retrieval.",
  "positive": "Title: ...  Abstract: ..."
}
```

Evaluation triple (from `BeIR/scifact` qrels):

```text
query_id   corpus_id   relevance
q_104      doc_8123    1
```

---

## 11. Maintenance

- **Owner:** Le Dinh Minh Quan (student 23127460).
- **Code home:** `src/scisearch/data/` — `corpus.py` (load), `pairs.py` (generate),
  `download_dataset.py` (fetch), `samples.py` (offline mini-corpus).
- **Versioning.** Models and their data provenance are tracked via a model registry
  (`model_meta.json` + a `latest` pointer, `repo@revision`) so a trained checkpoint
  can be traced to the corpus slice and pair-generation settings used.
- **Update plan.** Because of the **2021 cutoff**, the corpus is refreshed via
  incremental ingestion of newer arXiv records; FAISS index sharding and cached
  embeddings support re-indexing without retraining from scratch.
- **Regeneration.** Training pairs are reproducible from the corpus via the CLI
  (`scisearch data`, `scisearch pairs`); evaluation is reproducible via
  `scisearch evaluate` against `BeIR/scifact`.

---

*This card describes data only. For model architecture, training hyperparameters,
and serving details, see the accompanying model card and README.*
