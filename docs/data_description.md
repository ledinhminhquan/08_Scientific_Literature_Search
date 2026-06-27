# Data Description — P08 Exploratory Scientific Literature Search

This document describes the data that powers the **Exploratory Scientific Literature Search System**: the source corpus, how supervised `(query, paper)` training pairs are generated from it, the real evaluation benchmarks, the offline fallback corpus, the preprocessing pipeline, the train/eval split design, and the known limitations and biases. Every fact here is grounded in the project's verified data and model inventory.

---

## 1. Overview

The ML/IR core of this project is a **fine-tuned bi-encoder dense retriever** that maps a query and a paper into a shared 384-dimensional embedding space. Training it requires `(query, positive_paper)` pairs; evaluating it requires a benchmark with `(query → relevant paper)` judgments. Because the source corpus ships with **no human relevance labels**, the system **generates** training pairs self-supervised from the corpus structure, and **evaluates** on an external, human-labeled IR benchmark. A small built-in corpus guarantees the demo runs with no network or GPU.

| Role | Dataset / artifact | Provenance | Notes |
|------|--------------------|-----------|-------|
| Training corpus & pair source | `gfissore/arxiv-abstracts-2021` | CC0-1.0, ~2.0M rows | Generates self-supervised `(query, paper)` pairs |
| Real IR evaluation (primary) | `BeIR/scifact` | corpus + queries + qrels | Drop-in for `InformationRetrievalEvaluator` |
| Real IR evaluation (secondary) | `mteb/scidocs-reranking` | reranking benchmark | Secondary signal |
| Offline fallback corpus | built-in mini-corpus | ~40 real NLP papers | Demo always runs, no network/torch |

---

## 2. Source Corpus — `gfissore/arxiv-abstracts-2021`

The primary corpus is **`gfissore/arxiv-abstracts-2021`** (license **CC0-1.0**, public domain; **~2.0M rows**). Each row is a single arXiv paper. The columns the system relies on:

| Column | Type | Used for |
|--------|------|----------|
| `id` | string | Stable paper identifier / FAISS row key |
| `title` | string | Retrieval text + lexical query source |
| `abstract` | string | Retrieval text + conceptual query source |
| `categories` | **list** | arXiv field facets (e.g. `cs.CL`, `cs.LG`) |
| `authors` | string | Author facets |
| `versions` | list | Year / recency facets |

The combination of `categories`, `authors`, and `versions` is what enables the system's **field / year / author faceting** — the exploratory layer that turns a flat ranked list into a navigable result set. The `categories` column is a **list** (a paper can belong to multiple fields), which is handled explicitly in preprocessing (see §5).

CC0 licensing matters here: it places no redistribution or attribution constraints on the derived embeddings, indices, or generated pairs, so the trained artifacts can be shipped freely.

---

## 3. Generating `(query, paper)` Training Pairs

The corpus has titles, abstracts, and categories — but **no `(query, relevant-paper)` labels**. The retriever is therefore trained on **self-supervised pairs generated from the corpus itself**, in two complementary styles, with negatives supplied automatically by the loss function.

### 3.1 Lexical pairs — `title → abstract`

For a paper, the **title** acts as a short, keyword-dense "query" and its **abstract** acts as the positive passage:

```
query    = paper.title
positive = paper.abstract        # (or title + abstract as the indexed text)
```

This teaches the encoder that a concise topical phrase should retrieve the matching paper body — the lexical / exact-term matching regime where keyword search already does well, now folded into the dense space.

### 3.2 Conceptual pairs — `concept-query → paper`

To push the encoder beyond keyword overlap, a second pair type uses the **first sentence of the abstract** as a natural, concept-level query for the paper:

```
query    = first_sentence(paper.abstract)
positive = paper.title + paper.abstract
```

This is the **conceptual** regime — paraphrase- and concept-level matching — where a trained dense retriever is expected to beat BM25, since the query and the indexed text rarely share exact tokens.

### 3.3 Negatives — in-batch via `MultipleNegativesRankingLoss`

No explicit negatives are mined. Training uses **`MultipleNegativesRankingLoss`** (from `sentence-transformers`): within a training batch, for each anchor query, the **positives of all the other examples in the batch act as negatives** ("in-batch negatives"). Consequences for data design:

- **Large batches are valuable.** A larger `per_device_train_batch_size` (128–256) yields more in-batch negatives per anchor, which strengthens the contrastive signal. GPU profiles: H100 bs256 · A100-40 bs128 · L4 bs64 · T4 bs32 (fp16).
- **`batch_sampler="no_duplicates"`** is used so the same paper does not appear twice in a batch (which would make one of its own positives a false negative).

This pairing scheme means **one corpus → two pair styles → automatic negatives**, producing a large, diverse contrastive training set from unlabeled data at near-zero curation cost.

---

## 4. Evaluation Data

Because the generated pairs are a *proxy* for relevance, the headline numbers come from **human-labeled, external benchmarks** — not from held-out generated pairs alone.

- **`BeIR/scifact` (primary, VERIFIED).** Ships `corpus`, `queries`, and `qrels` (relevance judgments), so it plugs **directly** into the `sentence-transformers` `InformationRetrievalEvaluator`. It is a scientific-claim verification IR task — the right domain (scientific text) and the right shape (real queries with graded relevance) for measuring this retriever.
- **`mteb/scidocs-reranking` (secondary).** A reranking benchmark used as a secondary scientific-domain signal.

### Metrics

All IR metrics are computed via the `InformationRetrievalEvaluator` (or a self-contained equivalent) over the held-out `(query → relevant paper)` set:

- **nDCG@10 — headline metric**
- **Recall@{1, 5, 10}**
- **MRR@10**

Recall is reported deliberately (not precision alone) so users can judge **coverage**, not just top-1 correctness — important for an *exploratory* tool.

### Baselines to beat

The fine-tuned retriever is measured against two baselines:

1. **BM25** (self-contained Okapi) — strong on exact terms.
2. **Zero-shot base encoder** — the **un-fine-tuned** `bge-small` / `MiniLM`.

The fine-tuned model must beat **both**, and the conceptual-query slice is where the gain should be largest (dense ≫ BM25 on concept matching).

---

## 5. Preprocessing

The same deterministic pipeline is applied to corpus papers and to generated queries.

1. **Indexed text = `title` + `abstract`.** Each paper's retrievable representation concatenates title and abstract; this is what gets embedded (dense) and tokenized (BM25 over title+abstract).
2. **Category splitting.** The `categories` **list** is split into individual arXiv codes and mapped to **readable field names** for faceting (e.g. `cs.CL → Computation and Language`). A paper with multiple categories contributes to multiple facets.
3. **Deduplication.** Papers are de-duplicated before pair generation and indexing. Dedup is also an **anti-overfitting** measure — it prevents the same content from dominating the contrastive batches.
4. **Leakage-free query / corpus split.** Queries are derived from papers, so naive splitting would let a paper's own title/abstract sentence appear on **both** sides. The split is constructed so that the **paper whose text generated a query is held consistently** — the held-out evaluation `(query → relevant paper)` pairs do not leak their answer into the training corpus. This keeps the reported nDCG/Recall/MRR honest.

---

## 6. Built-in Mini-Corpus (Offline Fallback)

A **built-in mini-corpus of ~40 real NLP papers** ships inside the package. It is the **offline fallback** that lets the system run with **no network, no `datasets` download, and no `torch`**:

- When torch/network are unavailable, a **TF-IDF retriever (sklearn)** stands in for the dense slot and an **identity reranker** stands in for the cross-encoder, so the full hybrid search + agent pipeline still executes.
- It is validated end-to-end: all four agent decision points fire, intent routing is correct (e.g. the query *"dense passage retrieval"* ranks the Dense Passage Retrieval paper **#1**), and facets + topic clusters are produced.

This guarantees the demo, the FastAPI/Gradio app, and the report/slides generation **always work**, independent of external data availability.

---

## 7. Train / Eval Justification

- **Why generated pairs for training.** The corpus has no relevance labels, and hand-labeling 2M papers is infeasible. Self-supervised `title→abstract` (lexical) and `first-sentence→paper` (conceptual) pairs give a **large, diverse, free** training signal that directly matches the two retrieval regimes the product must serve. In-batch negatives turn every batch into a contrastive learning problem without explicit negative mining.
- **Why external benchmarks for evaluation.** Generated pairs are a *proxy* for relevance; scoring only on them would reward learning the generation heuristic, not real relevance. **`BeIR/scifact`** provides **human qrels** in the scientific domain, giving a trustworthy, comparable nDCG@10 headline.
- **Why these baselines.** Comparing against **BM25** and the **zero-shot encoder** isolates the value added by fine-tuning specifically; beating both (especially on conceptual queries) is the project's core technical claim.
- **Anti-overfitting by data design.** Large diverse pair set + dedup + only **1–2 epochs** + in-batch negatives keep the model from memorizing the generation pattern.

---

## 8. Known Limitations & Biases

| Limitation / bias | Why it matters | Mitigation |
|-------------------|----------------|-----------|
| **arXiv-only coverage** | Only arXiv preprints are indexed; journal-only / non-arXiv venues are absent | Document scope; faceting exposes which fields are actually covered |
| **English-only** | The corpus text is English; non-English queries/papers are out of scope | Stated explicitly as a coverage limit |
| **2021 recency cutoff** | `arxiv-abstracts-2021` is a snapshot — papers after the cutoff are **missing** (stale-index risk) | Document the cutoff; plan continual / incremental ingestion |
| **Self-supervised pairs ≈ relevance** | `title→abstract` and `first-sentence→paper` are **approximations** of true query–document relevance, not human judgments | Headline metrics come from human-labeled `BeIR/scifact`, not from generated pairs |
| **Stale index** | New research is not retrievable until re-ingested | Continual ingestion plan; never present results as exhaustive |
| **Over-trust in ranking** | Users may treat the top result as authoritative/complete | Show scores + diversity + "broaden" suggestions; report **Recall@k** so coverage is visible |
| **Query privacy** | Search queries can reveal a sensitive research direction | Minimize retention/logging of raw queries; on-prem option |
| **Dual-use** | Could surface dual-use research | System only indexes **public** papers |

The single most important caveat: **the training pairs are a generated approximation of relevance**. This is why the project never reports training-pair scores as the headline — the public, human-labeled **`BeIR/scifact`** benchmark is the source of truth for whether the fine-tuned retriever actually retrieves *relevant* science.
