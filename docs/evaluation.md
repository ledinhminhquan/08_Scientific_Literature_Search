# Retrieval Quality Evaluation

This document is the quality deep-dive for **P08 — Exploratory Scientific Literature Search**. The system finds *and* explores scientific papers: a query returns a ranked list plus facets, topic clusters, related-paper rails and broaden/narrow guidance, backed by a fine-tuned dense bi-encoder. The single most important engineering claim we make is that **fine-tuning the retriever improves ranking quality over both a keyword baseline and the un-fine-tuned (zero-shot) encoder.** Everything below exists to measure that claim rigorously.

The headline metric is **nDCG@10**. We also report **Recall@{1, 5, 10}** and **MRR@10**, with a per-query error analysis to explain *where* each method wins or fails.

Code referenced throughout:

- `src/scisearch/training/metrics.py` — metric implementations (nDCG, Recall@k, MRR).
- `src/scisearch/training/evaluate.py` — evaluation driver (held-out + real BeIR/scifact).
- `src/scisearch/analysis/error_analysis.py` — per-query win/loss breakdown.

---

## 1. What we evaluate

We evaluate the **retriever** — the component that, given a query, returns a ranked list of candidate papers. Three rankers are compared on identical queries and corpora:

| Ranker | What it is | Why it is here |
| --- | --- | --- |
| **BM25** | Self-contained Okapi BM25 over tokenized `title + abstract`. | Strong lexical baseline; precise on exact terms, blind to concept/paraphrase. |
| **Zero-shot base** | Un-fine-tuned `BAAI/bge-small-en-v1.5` (384-dim; fallback `sentence-transformers/all-MiniLM-L6-v2`). | Isolates the value of *fine-tuning* — same architecture, no domain adaptation. |
| **Fine-tuned** | The same bi-encoder after fine-tuning with `sentence-transformers` `MultipleNegativesRankingLoss` (in-batch negatives). | The system we ship; must beat both baselines, especially on conceptual queries. |

The three-way split is deliberate. BM25 vs. dense isolates the **lexical-vs-semantic** axis; zero-shot vs. fine-tuned isolates the **fine-tuning** axis. Only the second comparison proves that the training step (not just "using embeddings") earns its keep.

> Note on the hybrid system. In production the served ranking is **dense + BM25 fused with Reciprocal Rank Fusion (RRF)**, optionally reranked by a cross-encoder. For the quality study we evaluate the retrievers in isolation so the *source* of any gain is attributable. RRF is defined in §3 and is what the deployed engine uses.

---

## 2. Metrics, defined precisely

Let a single query `q` have a set of relevant papers `R_q` (the gold/qrels set). The ranker returns an ordered list `d_1, d_2, ...`. Let `rel(i) ∈ {0, 1}` (or a graded relevance from qrels) be the relevance of the document at rank `i`. Metrics are computed per query and then **averaged over all queries**. Implementations live in `src/scisearch/training/metrics.py`.

### 2.1 nDCG@10 — the headline

Discounted Cumulative Gain at cutoff `k` rewards placing relevant documents high, with a logarithmic position discount:

```
DCG@k = sum_{i=1..k}  rel(i) / log2(i + 1)
```

The Ideal DCG (`IDCG@k`) is the DCG of the best possible ordering of the relevant documents. Normalized DCG is the ratio:

```
nDCG@k = DCG@k / IDCG@k          (0 ≤ nDCG@k ≤ 1)
```

We report **nDCG@10**. It is the headline because it is *graded* and *rank-aware*: it credits getting the right paper near the top, not just somewhere in the list, which matches how a literature-search user actually behaves.

### 2.2 Recall@{1, 5, 10}

Fraction of a query's relevant documents found within the top `k`:

```
Recall@k = |{retrieved top-k} ∩ R_q| / |R_q|
```

We report `k ∈ {1, 5, 10}`. Recall answers the *coverage* question — "did we surface the relevant papers at all?" — which is essential for an exploratory tool where the user is browsing, not just chasing one answer. Recall@1 doubles as a strict "got it first try" signal.

### 2.3 MRR@10

For each query take the reciprocal of the rank of the **first** relevant document (0 if none appears within the cutoff), then average:

```
MRR@k = (1 / |Q|) * sum_{q in Q}  1 / rank_of_first_relevant(q)
```

MRR@10 captures "how quickly does the user hit *a* relevant paper" — directly tied to the business metric *time-to-first-relevant-result*.

### 2.4 How metrics relate

- nDCG@10 is the all-round summary (graded, rank-aware).
- MRR@10 zooms in on the *first* hit.
- Recall@k ignores order within the cutoff and measures coverage.

A method can win on Recall@10 (finds everything eventually) yet lose on MRR@10 (buries the best result) — surfacing exactly this kind of disagreement is the job of the error analysis in §5.

---

## 3. Reciprocal Rank Fusion (RRF)

The deployed engine fuses the dense ranking and the BM25 ranking with RRF. For a document `d`, summing over the set of rankings `r` in which it appears:

```
RRF_score(d) = sum_{r}  1 / (k + rank_r(d))          with k = 60
```

`rank_r(d)` is `d`'s 1-based position in ranking `r`. The constant `k = 60` damps the influence of very top ranks so no single list dominates, and any document absent from a list simply contributes nothing from that list. RRF needs only ranks (not comparable scores), which is why it cleanly fuses a cosine-similarity dense list with a BM25 list. The agent biases fusion toward the intent-preferred ranking by **duplicating that ranking** in the sum (e.g. a keyword query weights BM25 more; a conceptual query weights dense more).

---

## 4. Evaluation protocol

`src/scisearch/training/evaluate.py` runs two complementary protocols.

### 4.1 Held-out (query → relevant paper) set

Training pairs are *generated* from the corpus `gfissore/arxiv-abstracts-2021` (CC0-1.0, 2.0M rows): `(title → abstract)` lexical pairs and `(first-sentence-of-abstract → paper)` conceptual pairs. We hold out a disjoint slice of these pairs as an in-domain evaluation set: the query is the title/first-sentence and the single relevant paper is its source document. This set is constructed once, de-duplicated against the training split (no query or paper leaks across the boundary), and scored with a self-contained evaluator.

### 4.2 Real benchmark — BeIR/scifact

To avoid grading the model only on our own pair-generation recipe, we also evaluate on **`BeIR/scifact`** (corpus + queries + qrels), a real scientific IR benchmark that drops straight into `sentence-transformers`' `InformationRetrievalEvaluator`. `mteb/scidocs-reranking` is a secondary check. This is the credible, external number: same queries, same qrels, three rankers, identical cutoffs.

### 4.3 Procedure (identical across rankers)

1. Build the corpus index (FAISS over L2-normalized embeddings for dense; Okapi index for BM25).
2. For each query, retrieve the top 10.
3. Compute nDCG@10, Recall@{1,5,10}, MRR@10 against the qrels via `metrics.py`.
4. Average over all queries; report per ranker.

The corpus, query set, and qrels are frozen across the three rankers so any difference is attributable to the ranker alone.

> Offline / no-torch mode. When torch or network is unavailable, a **TF-IDF retriever** (sklearn) stands in for the dense slot and a built-in **~40-paper mini-corpus** of real NLP papers backs the demo. The evaluation harness still runs end-to-end on this slice; numbers are not comparable to the full-corpus run but prove the pipeline is correct.

---

## 5. Results

### 5.1 Results table template (fill after training)

Report each metric for all three rankers on **both** the held-out set and BeIR/scifact. The decisive comparison is **fine-tuned vs. zero-shot** (does training help?) and **fine-tuned vs. BM25** (does dense beat lexical?).

**BeIR/scifact (headline table):**

| Metric | BM25 | Zero-shot base | Fine-tuned |
| --- | --- | --- | --- |
| **nDCG@10** | _TBD_ | _TBD_ | _TBD_ |
| Recall@1 | _TBD_ | _TBD_ | _TBD_ |
| Recall@5 | _TBD_ | _TBD_ | _TBD_ |
| Recall@10 | _TBD_ | _TBD_ | _TBD_ |
| MRR@10 | _TBD_ | _TBD_ | _TBD_ |

**Held-out (query → relevant paper) set:**

| Metric | BM25 | Zero-shot base | Fine-tuned |
| --- | --- | --- | --- |
| **nDCG@10** | _TBD_ | _TBD_ | _TBD_ |
| Recall@1 | _TBD_ | _TBD_ | _TBD_ |
| Recall@5 | _TBD_ | _TBD_ | _TBD_ |
| Recall@10 | _TBD_ | _TBD_ | _TBD_ |
| MRR@10 | _TBD_ | _TBD_ | _TBD_ |

**How to read it.** Success = the **Fine-tuned** column ≥ both others on nDCG@10, with the largest margin over BM25 on *conceptual* queries (paraphrase/concept matches that have little lexical overlap), and a clear lift over **Zero-shot** that demonstrates fine-tuning — not merely "using embeddings" — produced the gain.

### 5.2 Metric saturation on a tiny corpus

On the built-in ~40-paper mini-corpus, **metrics saturate**: with so few candidates per query, all three rankers frequently place the single relevant paper at or near rank 1, so nDCG@10 and MRR@10 crowd toward 1.0 and the rankers look indistinguishable. This is expected, not a bug — there is simply not enough room for one method to out-rank another. **Differentiation appears at scale**, where many near-miss distractors (lexically similar but conceptually wrong, and vice-versa) force the rankers apart. Treat mini-corpus numbers as a *smoke test* of the pipeline; treat the full-corpus and BeIR/scifact numbers as the real verdict.

---

## 6. Per-query error analysis

`src/scisearch/analysis/error_analysis.py` goes beyond averages and classifies **each query** by which method retrieved the relevant paper higher. Averaging hides disagreement; the per-query view is where we learn the system's actual behavior. Each query falls into one of four buckets:

| Bucket | Definition | What it teaches us |
| --- | --- | --- |
| **Dense wins** | Fine-tuned dense ranks the relevant paper materially higher than BM25. | The intended payoff — typically conceptual/paraphrase queries with low lexical overlap. Confirms why we fine-tune. |
| **BM25 wins** | BM25 ranks it higher than dense. | Usually exact-term, rare-token, or acronym queries. Justifies keeping BM25 in the RRF hybrid. |
| **Both succeed** | Both place it in the top-k (often top-1). | Easy / saturated queries; contribute little to differentiation (see §5.2). |
| **Both fail** | Neither retrieves it within the cutoff. | The hard tail — out-of-domain, garbled, or genuinely ambiguous queries. Targets for the agent's D2 coverage gate (query expansion + re-retrieval). |

The report tabulates the count and example queries in each bucket and the average rank gap between dense and BM25. This directly motivates three design choices: (1) **fine-tuning** is justified by the size of the *Dense wins* bucket; (2) **keeping BM25 in RRF fusion** is justified by the *BM25 wins* bucket — dropping it would regress those queries; (3) the **agent's coverage gate (D2)** and **query expansion** target the *Both fail* bucket, backstopping the zero-result rate (target < 2%).

---

## 7. Caveats and reporting hygiene

- **Coverage, not just precision.** We always report Recall@k alongside nDCG/MRR so users and reviewers can see how exhaustively relevant papers are surfaced — important for an exploratory tool that should never imply the top result is the only one.
- **Generated-pair bias.** The held-out set shares the generation recipe of the training pairs; BeIR/scifact is the independent check. Trust the external benchmark when the two disagree.
- **Recency cutoff.** The primary corpus is a 2021 arXiv snapshot, so any "missing" relevant paper published later is a stale-index artifact, not a retriever failure — accounted for when reading *Both fail* cases.
- **Frozen comparison.** Corpus, queries, qrels and cutoffs are held constant across BM25, zero-shot, and fine-tuned so every reported delta is attributable to the ranker.
