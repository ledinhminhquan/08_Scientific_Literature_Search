# Model Card: `scisearch-retriever-bge-small-v1`

A fine-tuned dense bi-encoder for **dense passage retrieval of scientific papers** (title + abstract), built for the P08 *Exploratory Scientific Literature Search System*. The model maps a free-text query and a paper into a shared embedding space so that semantically relevant papers can be found by cosine similarity — covering conceptual and paraphrase matches that keyword search (BM25) misses.

- **Author:** Le Dinh Minh Quan (student 23127460)
- **Course:** NLP in Industry — final assignment
- **Package:** `src/scisearch/`
- **Reference:** [NLP-KG-WebApp](https://github.com/NLP-Knowledge-Graph/NLP-KG-WebApp)

---

## Model Details

| Property | Value |
|---|---|
| Model type | Dense bi-encoder (Sentence Transformer) |
| Base model | `BAAI/bge-small-en-v1.5` (MIT, 33.4M params) |
| Embedding dimension | 384 |
| Output | L2-normalized embeddings (inner product = cosine similarity) |
| Task | Dense passage retrieval over scientific paper title + abstract |
| Loss (training) | `MultipleNegativesRankingLoss` (in-batch negatives) |
| Framework | `sentence-transformers` (`SentenceTransformerTrainer`) |
| Language | English |

The model is a **bi-encoder**: query and paper are encoded *independently* into the same 384-dim space, which makes the corpus side pre-computable and the query side a single forward pass — enabling fast approximate nearest-neighbor search via FAISS at query time.

### Base / fallback / domain options

| Role | Checkpoint | License | Params | Dim |
|---|---|---|---|---|
| **Primary base** | `BAAI/bge-small-en-v1.5` | MIT | 33.4M | 384 |
| Fallback base | `sentence-transformers/all-MiniLM-L6-v2` | Apache-2.0 | 22.7M | 384 |
| Domain option | `malteos/scincl` | MIT | — | 768 |
| Domain option | `allenai/specter2_base` | Apache-2.0 | — | 768 |

### Query prompt (optional)

BGE models support an asymmetric query prefix. When enabled, queries (only — not the paper/passage side) are prefixed with:

```
Represent this sentence for searching relevant passages:
```

---

## Intended Use

- **Primary use:** Dense retrieval of scientific papers by encoding `title + abstract` and ranking corpus papers against a user query by cosine similarity. The model is the ML/IR core of an exploratory search system that also returns facets, topic clusters, related-paper rails, and broaden/narrow guidance.
- **Search stack integration:** The dense ranking is fused with a self-contained BM25-Okapi ranking via **Reciprocal Rank Fusion (RRF)** (`score(d) = Σ_r 1/(k + rank_r(d))`, `k = 60`), and the head of the fused list can be re-scored by a cross-encoder reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`).
- **Related-paper retrieval:** The same embeddings power dense nearest-neighbor lookups for "related papers" given an existing paper id.
- **Intended users:** Researchers, students, and engineers performing exploratory scientific literature search over an arXiv-derived corpus.

### Out-of-Scope Use

This model is **not** intended for:

- **Full-text retrieval** — it is trained and evaluated on `title + abstract` only, not paper bodies.
- **Non-arXiv corpora** — it was tuned on arXiv abstracts; behavior on other document collections (patents, clinical notes, news) is untested.
- **Non-English text** — training data and base model are English; no multilingual capability is claimed.
- **A relevance oracle / authoritative judgment** — outputs are a *ranked sample*, not an exhaustive or ground-truth statement of relevance. Do not use scores as a definitive measure of a paper's importance or correctness.

---

## Training Data

| Component | Source | License | Notes |
|---|---|---|---|
| Corpus | `gfissore/arxiv-abstracts-2021` | CC0-1.0 | 2.0M rows; columns `id, authors, title, abstract, categories (LIST), versions` |
| Training pairs (primary) | **Synthetically generated** from the corpus | — | `(query, positive_paper)` pairs |
| Eval (real, drop-in) | `BeIR/scifact` | — | corpus + queries + qrels for `InformationRetrievalEvaluator` |
| Eval (secondary) | `mteb/scidocs-reranking` | — | reranking evaluation |
| Offline fallback | Built-in mini-corpus (~40 real NLP papers) | — | so the demo always runs without network/data |

### Synthetic pair generation

The primary training signal is **self-supervised** `(query, positive_paper)` pairs mined from the arXiv corpus:

- **Lexical pairs:** `title → abstract` (the title acts as a short query for its own abstract).
- **Conceptual pairs:** `first-sentence-of-abstract → paper` (a concept-level query mapped to the paper).

Negatives are **not** explicitly mined — under `MultipleNegativesRankingLoss`, every other paper in the batch serves as an in-batch negative.

The `categories` (list), `versions`, and `authors` columns of the corpus additionally enable **field / year / author facets** in the downstream exploratory UI (not used as model inputs).

---

## Training Procedure

Training uses `SentenceTransformerTrainer` with `SentenceTransformerTrainingArguments` and `MultipleNegativesRankingLoss`. A **large per-device batch size** is central to the method: more in-batch negatives yield a stronger contrastive signal.

### Hyperparameters

| Hyperparameter | Value |
|---|---|
| Loss | `MultipleNegativesRankingLoss` (in-batch negatives) |
| Learning rate | ~2e-5 |
| Epochs | 1–3 (1–2 recommended to limit overfitting) |
| Warmup ratio | 0.1 |
| Batch sampler | `no_duplicates` |
| Mixed precision | bf16 + tf32 (H100/A100); fp16 (T4) |
| Checkpointing | eval/save steps; resume via `get_last_checkpoint` |

### Hardware / batch-size profiles

| GPU | Per-device train batch size | Precision |
|---|---|---|
| H100 | 256 | bf16 + tf32 |
| A100-40GB | 128 | bf16 + tf32 |
| L4 | 64 | — |
| T4 | 32 | fp16 |

Approximate wall-clock for `bge-small` ranges from **minutes to ~1 hour** depending on the number of pairs.

### Anti-overfitting measures

- Large, diverse pair set with **deduplication**.
- Only **1–2 epochs**.
- In-batch negatives (no single fixed negative set to memorize).

### Offline degradation

When `torch` / network are unavailable, a **TF-IDF retriever** (scikit-learn) stands in for the dense slot so hybrid search still runs, and an **identity reranker** replaces the cross-encoder. This is a deployment fallback, not the fine-tuned model itself.

---

## Evaluation

Metrics are computed via the `InformationRetrievalEvaluator` (or a self-contained equivalent) on a held-out `(query → relevant paper)` set.

| Metric | Role |
|---|---|
| **nDCG@10** | **Headline metric** |
| Recall@{1, 5, 10} | Coverage (so users know how exhaustive results are) |
| MRR@10 | Mean reciprocal rank of the first relevant result |

### Baselines

The fine-tuned retriever is required to beat **both**:

1. **BM25** (self-contained Okapi over tokenized title + abstract).
2. The **zero-shot base encoder** (un-fine-tuned `bge-small` / MiniLM).

The expected advantage is largest on **conceptual queries**, where dense semantic matching substantially outperforms exact-term BM25.

### Qualitative validation (offline)

End-to-end on the built-in corpus, intent routing is correct: e.g. the query *"dense passage retrieval"* ranks the Dense Passage Retrieval paper **#1**, and facets plus topic clusters are produced.

---

## Limitations and Biases

- **Synthetic-pair relevance is approximate.** Training pairs are self-supervised (`title → abstract`, `first-sentence → paper`); they *approximate* true relevance rather than encode human judgments, so the model can learn surface regularities of how titles/first sentences relate to abstracts rather than genuine topical relevance.
- **Recency / stale-index cutoff.** The primary corpus (`gfissore/arxiv-abstracts-2021`) has a **2021 cutoff**. Papers published after the snapshot are absent; the index is stale without continual ingestion (an incremental-update plan is required for production use).
- **Ranking bias and over-trust.** Results are a *ranked sample*, not an exhaustive set. Users may treat the top results as authoritative or complete. The system mitigates this by surfacing scores, result diversity, and explicit "broaden / narrow / related" suggestions, and by never hiding that the output is a ranked sample. Recall@k is reported precisely so users understand coverage, not just top-of-list precision.
- **Domain and language coverage.** arXiv-only and English-only; coverage and behavior outside that distribution are not characterized.
- **Title + abstract only.** Relevance signals present only in full text are invisible to the model.

---

## Ethical Considerations

- **Corpus bias / coverage:** The index reflects arXiv's English-language, field-skewed coverage and its 2021 recency cutoff. The cutoff and an incremental-update plan are documented so the staleness is explicit rather than hidden.
- **Query privacy:** Search queries can be sensitive and may reveal a user's research direction. Deployments should **minimize retention/logging of raw queries** and offer an on-prem option.
- **Over-trust in ranking:** To avoid users treating the top of the list as exhaustive or authoritative, the system shows scores and diversity and offers broaden/narrow suggestions.
- **Dual-use / misuse:** The system can surface dual-use research, but it only indexes **public** papers; it does not create or expose non-public information.
- **Robustness:** Out-of-domain or garbled queries are backstopped by a retrieval-coverage gate (query expansion + re-retrieval) and by graceful degradation (TF-IDF retriever + identity reranker + built-in corpus when `torch`/network are absent).

---

## How to Use

The model is a standard `SentenceTransformer`. Encode the query and the papers (`title + abstract`), then rank papers by cosine similarity. Embeddings are L2-normalized, so a dot product equals cosine similarity.

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("scisearch-retriever-bge-small-v1")

# Corpus side: encode "title + abstract"
papers = [
    "Dense Passage Retrieval for Open-Domain Question Answering. "
    "We introduce a dense retriever trained with in-batch negatives ...",
    "BM25 and Beyond. A study of classic lexical ranking functions ...",
]

# Query side. Optional BGE prompt (queries only):
query = "Represent this sentence for searching relevant passages: " \
        "dense passage retrieval"

# normalize_embeddings=True -> dot product == cosine similarity
paper_emb = model.encode(papers, normalize_embeddings=True)
query_emb = model.encode(query, normalize_embeddings=True)

scores = query_emb @ paper_emb.T          # cosine similarity per paper
ranking = np.argsort(-scores)             # descending
for rank, idx in enumerate(ranking, 1):
    print(rank, round(float(scores[idx]), 4), papers[idx][:60])
```

For production, the corpus embeddings are pre-computed and stored in a **FAISS** index (inner product over normalized vectors = cosine), with a NumPy brute-force fallback. Dense results are fused with BM25 via RRF (`k = 60`), and the head of the list is optionally re-scored by `cross-encoder/ms-marco-MiniLM-L-6-v2`.

---

## Versioning

The model is tracked via a model registry (`model_meta.json` + a `latest` pointer, referenced as `repo@revision`), enabling reproducible deployment and rollback.
