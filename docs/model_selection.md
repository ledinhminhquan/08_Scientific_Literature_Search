# Model Selection & Optimization

**Project:** Exploratory Scientific Literature Search System (P08)
**Author:** Le Dinh Minh Quan (23127460) — NLP in Industry, final assignment
**Package:** `src/scisearch/`

This document explains the central modeling decision of P08: **why the retrieval core is a fine-tuned bi-encoder** (`BAAI/bge-small-en-v1.5`) rather than a BM25-only, zero-shot, or cross-encoder-only design. It covers the encoder architecture, the contrastive fine-tuning procedure (with the full training config and GPU profile), the baselines it must beat, the error-analysis methodology, and the accuracy/speed/complexity trade-offs that shaped the surrounding hybrid (RRF) + reranker stack.

---

## 1. The problem framing

Scientific literature search is fundamentally an **NLP/IR problem**. A researcher rarely types the exact lexical form an author used. They paraphrase ("models that find papers by meaning" vs. "dense passage retrieval"), use concepts instead of keywords, and want to *explore* a field rather than fetch one known document. Keyword search (BM25) is precise on exact terms but **blind to concept and paraphrase**. The modeling core therefore has to do *semantic* matching — mapping a query and a paper into a shared space where conceptual closeness, not surface overlap, drives the ranking.

The trainable model is a **dense passage retriever**: a bi-encoder that independently embeds the query and each paper, so that cosine similarity in embedding space approximates relevance.

---

## 2. Why a fine-tuned bi-encoder (vs. the alternatives)

| Option | What it does | Why it is **not** sufficient alone |
|---|---|---|
| **BM25-only** | Lexical, Okapi term-frequency scoring over tokenized title+abstract | Precise on exact terms but cannot match paraphrase/concept; misses semantically relevant papers that share no keywords. Strong on rare exact terms (IDs, acronyms), so we *keep* it — as a fusion partner, not the sole ranker. |
| **Zero-shot (un-fine-tuned) encoder** | Off-the-shelf `bge`/`MiniLM` embeddings, no domain adaptation | Generic semantic space; on scientific phrasing it under-ranks the right paper because it was never tuned on (query → paper) pairs from this corpus. It is one of the baselines we must *beat*. |
| **Cross-encoder-only** | Jointly encodes (query, paper) for a precise relevance score | Highest accuracy per pair, but **O(N) full forward passes per query** — cannot score a 100k–500k-paper corpus at query time. Unusable as a first-stage retriever. |
| **Fine-tuned bi-encoder (chosen)** | Pre-computes one vector per paper; query is one forward pass + ANN search | Fast (sub-linear ANN over cached embeddings) **and** semantic. Fine-tuning on in-domain pairs closes most of the accuracy gap to a cross-encoder for first-stage retrieval. |

The chosen design is therefore **not** "bi-encoder *instead of* the others" but a **layered stack** where the fine-tuned bi-encoder is the fast semantic first stage:

```
query ──► [bi-encoder dense]  ┐
       └─► [BM25 lexical]     ├─► RRF fusion ─► (gated) cross-encoder rerank ─► results
                              ┘
```

The bi-encoder is the part that is *trained*; BM25 is self-contained, the cross-encoder is used off-the-shelf, and RRF is parameter-free.

---

## 3. Encoder architecture

**Primary model: `BAAI/bge-small-en-v1.5`** (VERIFIED — MIT license, 33.4M parameters, 384-dim output).
**Fallback: `sentence-transformers/all-MiniLM-L6-v2`** (VERIFIED — Apache-2.0, 22.7M parameters, 384-dim).
Domain-specialized options considered: `malteos/scincl` (MIT, 768-dim) and `allenai/specter2_base` (Apache-2.0, 768-dim).

| Property | Value |
|---|---|
| Backbone | BERT-style transformer encoder (bi-encoder configuration) |
| Embedding dim | **384** |
| Pooling | **Mean pooling** over token embeddings |
| Normalization | **L2-normalized** vectors → inner product equals cosine similarity |
| Query prompt (optional) | `"Represent this sentence for searching relevant passages: "` prepended to **queries only** (bge convention) |

Two model-family decisions worth calling out:

- **Small over base.** `bge-small` (33.4M, 384-dim) is chosen over 768-dim domain models (`scincl`, `specter2_base`). The smaller dimension halves the FAISS index footprint and per-vector compute, and the 33.4M backbone fine-tunes in minutes-to-~1h. For first-stage retrieval that is then fused and reranked, the marginal accuracy of a 768-dim domain encoder does not justify ~2× the storage and latency.
- **384-dim + L2-normalize is what lets the search stack be cheap.** Because vectors are normalized, FAISS inner-product search *is* cosine ranking — no extra normalization at query time, and the numpy brute-force fallback is a single matrix multiply.

---

## 4. Training procedure

### 4.1 Objective: contrastive learning with in-batch negatives

The retriever is fine-tuned with **`sentence-transformers` `MultipleNegativesRankingLoss` (MNRL)** via `SentenceTransformerTrainer`. MNRL takes a batch of `(query, positive_paper)` pairs and treats **every other paper in the batch as a negative** for a given query. This is why the batch size is a *modeling* hyperparameter, not just a throughput knob: a batch of 256 gives each query 255 in-batch negatives, which makes the contrastive signal sharper and the learned space more discriminative.

### 4.2 Training data — generated pairs + real evaluation

Labeled (query → paper) data does not exist for the full arXiv corpus, so positive pairs are **generated** from `gfissore/arxiv-abstracts-2021` (VERIFIED — CC0-1.0, 2.0M rows; columns `id, authors, title, abstract, categories (LIST), versions`):

- **Lexical pairs:** `title → abstract` (the title is a natural short query for its own paper).
- **Conceptual pairs:** `first-sentence-of-abstract → paper` (a concept-level description mapped to the full paper).
- **Negatives:** supplied implicitly by MNRL in-batch negatives — no hard-negative mining needed for the primary run.

Evaluation uses **real, human-labeled IR data** so the headline numbers are trustworthy:

- **`BeIR/scifact`** (VERIFIED — corpus + queries + qrels) — drop-in for `InformationRetrievalEvaluator`.
- **`mteb/scidocs-reranking`** — secondary.

A built-in mini-corpus (~40 real NLP papers) ships with the package as an **offline fallback** so the demo and tests always run without network or torch.

### 4.3 Full training configuration

```python
# SentenceTransformerTrainingArguments + MultipleNegativesRankingLoss
training_config = {
    "loss": "MultipleNegativesRankingLoss",   # in-batch negatives
    "per_device_train_batch_size": 128,       # 128–256 → more negatives → better contrastive learning
    "learning_rate": 2e-5,
    "num_train_epochs": 2,                     # 1–3; kept low to avoid overfitting generated pairs
    "warmup_ratio": 0.1,
    "batch_sampler": "no_duplicates",          # avoid the same paper appearing twice as a negative
    "bf16": True,                              # H100 / A100  (fp16 on T4)
    "tf32": True,                              # H100 / A100
    "eval_steps": 500,
    "save_steps": 500,
    # resume support: get_last_checkpoint(output_dir)
}
```

Notes on each choice:

- **`per_device_train_batch_size` 128–256** — the dominant lever. More in-batch negatives directly improve the contrastive objective.
- **`lr ≈ 2e-5`, 1–3 epochs, `warmup_ratio` 0.1** — standard light-touch fine-tuning of a pretrained encoder.
- **`batch_sampler="no_duplicates"`** — prevents a paper from appearing as both a positive elsewhere and a "negative" in the same batch, which would otherwise corrupt the loss.
- **`bf16+tf32`** on H100/A100, **`fp16`** on T4 — precision picked per GPU.
- **Resume** via `get_last_checkpoint` plus `eval_steps`/`save_steps` checkpointing.

### 4.4 GPU profile

Batch size is set per accelerator to keep the in-batch-negative count as high as memory allows:

| GPU | Precision | `per_device_train_batch_size` |
|---|---|---|
| **H100** | bf16 + tf32 | **256** |
| **A100-40GB** | bf16 + tf32 | **128** |
| **L4** | fp16/bf16 | **64** |
| **T4** | **fp16** | **32** |

Rough wall-clock for `bge-small`: **minutes to ~1 hour**, depending on how many generated pairs are used.

### 4.5 Anti-overfitting

The training pairs are *generated*, so overfitting to spurious title→abstract patterns is the main risk. Mitigations: a **large, diverse pair set**, **deduplication**, **only 1–2 epochs**, and reliance on **in-batch negatives** (which change every batch) rather than a fixed negative set.

---

## 5. Baseline comparison

The fine-tuned retriever is required to **beat two baselines**:

1. **BM25** — self-contained Okapi BM25 over tokenized title+abstract.
2. **Zero-shot base encoder** — the *un-fine-tuned* `bge`/`MiniLM`.

**Metrics (IR):** **nDCG@10 (headline)**, **Recall@{1,5,10}**, **MRR@10**, computed on a held-out (query → relevant paper) set via `InformationRetrievalEvaluator` (or a self-contained evaluator offline).

**Expected outcome:** the fine-tuned bi-encoder beats both baselines, and the margin is **largest on conceptual queries** — exactly where dense >> BM25, because BM25 has no way to bridge paraphrase while the fine-tuned space was trained on concept→paper pairs. On rare-exact-term queries BM25 can still win individually, which is precisely why the production path **fuses** the two rather than discarding BM25.

> Offline (no torch), a **TF-IDF retriever** (sklearn) stands in for the dense slot so hybrid search still runs; the dense numbers above require the trained encoder.

---

## 6. Error-analysis approach

Error analysis is **per-query, dense-vs-BM25**, run via the CLI (`error-analysis`) and `src/scisearch/analysis/error_analysis.py`:

- For each evaluation query, record where the relevant paper ranks under **dense alone**, **BM25 alone**, and the **fused** ranking.
- Bucket queries into **dense wins / BM25 wins / both fail** to see *which* query types each retriever handles. Expectation: conceptual/paraphrase queries land in "dense wins"; rare-exact-term/ID queries land in "BM25 wins."
- Use the "BM25 wins" and "both fail" buckets to validate two design decisions: (a) keeping BM25 in the fusion is justified by real per-query wins, and (b) the agent's **D2 coverage gate** (query expansion + re-retrieval when too few results or top similarity is below threshold) targets the right failure cases.

This per-query win/loss view is what tells us the *fusion* (Section 7) is earning its keep rather than just averaging two rankers.

---

## 7. Search-stack design: RRF fusion + gated reranker

The bi-encoder does not act alone. The trade-off analysis (Section 8) drives a layered design where each component is added only where it pays for itself.

### 7.1 Reciprocal Rank Fusion (RRF)

Dense and BM25 rankings are combined with **Reciprocal Rank Fusion**, which is rank-based (no score calibration needed):

```
score(d) = Σ_r  1 / (k + rank_r(d)),   k = 60
```

where `r` ranges over the input rankings (dense, BM25). RRF is robust because it ignores the raw, incomparable score scales of the two retrievers and uses only positions. The agent **biases** the fusion by **duplicating the preferred ranking** based on detected intent (keyword intent → weight BM25 more; conceptual intent → weight dense more) — a cheap way to make fusion intent-aware without retraining.

### 7.2 Cross-encoder reranker (gated)

A **cross-encoder, `cross-encoder/ms-marco-MiniLM-L-6-v2`** (VERIFIED — Apache-2.0), re-scores the **top-k** fused results; an identity reranker is the fallback. Crucially, reranking is **gated** (agent decision **D3**): it fires **only when the head of the ranking is close/ambiguous** (small score margin). When the top result is unambiguous, the rerank is skipped, saving 150–400 ms of latency. This is the practical answer to "cross-encoders are accurate but slow" — pay for their accuracy only on the queries where it changes the answer.

---

## 8. Trade-offs

### 8.1 Accuracy vs. speed

| Stage | Accuracy role | Speed role |
|---|---|---|
| **Bi-encoder + FAISS ANN** | Semantic recall over the whole corpus | Fast: pre-computed vectors, sub-linear ANN; hot path ~tens of ms on a 100k–500k slice |
| **Cross-encoder rerank** | Highest precision at the head | Slow: O(k) joint forward passes → **gated** so it only runs when needed (+150–400 ms) |
| **small vs. base encoder** | base/768-dim domain models slightly more accurate | `bge-small` (384-dim) → smaller index, faster search, faster training; chosen |

The principle throughout: **use the cheap, fast component everywhere and the expensive, accurate component only where it changes the outcome.** End-to-end latency stays **sub-second** (hot path tens of ms; gated rerank adds 150–400 ms only when triggered).

### 8.2 Complexity vs. maintainability

- **Bi-encoder over cross-encoder-only** keeps the system *maintainable at scale*: paper embeddings are computed once, cached, and indexed in FAISS; adding papers means appending vectors, not re-scoring the corpus. Scalability comes from **FAISS index sharding + cached embeddings**.
- **RRF is parameter-free** (`k=60`), so fusion adds almost no tuning surface or failure modes.
- **Versioning** is handled by a model registry (`model_meta.json` + a `latest` pointer, `repo@revision`), so a retrained encoder can be swapped without touching the search code.
- **Graceful degradation** keeps the system runnable in constrained environments: **TF-IDF retriever** replaces the dense slot when torch is absent, the **identity reranker** replaces the cross-encoder, and the **built-in ~40-paper corpus** replaces the dataset — so the full pipeline (and the demo) runs offline with no GPU and no network.

---

## 9. Summary

A **fine-tuned bi-encoder (`BAAI/bge-small-en-v1.5`, 384-dim, mean-pooled, L2-normalized)** is the right retrieval core because it is the only option that is **both** semantically aware (unlike BM25), **domain-adapted** (unlike the zero-shot encoder), and **corpus-scalable** (unlike a cross-encoder-only design). Fine-tuning with `MultipleNegativesRankingLoss` and large in-batch-negative batches (128–256) yields a retriever that beats both BM25 and the zero-shot base — most clearly on conceptual queries — while **RRF fusion** preserves BM25's exact-term strength and a **gated cross-encoder reranker** buys head-of-list precision only when the ranking is ambiguous, keeping end-to-end latency sub-second.
