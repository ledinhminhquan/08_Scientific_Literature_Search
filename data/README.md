# data/

This project's **training pairs are generated from a paper corpus**, not labelled by hand.

- The corpus loads from a configurable HF dataset (default **`gfissore/arxiv-abstracts-2021`**,
  CC0, 2M papers with title + abstract + categories) and falls back to a **built-in mini-corpus**
  (~40 real NLP papers, `src/scisearch/data/samples.py`) when the dataset / network is unavailable —
  so the demo and tests always run offline.
- `(query, positive paper)` contrastive pairs are built by `src/scisearch/data/pairs.py`
  (title→abstract + concept-query→paper). In-batch negatives supply the rest.
- Build/inspect: `scisearch data` · `scisearch pairs`.

Large artifacts (FAISS index, embeddings, downloaded models) are **git-ignored** — only this
README is committed. All paths come from environment variables (`SCISEARCH_DATA_DIR`,
`SCISEARCH_INDEX_DIR`, `SCISEARCH_ARTIFACTS_DIR`, `HF_HOME`).

## Datasets (VERIFIED on HF)
- `gfissore/arxiv-abstracts-2021` — CC0-1.0, title+abstract+categories(list)+authors (corpus + facets).
- `BeIR/scifact` — real IR eval set (corpus + queries + qrels) for `InformationRetrievalEvaluator`.
- `mteb/scidocs-reranking` — optional secondary eval / hard-negative triplets.

> Note: the primary corpus snapshot ends in **2021** — for production, ingest newer papers
> incrementally (see `docs/continual_learning_monitoring.md`).
