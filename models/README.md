# models/

Trained checkpoints live here (git-ignored — only this README is committed).

```
models/retriever/
├── bge-small-en-v1.5-YYYYMMDD-HHMMSS/   # fine-tuned bi-encoder (sentence-transformers format + model_meta.json)
└── latest/                              # pointer to the most recent version
```

- Train: `scisearch --config configs/train.yaml train`
- The search engine / API resolve `latest` automatically; with no checkpoint they fall back to
  the **base encoder** (zero-shot), and with no `sentence-transformers`/`torch` to a **TF-IDF**
  retriever — so the system always runs.
- Override the location with `SCISEARCH_MODEL_DIR`; on Colab point it at Google Drive so checkpoints
  survive disconnects (resume-safe via `get_last_checkpoint`).

Pretrained models (downloaded to `HF_HOME`, never committed):
`BAAI/bge-small-en-v1.5` (+ `sentence-transformers/all-MiniLM-L6-v2` fallback,
`malteos/scincl` / `allenai/specter2_base` domain options) and the reranker
`cross-encoder/ms-marco-MiniLM-L-6-v2`.
