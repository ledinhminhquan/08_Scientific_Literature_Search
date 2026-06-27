# Deploying to a Hugging Face Space (Gradio)

The Gradio UI (`app/gradio_app.py`) runs the full agent: a query → ranked results +
facet/cluster sidebar + decision log.

## Option A — Gradio SDK Space (simplest)

1. Create a Space → SDK **Gradio**.
2. Add at the Space repo root:
   - `app.py`:
     ```python
     from scisearch.api.ui import build_ui
     demo = build_ui()
     ```
   - `requirements.txt` (copy `requirements_colab.txt` **plus** `torch`),
   - the `src/` folder (so `scisearch` imports), or `pip install git+https://github.com/<you>/08_Scientific_Literature_Search`.
3. Hardware: a **T4/A10 GPU** speeds index build + encoding; CPU works for the built-in/small corpus.
4. Push your fine-tuned retriever to the Hub and set `SCISEARCH_MODEL_DIR`, or bake it in.

## Option B — Docker Space (REST API + UI)

1. Create a Space → SDK **Docker**; push this repo (it has a `Dockerfile`).
2. The image serves `scisearch.api.app_combined:app` on port **7860** (REST `/search` + `/related` + Gradio at `/ui`).

## Notes
- The first request builds the index (encodes the corpus). Lower `data.corpus_limit` for a fast demo,
  or pre-build + cache the FAISS index.
- Without `sentence-transformers`/`torch`, the engine falls back to **TF-IDF + BM25** so the Space still works.
- All ids in the default stack are permissive (bge-small MIT, MiniLM/cross-encoder Apache, gfissore corpus CC0).
- Search queries can be sensitive — disable query logging (`serving.log_queries: false`) for privacy-sensitive deployments.
