# ☁️ Colab Training Guide — Scientific Literature Search

Fine-tune the **dense retriever** on Colab (Pro/Pro+), then test search and collect the
deliverables. The notebook auto-adapts to **H100 / A100 / L4 / T4** and **resumes** after a disconnect.

---

## 0. What you need
- Google **Colab** (Pro+ for H100/A100; T4/L4 also work — bigger GPU = bigger batch = more in-batch negatives).
- (Recommended) a **public GitHub repo** with this project, or upload the folder to Drive.

## 1. Get the project onto Colab
- **Option A (GitHub):** push this folder to `https://github.com/<you>/08_Scientific_Literature_Search`.
- **Option B (Drive):** upload `08_Scientific_Literature_Search/` to `MyDrive/08_Scientific_Literature_Search/`.

## 2. Drive layout (artifacts persist here → training survives disconnects)
Auto-created by the notebook:
```
MyDrive/scisearch/
├── data/        # corpus sample + signature
├── models/      # retriever/latest (fine-tuned bi-encoder)
├── index/       # (optional) cached FAISS index
├── runs/        # eval / error-analysis / benchmark JSON
├── submission/  # report.pdf + slides.pptx + bundle.zip
└── hf_cache/    # HuggingFace model + dataset cache
```

## 3. Configure & run
1. Open `notebooks/SciSearch_Colab_Training_H100_AUTOPILOT.ipynb` in Colab.
2. `Runtime → Change runtime type → GPU` (H100 if available).
3. **Cell 0 (Controls):** set `GIT_REPO_URL` (or use Drive); `BASE_MODEL=auto`; `CORPUS_LIMIT`
   (20000 default — raise for a bigger index); `MAX_TRAIN_PAIRS`; `EPOCHS`.
4. `Runtime → Run all` → installs Colab-safe deps (never touches torch), auto-profiles the GPU
   (batch size = #in-batch negatives), builds the corpus, and runs **autopilot** (cell 9):
   fine-tune → evaluate → analysis → `report.pdf` + `slides.pptx`.
5. **Disconnected?** Re-run **cell 9** — it resumes from the last checkpoint on Drive.

## 4. Verify it worked
- **Cell 10b / 11** — `evaluate` should show the fine-tuned retriever **beating BM25 and the
  zero-shot base** on **nDCG@10 / Recall@10** (the differentiation is clearest on conceptual queries).
- **Cell 12** — search returns relevant papers with **facets** (fields) and **topic clusters**.
- **Cell 13** — find `report.pdf` + `slides.pptx` in `…/submission/`.

## 5. Use the model later
```python
from sentence_transformers import SentenceTransformer
m = SentenceTransformer("…/models/retriever/latest")
emb = m.encode(["contrastive learning for sentence embeddings"], normalize_embeddings=True)
# cosine-search emb against your indexed paper embeddings
```
or simply: `scisearch search --query "..."` / `scisearch serve --ui`.

## Troubleshooting
- **OOM** → lower `per_device_train_batch_size` (the GPU profile sets it) or `CORPUS_LIMIT`.
- **Slow index build** → lower `CORPUS_LIMIT`; encoding the corpus is the one-time cost.
- **nDCG saturates** on a tiny corpus — use a larger `CORPUS_LIMIT` so BM25 / zero-shot / fine-tuned differ.
- **No `sentence-transformers`** → the engine falls back to TF-IDF + BM25 (still runs, lower quality).
