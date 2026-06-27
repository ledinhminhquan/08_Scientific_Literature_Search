<!-- DESIGN BRIEF — P08 Exploratory Scientific Literature Search. Single source of truth (verified research).
     Package src/scisearch/ mirrors the P02-P07 template. -->

# P08 — Exploratory Scientific Literature Search — Design Brief

> Single source of truth for the implementer. Every Hugging Face / dataset id below is marked **VERIFIED** (confirmed on the Hub during research). Keep ids exact — do not "correct" casing or hyphenation.

---

## 1. Problem & business value

**Problem.** Researchers, analysts, and R&D teams cannot keep up with the firehose of scientific papers. Keyword search (Google Scholar / arXiv listing) is precise on exact terms but blind to paraphrase and concept; it returns a flat list and offers no way to *explore* a literature — to see sub-topics, pivot to adjacent fields, or find "more like this." P08 delivers **exploratory search**: a query returns not just a ranked list but **facets, topic clusters, related-paper rails, and broaden/narrow guidance**, backed by a **fine-tuned dense retriever** that understands scientific phrasing.

**Users & jobs-to-be-done.**
- *Literature review* — "what's been done on contrastive learning for GNNs since 2020?" → ranked papers + sub-topic clusters + surveys to start from.
- *Scoping / gap-finding* — drill facets (field, year, author) to map a space.
- *Citation expansion* — "more like this paper" via embedding neighbours filtered by shared field.

**Business value.** Cuts time-to-relevant-paper; surfaces papers keyword search misses (semantic recall); turns one query into a guided exploration session (higher engagement, fewer dead-end searches); fully offline-capable and license-clean, so it can be embedded in internal R&D tooling without data-governance risk.

**Success metrics.**

| Type | Metric | Target / definition |
|---|---|---|
| Business | Time-to-first-relevant-result | ↓ vs. keyword baseline (user clicks a relevant paper sooner) |
| Business | Exploration depth | ≥1 facet/cluster/related interaction per session |
| Business | Zero-result rate | < 2% of queries (coverage gate D2 backstops this) |
| Business | Coverage | Corpus spans all of arXiv (cs.* + math/stat/eess) with field facets |
| Technical (retrieval) | **nDCG@10** | **Headline metric**; fine-tuned retriever must beat BM25 and zero-shot base |
| Technical (retrieval) | **Recall@{1,5,10}** | Report all three; Recall@10 is the recall headline |
| Technical (retrieval) | **MRR@10** | Rank-of-first-relevant quality |
| Technical (latency) | Hot path (dense+BM25+RRF) | ~tens of ms on a 100k–500k slice (FAISS IVF/HNSW + BM25) |
| Technical (latency) | Rerank (cross-encoder, gated) | +150–400 ms when invoked; **D3 gate skips it when head is unambiguous** |
| Technical (latency) | End-to-end (rule-only) | sub-second; brain (LLM) is off the critical path and wall-clock-budgeted |

---

## 2. VERIFIED stack table

Every id below was confirmed on the Hub. Licenses as reported.

### Corpus & training-source datasets

| id | VERIFIED | License | Size | Key fields | Role |
|---|---|---|---|---|---|
| `gfissore/arxiv-abstracts-2021` | **VERIFIED** | **cc0-1.0** | 2.0M rows, 1.5 GB parquet | `id, authors, title, abstract, categories(list), versions` | **PRIMARY corpus.** Only verified set with title+abstract+categories+authors → enables field/year/author facets. Parquet-native (no `trust_remote_code`). |
| `BeIR/scifact` | **VERIFIED** | cc-by-sa-4.0 | corpus 5.2K + queries 1.1K + qrels | `_id, title, text` / `_id, text` | **Real held-out IR eval** (claim→abstract); drop-in for `InformationRetrievalEvaluator`. |
| `mteb/scidocs-reranking` | **VERIFIED** | mteb (en) | 4.0K test / 4.0K val | `query, positive[], negative[]` | Secondary eval / ready-made hard-negative triplets. |
| `CShorten/ML-ArXiv-Papers` | **VERIFIED** | afl-3.0 | 117.6K rows | `title, abstract` | Smallest/fastest smoke-test corpus (cs.LG only, no facets). |
| `allenai/scirepeval` | **VERIFIED** | per-config | 12.4M | `search` config: real (query,paper) pairs; `fos` labels | Optional real supervised pairs / Fields-of-Study labels. |
| `somewheresystems/dataclysm-arxiv` | **VERIFIED** | cc0-1.0 | 3.4M, 7.3 GB | same schema + precomputed bge-small embeddings | Optional heavy alt (only if you want precomputed vectors). |

**Avoid as primary:** `arxiv-community/arxiv_dataset` (Viewer 501, loading script + Kaggle JSON); `BeIR/scidocs`, `mteb/scidocs` (no categories — keep only as eval); `ccdv/arxiv-classification` (full-body text, no title/abstract split).

**Offline fallback corpus:** bundle `data/seed_papers.jsonl` (~200–500 cs.CL rows drawn from the CC0 primary, redistribution-permitted) + a precomputed seed FAISS index + the static `ARXIV_CAT_NAMES` facet map, so the demo always runs with zero network.

### Models

| Role | id | VERIFIED | License | Params / dim | Load note |
|---|---|---|---|---|---|
| **Retriever base (PRIMARY, trainable)** | `BAAI/bge-small-en-v1.5` | **VERIFIED** | **mit** | 33.4M / 384-dim | `SentenceTransformer(...)`. Prepend query prompt `"Represent this sentence for searching relevant passages: "` to **queries only**. |
| Retriever base (fallback) | `sentence-transformers/all-MiniLM-L6-v2` | **VERIFIED** | apache-2.0 | 22.7M / 384-dim | Native ST; no prompt needed. Same 384-dim → index code portable. |
| Retriever base (domain ablation) | `malteos/scincl` | **VERIFIED** | mit | 109.9M / 768-dim | `library:sentence-transformers` → loads directly. Strong SciDocs. |
| Retriever base (domain alt) | `allenai/specter2_base` | **VERIFIED** | apache-2.0 | ~110M / 768-dim | SPECTER2; needs `models.Transformer` + mean-pool wrapper. Best for "related work" NN. |
| **Reranker (PRIMARY)** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | **VERIFIED** | apache-2.0 | 22.7M / ~90 MB | `CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")`. Hub canonicalizes to `cross-encoder/ms-marco-MiniLM-L6-v2`; both load. CPU-friendly. |
| Reranker (higher-quality alt) | `BAAI/bge-reranker-base` | **VERIFIED** | mit | 278M | Heavier, optional. |
| LLM brain (optional, advisory) | `claude-opus-4-8` | **VERIFIED** | Anthropic API | — | 1M context, `thinking={"type":"adaptive"}`, structured-output routing JSON. Auto-disables to rule-only if no key. |
| FoS enrichment (optional) | `TimSchopf/nlp_taxonomy_classifier` | **VERIFIED** | — | — | Classifies title+abstract into NLP taxonomy for richer cs.CL facets. |

### Libraries

```
sentence-transformers>=3.0   # bi-encoder train (SentenceTransformerTrainer) + CrossEncoder reranker
faiss-cpu                    # dense ANN vector store (IndexFlatIP small / IndexIVFFlat|HNSW large)
rank-bm25                    # BM25Okapi sparse keyword scoring   (bm25s = faster drop-in)
scikit-learn                 # KMeans / AgglomerativeClustering + TfidfVectorizer (cluster labels)
datasets, numpy              # corpus loading, RRF fusion / NN math
fastapi + uvicorn, gradio    # serving surfaces
anthropic                    # optional LLM brain
```
Optional: `keybert`/`yake` (extractive synthetic queries), a T5 doc2query model for LLM synthetic queries.

**External (not HF-loadable):** NLP-KG Fields-of-Study OWL hierarchy (421 fields, 7 levels) — overlay for NLP drill-down facets. Ship a trimmed JSON in-repo.

---

## 3. System pipeline

```
                                  ┌──────────────────────────────────────────────┐
   raw query ───────────────────► │ S0 UNDERSTAND                                │
                                  │  parse_query · extract_filters · expand_query │
                                  └───────────────┬──────────────────────────────┘
                                                  ▼  (intent + filters + expansions)
                                  ┌──────────────────────────────────────────────┐
                                  │ RETRIEVE  (filters pushed down as index mask) │
                                  │                                               │
                                  │   dense_search(k=50)        bm25_search(k=50) │
                                  │   FAISS, cosine/IP          rank-bm25 / bm25s │
                                  │        └──────────┬────────────────┘          │
                                  │                   ▼                           │
                                  │            rrf_fuse(k0=60, weights)           │
                                  │            → candidates (top ~200)            │
                                  └───────────────┬──────────────────────────────┘
                                                  ▼
        ┌──────────────── pre-rerank pool (top ~200) used for faceting ───────────┐
        ▼                                         ▼                                ▼
  facets(field/year/        ┌────────────────────────────┐            (held for cluster/related)
   author/venue)            │ RERANK (gated by D3)        │
  count aggregation         │ cross-encoder re-score      │
        │                   │ top-50 → top-k, survey/cited │
        │                   │ boost                        │
        │                   └──────────────┬──────────────┘
        │                                  ▼
        │                   ┌────────────────────────────┐
        │                   │ EXPLORE                     │
        │                   │  cluster_results (KMeans/   │
        │                   │   agglomerative + TF-IDF    │
        │                   │   labels)                   │
        │                   │  related(paper_id) NN +     │
        │                   │   shared-category boost     │
        │                   └──────────────┬──────────────┘
        └───────────────────────────┬──────┘
                                     ▼
                          ┌────────────────────────────┐
                          │ PRESENT                     │
                          │  results[] + facets[] +     │
                          │  clusters[] + related[] +   │
                          │  strategy (broaden/narrow)  │
                          └────────────────────────────┘
```

RRF fuses **rank** lists (scale-invariant across cosine vs. BM25): `RRF(d) = Σ_r 1/(k0 + rank_r(d))`, `k0=60`. Optional per-ranker weights `w_r/(k0+rank_r(d))` let D1 tilt dense vs. BM25.

---

## 4. Trainable model plan — fine-tuned bi-encoder

**Objective.** Fine-tune `BAAI/bge-small-en-v1.5` (MIT, 33M; fallback `all-MiniLM-L6-v2`; domain ablation `malteos/scincl`) into a scientific-paper dense retriever using **`MultipleNegativesRankingLoss`** (MNRL / InfoNCE / in-batch negatives) on (query → paper) pairs mined offline.

### 4.1 Corpus → training pairs (offline, from `gfissore/arxiv-abstracts-2021`)

Positive paper text = `title + ". " + abstract`. MNRL needs only positives (every other in-batch positive is an implicit negative); supply a `negative` column for hard negatives when available.

Mix (~target proportions):
1. **title → abstract** (~40%) — strongest, cheapest signal; title acts as a terse query.
2. **first-sentence-of-abstract → abstract** (~20%) — partial-text → full-paper matching.
3. **co-category / citation-related** (~20%) — `(paper_A, paper_B)` sharing a fine-grained category (e.g. both `cs.CL`); use cited-pairs (scirepeval `cite_prediction`) when available (the SPECTER/SciNCL signal).
4. **synthetic doc2query queries** (~20%) — generate 1–3 natural-language queries per abstract. Extractive (KeyBERT/YAKE, CPU, deterministic) or LLM doc2query (`doc2query/all-t5-base-v1` / `BeIR/query-gen-msmarco-t5-base-v1`). Offline analogue of GPL/InPars.

**Hard negatives (recommended, biggest lever after loss):** for each anchor, BM25-retrieve top-k (`BM25Okapi` over tokenized title+abstract), discard the true positive, keep ranks ~10–50 as hard negatives → triplets `(anchor, positive, negative)`. MNRL consumes the negative column *and* adds in-batch negatives on top.

**Dedup:** drop exact-duplicate queries and trivial `query ⊂ positive` cases; use `BatchSamplers.NO_DUPLICATES` to avoid false negatives in-batch.

### 4.2 Training (SentenceTransformerTrainer, ST v3+) — H100 config dict

```python
import torch
from sentence_transformers import (
    SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.evaluation import InformationRetrievalEvaluator

# ---- H100 fast-math toggles (BEFORE model creation) ----
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

MODEL_ID = "BAAI/bge-small-en-v1.5"          # fallback: "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_ID)
model.max_seq_length = 256                    # titles+abstracts are short; lower seq = bigger batch
model.prompts = {"query": "Represent this sentence for searching relevant passages: "}  # BGE: queries only

loss = MultipleNegativesRankingLoss(model)    # scale=20.0 default (= 1/temperature)

CONFIG = dict(
    output_dir                  = "bge-small-arxiv-retriever",
    num_train_epochs            = 1,          # 1 for large sets, up to 3 for small; MNRL overfits fast
    per_device_train_batch_size = 256,        # H100 80GB — THE key knob (255 negatives/anchor); push 384–512 @ seq128
    per_device_eval_batch_size  = 256,
    learning_rate               = 2e-5,       # range 1e-5..3e-5
    warmup_ratio                = 0.1,
    lr_scheduler_type           = "cosine",
    weight_decay                = 0.01,
    bf16                        = True,        # bf16 on Hopper/Ampere (more stable than fp16 for contrastive)
    tf32                        = True,
    batch_sampler               = BatchSamplers.NO_DUPLICATES,
    eval_strategy               = "steps", eval_steps = 200,
    save_strategy               = "steps", save_steps = 200, save_total_limit = 2,
    logging_steps               = 50,
    load_best_model_at_end      = True,
    metric_for_best_model       = "eval_scifact_cosine_ndcg@10",  # MATCH evaluator name prefix (see gotcha)
    greater_is_better           = True,
    dataloader_num_workers      = 4,
    report_to                   = "none",
    seed                        = 42,
)
args = SentenceTransformerTrainingArguments(**CONFIG)

ir_evaluator = InformationRetrievalEvaluator(
    queries=queries, corpus=corpus, relevant_docs=relevant_docs,
    name="scifact",                           # prefixes metric keys → eval_scifact_cosine_ndcg@10
    show_progress_bar=True,
)

trainer = SentenceTransformerTrainer(
    model=model, args=args,
    train_dataset=train_ds,                   # columns: ["anchor","positive"] (+ "negative" for hard negs)
    eval_dataset=eval_ds, loss=loss, evaluator=ir_evaluator,
)
trainer.train()
model.save_pretrained("bge-small-arxiv-retriever/final")
```

> **Metric-key gotcha:** `InformationRetrievalEvaluator` prefixes metrics with its `name`. With `name="scifact"` the key is `eval_scifact_cosine_ndcg@10` — set `metric_for_best_model` to the exact printed key (run one eval step and read the log to confirm).
> **For huge effective batch on small VRAM:** use `CachedMultipleNegativesRankingLoss` (GradCache) or `gradient_accumulation_steps`.

### 4.3 Evaluation, metrics, baselines

**Evaluator:** `InformationRetrievalEvaluator` — needs `queries={qid:text}`, `corpus={cid:text}`, `relevant_docs={qid:set(cid)}`. Attach to trainer (in-loop) and run standalone on the final model.

**Metrics (report all):** **Recall@{1,5,10}**, **MRR@10**, **nDCG@10 (headline)**; also Accuracy@k, Precision@k, MAP@100.

**Held-out eval sets:**
1. **Real — `BeIR/scifact`:** corpus/queries/qrels map directly into the three dicts. Comparable to the BEIR leaderboard.
2. **Self-constructed:** hold out N papers; title (or synthetic doc2query query) = query, the paper = the single relevant doc, rest of corpus = distractors. Guarantees an eval set for any corpus offline.

**Baselines to beat (both):**
- **BM25** (`rank_bm25` `BM25Okapi` over tokenized title+abstract) — same qrels, same metrics via a small loop.
- **Zero-shot base encoder** — run the evaluator on un-fine-tuned `BAAI/bge-small-en-v1.5` (and `malteos/scincl`) before training.

| System | R@1 | R@5 | R@10 | MRR@10 | nDCG@10 |
|---|---|---|---|---|---|
| BM25 (rank-bm25) | | | | | |
| bge-small (zero-shot) | | | | | |
| scincl (zero-shot) | | | | | |
| **bge-small fine-tuned (ours)** | | | | | |

The fine-tuned model must beat BM25 and zero-shot on nDCG@10 / Recall@10; the gap is the headline result.

### 4.4 Anti-overfitting checklist

- **Scale + diversity over raw count** — aim ≥100k pairs, ideally 500k+ (in-batch negatives sample your data distribution).
- **Dedup aggressively** before training; `BatchSamplers.NO_DUPLICATES` prevents false negatives in-batch.
- **1–2 epochs** (3 only for small sets); MNRL overfits fast. `load_best_model_at_end` on the IR metric protects regardless.
- In-batch negatives are themselves regularizing (different every step); bigger batches amplify this.
- **Don't over-tune LR** — 2e-5 sweet spot; higher LRs blow up contrastive logits.
- **Eval with the IR evaluator, not loss** — loss keeps dropping while retrieval metrics plateau; `cosine_ndcg@10` is the honest `metric_for_best_model`.

### 4.5 GPU-profile table

Tuned for bge-small / MiniLM (~22–33M, 384-dim, seq 256). Batch maximizes in-batch negatives within VRAM.

| GPU | VRAM | `per_device_train_batch_size` (≈ neg/anchor) | Precision | Notes |
|---|---|---|---|---|
| **H100** | 80 GB | **256** (255) — push **384–512** @ seq128 | `bf16=True, tf32=True` | Target. Compute/data-loader-bound, not VRAM-bound; raise batch to ~90% VRAM. |
| A100 | 40/80 GB | 128 / 256 | `bf16=True, tf32=True` | Near H100-class for a 33M model. |
| L4 (Colab paid) | 24 GB | 64–96 | `bf16=True` | Cut batch if seq→512. |
| T4 (Colab free) | 16 GB | 32 (31) | `fp16=True` (no bf16; `tf32=False`) | Fewer negatives ⇒ lower quality; use `CachedMNRL` / grad-accum to recover effective batch. |

**Throughput / time (H100, bge-small, seq256):** ~2,000–4,000 pairs/sec; ~12–18 GB peak VRAM @ batch 256; **1 epoch over 1M pairs ≈ 10–20 min**, 100k pairs ≈ 2–4 min. **Precision rule:** bf16 on Hopper/Ada/Ampere, fp16 on T4 (Turing has no bf16); `tf32=True` only helps Ampere+.

---

## 5. Agent architecture — deterministic FSM with optional LLM brain

A deterministic finite-state machine over the hybrid pipeline. The optional Claude brain (`claude-opus-4-8`) is **advisory only**: rules always produce a decision; the LLM may only *propose* a value already in the legal set; any missing key / HTTP error / malformed JSON / out-of-set value / wall-clock-budget overrun → the rule value wins. **Same query + same index + brain disabled ⇒ identical output.**

### 5.1 States & decision points

```
ENTRY raw query
   │
   ▼
S0 UNDERSTAND ── parse_query · extract_filters · expand_query
   │
   �████ D1 QUERY-TYPE ROUTING (+ apply metadata filters) ████
   │   keyword-heavy → wBM25=0.7/wDense=0.3
   │   hybrid        → 0.5/0.5
   │   conceptual    → wBM25=0.3/wDense=0.7
   ▼
S1 RETRIEVE ── dense ∥ bm25 → rrf_fuse
   │
   �████ D2 COVERAGE GATE ████
   │   too few / low score (n<MIN_HITS=8 or top1<TAU_LOW=0.45)
   │        → expand_query + re-retrieve (loop guard MAX_EXPANSIONS=2)
   │   too many (n>MAX_HITS=2000, filters loose) → tighten (AND terms / require filters)
   │   ok → fall through
   ▼
   �████ D3 RERANK GATE (latency saver) ████
   │   rerank ONLY if ambiguous: (top1−topk)/top1 < SPREAD=0.15 OR top-15 split across ≥3 facets
   │   else skip cross-encoder (saves ~150–400 ms). cap RERANK_K=50
   ▼
S2 RERANK (cross-encoder, gated) ── survey/highly-cited boost
   │
   ▼
S3 EXPLORE ── cluster_facets (KMeans/agglomerative, auto-k by silhouette) · related_papers (specter2 NN)
   │
   �████ D4 PRESENTATION STRATEGY ████
   │   low diversity (1–2 tight clusters)  → suggest BROADEN + surface surveys
   │   high diversity (many scattered)     → suggest NARROW + field-split clusters
   │   medium → balanced
   ▼
S4 PRESENT (terminal) → {results, facets, clusters, related, strategy}
```

**D1 — Query-type routing (+ filters).** Features: quoted phrases, boolean ops, IDF/rare-token proportion, explicit author/year/field filter, query length. Filters (`author=`, `year=2019..2023`, `field=cs.LG`) pushed down as a pre-filter mask before fusion. *Brain:* classifies intent, proposes `w_bm25 ∈ {0.3,0.5,0.7}`; rule validates in-grid and that detected filters weren't dropped.

**D2 — Coverage gate.** On `n_hits, top1_score, mean_top_k`. Too few/low → expand + re-retrieve (cap 2); too many → tighten; ok → through. *Brain:* may suggest ≤5 expansions.

**D3 — Rerank gate (latency).** Rerank only when head is ambiguous (small score spread or top window split across ≥3 facets); else skip. *Brain:* may flip skip→rerank, **never** rerank→skip (quality-safe).

**D4 — Presentation strategy.** Cluster top-N (default 100), measure diversity (#facets / inter-cluster distance) → broaden / narrow / balanced. *Brain:* labels clusters + writes the 1–2 line refine note; rules own membership and the broaden/narrow choice.

### 5.2 Tool contracts (pure, typed, side-effect-free except index load)

```python
# Understanding
parse_query(q: str) -> Query              # {text, tokens, quoted_phrases[], booleans[], idf_profile}
extract_filters(q: str) -> Filters        # {authors|None, year_min|None, year_max|None, fields|None, residual_text}
expand_query(q: str, k: int = 5) -> list[str]

# Retrieval
dense_search(text, k, mask=None) -> list[Hit]   # FAISS over bge-small
bm25_search(text, k, mask=None) -> list[Hit]    # rank-bm25 / bm25s
rrf_fuse(runs: list[list[Hit]], weights, k0=60) -> list[Hit]   # Hit={doc_id,score,source,rank}

# Rerank (gated)
rerank(query, hits, top_k=50) -> list[Hit]      # cross-encoder/ms-marco-MiniLM-L-6-v2

# Exploration
cluster_facets(hits, n_max=100) -> list[Cluster]   # {cluster_id,label,doc_ids[],centroid,size,keywords[]}
related_papers(seed_ids, k=10) -> list[Doc]        # specter2 nearest-neighbours
suggest_strategy(clusters, diversity) -> Strategy  # {mode∈{broaden,narrow,balanced}, suggestions[], surface_surveys}

# Brain (optional, advisory — schema-validated; None on any failure → rule path)
brain_route(query, features) -> RouteAdvice | None       # D1
brain_expansions(query) -> list[str] | None              # D2
brain_rerank_hint(head_hits) -> bool | None              # D3
brain_label_clusters(clusters) -> dict[id,str] | None    # D4
```

### 5.3 Brain contract (`claude-opus-4-8`)

Single structured-output call per decision, only when `BRAIN_ENABLED` and key present. `thinking={"type":"adaptive"}` (adaptive only — no `budget_tokens`); `output_config={"format":{"type":"json_schema","schema":ROUTE_SCHEMA}}`. Guardrails: `w_bm25` constrained to the rule grid; LLM filters must be a subset/refinement of regex-extracted ones (may normalise "Yann LeCun"→author match, may not invent a year); per-decision wall-clock budget (~800 ms) — exceed → rule wins. Any exception → `None` → rule path.

### 5.4 Worked example

**Query:** `"contrastive learning for graph neural networks since 2020, not vision"`
1. **S0** — no quotes/booleans; filters `{year_min:2020}`; expansions `["graph contrastive learning","GCL","self-supervised GNN","node representation learning"]`.
2. **D1** — conceptual + year filter ⇒ `wDense=0.7/wBM25=0.3`, push `year≥2020` mask; brain normalises "not vision" → negative term; in-grid ⇒ accepted.
3. **S1** — dense (bge-small) ∥ bm25 → `rrf_fuse([dense,bm25],[0.7,0.3])` → 240 hits, top1=0.61.
4. **D2** — `8 < 240 < 2000`, `top1 0.61 > 0.45` ⇒ **ok**, no expansion.
5. **D3** — spread `(0.61−0.52)/0.61 = 0.15` and top-15 across 3 facets ⇒ **ambiguous → rerank** top-50 with `cross-encoder/ms-marco-MiniLM-L-6-v2`; vision-GNN papers demoted.
6. **S3** — 4 facets (*Graph SSL pretraining*, *Molecular GNNs*, *RecSys GNNs*, *Theory/augmentations*); `related_papers` (specter2) → 8 neighbours incl. 2 surveys.
7. **D4** — medium-high diversity ⇒ mode=`narrow`, suggestions `["restrict to Graph SSL pretraining","add field cs.LG","explore: molecular property prediction"]`, `surface_surveys=True`.
8. **S4** — emit `{results, facets, clusters, related, meta.strategy}`.

---

## 6. Deployment

### 6.1 Architecture

```
        CLIENTS:  Gradio UI  │  CLI (litsearch …)  │  curl
                       │ POST /search │              │
                       ▼              ▼              ▼
        ┌──────────────────────────────────────────────────────┐
        │  FastAPI app (uvicorn)                                │
        │   POST /search  {query, filters, top_k, options}      │
        │   GET  /healthz  /version  /facets/{id}               │
        │            │                                          │
        │            ▼                                          │
        │     AGENT (FSM D1–D4 + optional brain)                │
        │            │ reads ┌──────────────────────────────┐   │
        │            ├──────►│ Index store:                 │   │
        │            │       │  FAISS shards (dense)         │   │
        │            │       │  BM25 shards                  │   │
        │            │       │  meta.parquet (title/abstract/│   │
        │            │       │   fields/year/venue/citations)│   │
        │            │       │  embeddings.npy + id_map.json │   │
        │            ▼       └──────────────────────────────┘   │
        │   Anthropic API (claude-opus-4-8, optional)           │
        └──────────────────────────────────────────────────────┘
                       │ {results[], facets[], clusters[], related[], meta}
                       ▼
   Packaging: Docker (uvicorn + gradio @ :7860, /ui) ──► HF Space (Docker SDK)
              models pulled from Hub on first boot, cached to /data
```

### 6.2 API contract

```http
POST /search   {"query":"...", "filters":{"authors":[],"year_min":2020,"year_max":null,"fields":["cs.LG"]},
                "top_k":20, "options":{"use_brain":true,"rerank":"auto","cluster":true}}
```
```jsonc
// 200 OK
{ "results":[ {"doc_id":"2006.04131","title":"...","authors":["..."],"year":2020,
               "fields":["cs.LG"],"abstract":"...","score":0.71,"reranked":true} ],
  "facets":[ {"name":"field","values":[{"value":"cs.LG","count":142},{"value":"stat.ML","count":51}]},
             {"name":"year","values":[{"value":"2023","count":60},{"value":"2022","count":71}]} ],
  "clusters":[ {"cluster_id":0,"label":"Graph contrastive / SSL pretraining","size":63,
                "keywords":["augmentation","infomax","node2vec"],"doc_ids":["..."]} ],
  "related":[ {"doc_id":"2005.10243","title":"A survey on graph SSL","reason":"survey neighbour (specter2)"} ],
  "meta":{ "route":{"intent":"conceptual+filtered","w_bm25":0.3,"brain_used":true},
           "decisions":{"D2":"ok","D3":"reranked","D4":"narrow"},
           "strategy":{"mode":"narrow","suggestions":["restrict to cluster 0"],"surface_surveys":true},
           "timings_ms":{"retrieve":41,"rerank":180,"cluster":22,"brain":540,"total":790},
           "index_version":"arxiv-2021@v3","model_version":"bge-small-1.5+ce-minilm-L6" } }
```

### 6.3 Surfaces

- **Gradio (`gr.Blocks`):** left **facet sidebar** (author/year/field inputs + clickable counts) · center ranked **results grouped by topic-cluster accordion** (LLM labels) · right rail **related papers + surveys** + broaden/narrow chips. Clicking a facet value re-issues `/search`. `facets[]`→sidebar, `clusters[]`→result groups, `related[]`→rail, `meta.strategy`→bottom callout.
- **CLI (`litsearch`):** `litsearch "graph contrastive learning" --year-min 2020 --field cs.LG --top-k 20 --json`; `--no-brain` for pure-rule deterministic mode; `--explain` dumps the D1–D4 decision trace.
- **Docker:** single image, `uvicorn app:api` on `:7860` + Gradio at `/ui`; models & index cached to mounted `/data` (cold start downloads once).
- **HF Space (Docker SDK):** `ANTHROPIC_API_KEY` as a Space secret (brain auto-disables to rule-only if absent — Space still fully works). First boot pulls `BAAI/bge-small-en-v1.5` + `cross-encoder/ms-marco-MiniLM-L-6-v2` + a `gfissore/arxiv-abstracts-2021` slice and builds/loads FAISS+BM25.

### 6.4 Latency · scalability · versioning

- **Latency.** Dense ANN (FAISS IVF/HNSW) + BM25 are the hot path (~tens of ms on 100k–500k). The cross-encoder is the cost; **D3 gate skips it when the head is unambiguous** — the single biggest lever. Brain off the critical path for rule-only mode, wrapped in a wall-clock budget, and LRU-cached per normalised query.
- **Scalability.** **Index sharding** (by field or hash) — each shard has its own FAISS + BM25; fan-out then RRF-merge across shards (horizontal scale-out). **Cached embeddings** precomputed once and memory-mapped (FAISS on disk); query embeddings + brain decisions LRU-cached. Rerank only ever touches ≤50 candidates regardless of corpus size.
- **Versioning.** `index_version` (corpus slice + embedder + FAISS params) and `model_version` (embedder + reranker ids) pinned in `meta` and on `/version`. Re-embedding builds a new index version side-by-side (blue/green); API pins or rolls forward. Brain prompt/schema versioned (`route_schema@v1`).

---

## 7. Risks, limitations, ethics

| Risk | Description | Mitigation |
|---|---|---|
| **Corpus bias / coverage** | `gfissore/arxiv-abstracts-2021` ends in 2021 and is arXiv-only → misses recent work, non-arXiv venues, non-English papers, and over-represents CS/physics/math. Field facets inherit arXiv's taxonomy gaps. | State coverage window in the UI; offer corpus-refresh path; document arXiv-only scope; allow swapping in larger/newer slices. |
| **Stale index** | Embeddings/BM25 reflect a fixed snapshot; new papers absent until re-index. | `index_version` surfaced in every response and `/version`; blue/green re-embed; schedule re-index. |
| **Query privacy** | Queries can reveal sensitive research directions; optional LLM brain sends query text to the Anthropic API. | Brain is opt-in (`use_brain`/`--no-brain`); rule-only mode is fully local; don't log raw queries by default; document the data flow. |
| **Over-trust in ranking** | Users may treat the top result as authoritative; reranker/retriever can be confidently wrong; "highly-cited"/"survey" boosts can entrench popular-but-dated work. | Always show clusters + facets + related (exploration, not a single answer); expose scores and the D1–D4 decision trace via `--explain`/`meta`; label survey/cited boosts. |
| **Dual-use** | Powerful literature discovery can aid harmful research as readily as benign. | Standard scientific-search posture; no capability to synthesize methods beyond surfacing public abstracts; rely on corpus being public preprints. |
| **Synthetic-query artifacts** | doc2query / template queries may bias the retriever toward templated phrasings. | Mix four pair sources; evaluate on **real** queries (`BeIR/scifact`); keep BM25 in the hybrid as a lexical backstop. |
| **License hygiene** | Some candidate corpora are NC/AFL. | Primary path is CC0 (`gfissore/...`) + MIT/Apache models; NC sets (e.g. `neuralwork/arxiver`) used for research/eval only. |

---

## 8. Repo module map

Package `src/scisearch/` mirrors the P02–P07 layout.

| Module | Responsibility |
|---|---|
| `src/scisearch/data/` | Corpus loading (`gfissore/arxiv-abstracts-2021` via `datasets`; offline `data/seed_papers.jsonl` fallback). Category splitting (`categories[0].split()`), `ARXIV_CAT_NAMES` facet map, trimmed NLP-KG FoS hierarchy JSON. Builds `meta.parquet` (title, abstract, fields, year, venue, citation_count). |
| `src/scisearch/models/` | Model loaders/wrappers: bi-encoder (`BAAI/bge-small-en-v1.5` + BGE query prompt; fallback `all-MiniLM-L6-v2`; domain `malteos/scincl`/`specter2_base`), `CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")`, optional `claude-opus-4-8` client. |
| `src/scisearch/search/` | Hybrid retrieval: FAISS index build/load (`IndexFlatIP`/`IVFFlat`/`HNSW`, `IndexIDMap2`), BM25 (`rank_bm25`/`bm25s`), `rrf_fuse(k0=60)`, reranker call, survey/cited boost. Exploration: `facets`, `cluster_results` (KMeans/agglomerative + TF-IDF labels), `related`, `expand_query` (PRF/embedding). Persists `papers.faiss`, `bm25.pkl`, `embeddings.npy`, `id_map.json`. |
| `src/scisearch/training/` | Pair construction (title→abstract, firstsent→abstract, co-category, synthetic doc2query, BM25 hard negatives), `SentenceTransformerTrainer` + MNRL config (§4.2), checkpoint/save. |
| `src/scisearch/agent/` | FSM orchestrator (S0–S4), decision predicates D1–D4, tool contracts (§5.2), optional brain contract + guardrails/fallback (§5.3). |
| `src/scisearch/api/` | FastAPI app (`/search`, `/healthz`, `/version`, `/facets/{id}`), request/response schemas, Gradio `gr.Blocks` UI, `litsearch` CLI (`--no-brain`, `--explain`, `--json`). |
| `src/scisearch/analysis/` | IR evaluation: `InformationRetrievalEvaluator` harness, BM25 + zero-shot baselines, Recall@{1,5,10}/MRR@10/nDCG@10 reporting table (§4.3). |
| `src/scisearch/autoreport/` | Auto-generated run/eval reports (metrics tables, baseline comparisons, decision-trace summaries). |
| `src/scisearch/monitoring/` | Latency timings (`meta.timings_ms`), zero-result rate, brain-fallback rate, index freshness. |
| `src/scisearch/automation/` | Index (re)build & blue/green versioning, corpus-refresh / re-embed jobs, model/index pinning. |
| `src/scisearch/grading/` | Self-grading harness against P02 exemplar criteria (artifacts present, metrics beat baselines, surfaces runnable, offline fallback works). |

**Artifacts loaded at startup** (query-time cost = encode + search + (gated) rerank): `papers.faiss`, `bm25.pkl`, `embeddings.npy` + `id_map.json`, `meta.parquet`, `ARXIV_CAT_NAMES` + NLP-KG FoS JSON, trained `bge-small-arxiv-retriever/final`.

**Build path:** follow the P02 exemplar pattern (`C:\Users\ADMIN\.claude\projects\D--NLP-Industry-Projects\memory\p02-resume-exemplar-pattern.md`) using the VERIFIED ids above — corpus `gfissore/arxiv-abstracts-2021` (cc0), retriever `BAAI/bge-small-en-v1.5` (mit), reranker `cross-encoder/ms-marco-MiniLM-L-6-v2` (apache-2.0), domain NN `allenai/specter2_base`/`malteos/scincl`, brain `claude-opus-4-8`.
