"""Fine-tune the dense retriever (sentence-transformers bi-encoder, MNRL).

Contrastive training with MultipleNegativesRankingLoss (in-batch negatives) on
``(query, positive_paper)`` pairs. A large batch size => more negatives => better
contrastive learning. Resume-safe; bf16/tf32 on H100/A100. Heavy imports lazy.
"""

from __future__ import annotations

import json
from typing import Dict, Optional

from ..config import AppConfig
from ..logging_utils import get_logger
from ..models import model_registry as reg
from ..data.corpus import load_corpus
from ..data.pairs import build_pairs

logger = get_logger(__name__)


def train_retriever(cfg: AppConfig, limit: Optional[int] = None, resume: bool = True,
                    base_model: Optional[str] = None) -> Dict:
    import torch
    from datasets import Dataset
    from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                       SentenceTransformerTrainingArguments, losses)
    from sentence_transformers.evaluation import InformationRetrievalEvaluator
    from transformers.trainer_utils import get_last_checkpoint

    mc = cfg.model
    model_id = base_model or mc.base_model
    torch.backends.cuda.matmul.allow_tf32 = bool(mc.tf32)

    papers = load_corpus(cfg)
    pairs = build_pairs(papers, cfg.data)
    train_pairs = pairs["train_pairs"]
    if limit:
        train_pairs = train_pairs[:limit]
    logger.info("Training %s on %d pairs (eval %d queries / %d corpus)",
                model_id, len(train_pairs), pairs["n_eval_queries"], pairs["n_corpus"])

    instr = mc.query_instruction or ""
    train_ds = Dataset.from_dict({"anchor": [instr + q for q, _ in train_pairs],
                                  "positive": [d for _, d in train_pairs]})

    model = SentenceTransformer(model_id)
    model.max_seq_length = mc.max_seq_length
    loss = losses.MultipleNegativesRankingLoss(model)

    evaluator = InformationRetrievalEvaluator(
        queries={qid: instr + q for qid, q in pairs["queries"].items()},
        corpus=pairs["corpus"], relevant_docs=pairs["relevant"],
        name="papers", ndcg_at_k=[10], mrr_at_k=[10], accuracy_at_k=[1, 5, 10],
        precision_recall_at_k=[1, 5, 10], show_progress_bar=False)

    out_dir = mc.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    args = SentenceTransformerTrainingArguments(
        output_dir=str(out_dir), num_train_epochs=mc.num_train_epochs, learning_rate=mc.learning_rate,
        per_device_train_batch_size=mc.per_device_train_batch_size,
        warmup_ratio=mc.warmup_ratio, weight_decay=mc.weight_decay,
        bf16=bool(mc.bf16), fp16=bool(mc.fp16),
        eval_strategy="steps", save_strategy="steps", eval_steps=mc.eval_steps, save_steps=mc.save_steps,
        save_total_limit=2, logging_steps=50, seed=mc.seed, report_to=[],
        batch_sampler="no_duplicates",   # better in-batch negatives
    )
    trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=train_ds, loss=loss,
                                         evaluator=evaluator)
    last = get_last_checkpoint(str(out_dir)) if resume and out_dir.exists() else None
    if last:
        logger.info("Resuming from %s", last)
    trainer.train(resume_from_checkpoint=last)

    metrics = {}
    try:
        metrics = {k: float(v) for k, v in evaluator(model).items() if isinstance(v, (int, float))}
    except Exception as exc:
        logger.info("final evaluator failed (%s)", exc)

    version = reg.make_version(model_id)
    final_dir = out_dir / version
    model.save(str(final_dir))
    reg.write_metadata(final_dir, version=version, base_model=model_id,
                       dataset_signature={"train_pairs": len(train_pairs), "corpus": pairs["n_corpus"],
                                          "hf_corpus": cfg.data.hf_corpus, "seed": cfg.data.seed},
                       metrics=metrics)
    reg.update_latest_pointer(out_dir, final_dir)
    (out_dir / "last_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info("Retriever training done -> %s", final_dir)
    return {"version": version, "model_dir": str(final_dir), "base_model": model_id, "metrics": metrics}


__all__ = ["train_retriever"]
