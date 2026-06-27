"""Vector index for dense retrieval — FAISS when available, numpy brute-force else.

Embeddings are L2-normalized so inner product == cosine similarity. Brute-force
numpy is fine for the corpus sizes here (<= a few hundred thousand papers).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np


class VectorIndex:
    def __init__(self):
        self.ids: List[str] = []
        self.emb: np.ndarray = np.zeros((0, 0), dtype=np.float32)
        self._faiss = None

    def build(self, ids: List[str], embeddings: np.ndarray) -> "VectorIndex":
        self.ids = list(ids)
        self.emb = np.ascontiguousarray(embeddings.astype(np.float32))
        try:
            import faiss  # lazy
            index = faiss.IndexFlatIP(self.emb.shape[1])
            index.add(self.emb)
            self._faiss = index
        except Exception:
            self._faiss = None
        return self

    def search(self, query_emb: np.ndarray, k: int = 50) -> List[Tuple[str, float]]:
        q = np.ascontiguousarray(query_emb.astype(np.float32)).reshape(1, -1)
        k = min(k, len(self.ids))
        if k == 0:
            return []
        if self._faiss is not None:
            scores, idxs = self._faiss.search(q, k)
            return [(self.ids[i], float(s)) for i, s in zip(idxs[0], scores[0]) if i >= 0]
        sims = (self.emb @ q[0])
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        return [(self.ids[i], float(sims[i])) for i in top]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, ids=np.array(self.ids), emb=self.emb)

    @classmethod
    def load(cls, path: str | Path) -> "VectorIndex":
        data = np.load(Path(path), allow_pickle=True)
        inst = cls()
        return inst.build(list(data["ids"]), data["emb"])


__all__ = ["VectorIndex"]
