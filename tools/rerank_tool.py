"""
Multi-vector Rerank Tool — combined / topk_mean / max / set2set scoring.

Ports the rerank strategy from
`Gait-Project/fix_fangh_5_reid_search_26_0415.py` so that query-side
multi-window (gait) or multi-crop (reid) features can be matched against
gallery-side multi-feature sets via an All-to-All similarity matrix.

Inputs:
  query_features   : np.ndarray (M, D)  — already L2 normalized
  gallery_features : Dict[db_id -> np.ndarray (N, D)]  — already L2 normalized
  top_k            : int
  strategy         : "combined" (default) | "topk_mean" | "max" | "set2set"

Output:
  ranked: [{db_id, score}, ...]   sorted desc by score
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .base import BaseTool, ToolResult


# Defaults match fix_fangh_5
RERANK_TOPK = 5
COMBINED_W_TOPK = 0.6
COMBINED_W_MEAN = 0.3
COMBINED_W_MAX = 0.1


def _topk_mean(sim: np.ndarray, k: int = RERANK_TOPK) -> float:
    flat = sim.flatten()
    k = max(1, min(k, len(flat)))
    idx = np.argpartition(flat, -k)[-k:]
    return float(np.mean(flat[idx]))


def _max(sim: np.ndarray) -> float:
    return float(np.max(sim))


def _set2set(sim: np.ndarray) -> float:
    q2g = float(np.mean(np.max(sim, axis=1)))
    g2q = float(np.mean(np.max(sim, axis=0)))
    return 0.5 * (q2g + g2q)


def _combined(sim: np.ndarray, k: int = RERANK_TOPK) -> float:
    flat = sim.flatten()
    k = max(1, min(k, len(flat)))
    idx = np.argpartition(flat, -k)[-k:]
    return (COMBINED_W_TOPK * float(np.mean(flat[idx]))
            + COMBINED_W_MEAN * float(np.mean(flat))
            + COMBINED_W_MAX * float(np.max(flat)))


_STRATS = {
    "combined":  _combined,
    "topk_mean": _topk_mean,
    "max":       lambda s, k=RERANK_TOPK: _max(s),
    "set2set":   lambda s, k=RERANK_TOPK: _set2set(s),
}


def rerank_score(query_feats: np.ndarray,
                 gallery_feats: np.ndarray,
                 strategy: str = "combined",
                 k: int = RERANK_TOPK) -> float:
    """All-to-all similarity → scalar score."""
    sim = query_feats @ gallery_feats.T          # (M, N), since both L2-normalized
    fn = _STRATS.get(strategy, _combined)
    return float(fn(sim, k=k) if fn.__code__.co_argcount > 1 else fn(sim))


class MultivectorRerankTool(BaseTool):
    name = "multivector_rerank"
    description = (
        "Refine a coarse Top-C candidate list by computing All-to-All similarity "
        "between query multi-features and each candidate's multi-features, then "
        "scoring with `combined` (default), `topk_mean`, `max`, or `set2set`."
    )

    @property
    def json_schema(self):
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "candidate_db_ids": {"type": "array", "items": {"type": "integer"}},
                    "strategy":         {"type": "string",
                                         "enum": ["combined", "topk_mean", "max", "set2set"],
                                         "default": "combined"},
                    "top_k":            {"type": "integer", "default": 10},
                },
                "required": ["candidate_db_ids"],
            },
        }

    def __init__(self, stub: Optional[bool] = None):
        import os
        if stub is None:
            stub = os.getenv("SEARCHCOP_STUB", "0") == "1"
        self.stub = stub

    def run(self,
            candidate_db_ids: List[int],
            query_features: Optional[np.ndarray] = None,
            gallery_features: Optional[Dict[int, np.ndarray]] = None,
            strategy: str = "combined",
            top_k: int = 10) -> ToolResult:
        if not candidate_db_ids:
            return ToolResult(success=False, error="empty candidate_db_ids")

        if self.stub:
            rng = np.random.RandomState(abs(hash((tuple(candidate_db_ids), strategy))) % (2**32))
            scores = sorted(rng.uniform(0.4, 0.95, size=len(candidate_db_ids)).tolist(),
                            reverse=True)
            ranked = [{"db_id": int(d), "score": float(s)}
                      for d, s in zip(candidate_db_ids, scores)][:top_k]
            return ToolResult(success=True,
                              data={"ranked": ranked, "strategy": strategy},
                              meta={"stub": True, "n_candidates": len(candidate_db_ids)})

        if query_features is None or gallery_features is None:
            return ToolResult(success=False,
                              error="query_features and gallery_features are required in real mode")
        q = np.asarray(query_features, dtype="float32")
        if q.ndim == 1:
            q = q[None, :]

        out = []
        for db_id in candidate_db_ids:
            g = gallery_features.get(int(db_id))
            if g is None or g.size == 0:
                continue
            g = np.asarray(g, dtype="float32")
            if g.ndim == 1:
                g = g[None, :]
            score = rerank_score(q, g, strategy=strategy)
            out.append({"db_id": int(db_id), "score": float(score)})
        out.sort(key=lambda x: x["score"], reverse=True)
        return ToolResult(success=True,
                          data={"ranked": out[:top_k], "strategy": strategy},
                          meta={"stub": False, "n_candidates": len(candidate_db_ids)})
