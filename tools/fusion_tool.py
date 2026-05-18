"""
Evidence Fusion Tool — fuse multiple ranked lists into a final Top-K.

Three strategies:
  - weighted: normalize each list's scores to [0,1] and take a weighted sum
  - borda:    Borda count (positional voting)
  - rrf:      Reciprocal Rank Fusion, score(i) = sum_l 1/(k+rank_l(i))

Inputs:
  ranked_lists: List[List[{db_id, score}]]
  weights:      Optional[List[float]]  (used by 'weighted')
  k:            int  (used by 'rrf', default 60)
  top_k:        int
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional

from .base import BaseTool, ToolResult


def _normalize(scores: List[float]) -> List[float]:
    if not scores:
        return scores
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-12:
        return [1.0 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]


def fuse_weighted(lists, weights=None) -> Dict[int, float]:
    if weights is None:
        weights = [1.0] * len(lists)
    out: Dict[int, float] = defaultdict(float)
    for w, lst in zip(weights, lists):
        if not lst:
            continue
        norm = _normalize([item.get("score", 0.0) for item in lst])
        for s, item in zip(norm, lst):
            out[int(item["db_id"])] += w * s
    return out


def fuse_borda(lists) -> Dict[int, float]:
    out: Dict[int, float] = defaultdict(float)
    for lst in lists:
        n = len(lst)
        for rank, item in enumerate(lst):
            out[int(item["db_id"])] += (n - rank)
    return out


def fuse_rrf(lists, k: int = 60) -> Dict[int, float]:
    out: Dict[int, float] = defaultdict(float)
    for lst in lists:
        for rank, item in enumerate(lst, start=1):
            out[int(item["db_id"])] += 1.0 / (k + rank)
    return out


_STRATS = {"weighted": fuse_weighted, "borda": fuse_borda, "rrf": fuse_rrf}


class EvidenceFuseTool(BaseTool):
    name = "evidence_fuse"
    description = (
        "Fuse multiple ranked candidate lists (e.g. gait_rank, reid_rank, "
        "vlm_rank) into a single Top-K. Strategy: weighted | borda | rrf."
    )

    @property
    def json_schema(self):
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "ranked_lists": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "items": {"type": "object",
                                      "properties": {"db_id": {"type": "integer"},
                                                     "score": {"type": "number"}},
                                      "required": ["db_id", "score"]},
                        },
                    },
                    "strategy":   {"type": "string",
                                   "enum": ["weighted", "borda", "rrf"], "default": "rrf"},
                    "weights":    {"type": "array", "items": {"type": "number"}},
                    "rrf_k":      {"type": "integer", "default": 60},
                    "top_k":      {"type": "integer", "default": 10},
                },
                "required": ["ranked_lists"],
            },
        }

    def run(self,
            ranked_lists: List[List[Dict]],
            strategy: str = "rrf",
            weights: Optional[List[float]] = None,
            rrf_k: int = 60,
            top_k: int = 10) -> ToolResult:
        if not ranked_lists:
            return ToolResult(success=False, error="ranked_lists is empty")
        if strategy == "weighted":
            scores = fuse_weighted(ranked_lists, weights=weights)
        elif strategy == "borda":
            scores = fuse_borda(ranked_lists)
        elif strategy == "rrf":
            scores = fuse_rrf(ranked_lists, k=rrf_k)
        else:
            return ToolResult(success=False, error=f"unknown strategy: {strategy}")

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        out = [{"db_id": int(d), "score": float(s)} for d, s in ranked]
        return ToolResult(success=True,
                          data={"ranked": out, "strategy": strategy},
                          meta={"n_lists": len(ranked_lists)})
