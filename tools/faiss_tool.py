"""
FAISS Tool — coarse Top-K retrieval against the existing Gait-Project
indexes:
  faiss_indexes/gait_faiss_index.faiss      + gait_faiss_id_map.npy
  faiss_indexes/reid_faiss_index.faiss      + reid_faiss_id_map.npy

The tool keeps both indexes loaded (lazy) and routes by `modality`.

Output schema:
  candidates: [{db_id: int, score: float, faiss_id: int}, ...]
"""
from __future__ import annotations

import os
from typing import List, Optional

import numpy as np

from .base import BaseTool, ToolResult


# Default location: Gait-Project/faiss_indexes/
def _default_index_dir() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, "..", "..", "Gait-Project", "faiss_indexes"))


class _FaissBackend:
    _instances = {}      # key: (modality, dir) → backend

    def __init__(self, index_path: str, id_map_path: str):
        import faiss   # lazy
        self._faiss = faiss
        self.index = faiss.read_index(index_path)
        self.id_map = np.load(id_map_path).tolist()

    @classmethod
    def get(cls, modality: str, index_dir: str) -> "_FaissBackend":
        key = (modality, os.path.abspath(index_dir))
        if key not in cls._instances:
            if modality == "gait":
                idx = os.path.join(index_dir, "gait_faiss_index.faiss")
                m   = os.path.join(index_dir, "gait_faiss_id_map.npy")
            elif modality == "reid":
                idx = os.path.join(index_dir, "reid_faiss_index.faiss")
                m   = os.path.join(index_dir, "reid_faiss_id_map.npy")
            else:
                raise ValueError(f"unknown modality: {modality}")
            if not (os.path.exists(idx) and os.path.exists(m)):
                raise FileNotFoundError(f"FAISS index not found: {idx}")
            cls._instances[key] = cls(idx, m)
        return cls._instances[key]

    def search(self, query_vec: np.ndarray, top_k: int):
        v = query_vec.reshape(1, -1).astype("float32")
        # Caller should already have L2-normalized; do it again to be safe
        n = np.linalg.norm(v) + 1e-12
        v = v / n
        D, I = self.index.search(v, min(top_k, self.index.ntotal))
        return D[0].tolist(), I[0].tolist()


class FaissSearchTool(BaseTool):
    name = "faiss_search"
    description = (
        "Coarse Top-K retrieval over the existing person gallery. "
        "Pick modality='gait' or 'reid' depending on which feature you have."
    )

    @property
    def json_schema(self):
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "modality": {"type": "string", "enum": ["gait", "reid"]},
                    "top_k":    {"type": "integer", "default": 50},
                },
                "required": ["modality"],
            },
        }

    def __init__(
        self,
        index_dir: Optional[str] = None,
        stub: Optional[bool] = None,
    ):
        self.index_dir = index_dir or _default_index_dir()
        if stub is None:
            stub = os.getenv("SEARCHCOP_STUB", "0") == "1" or not os.path.isdir(self.index_dir)
        self.stub = stub

    def run(self,
            modality: str,
            query_vec: Optional[np.ndarray] = None,
            top_k: int = 50) -> ToolResult:
        if modality not in ("gait", "reid"):
            return ToolResult(success=False, error=f"bad modality: {modality}")
        if query_vec is None:
            return ToolResult(success=False, error="query_vec is required (np.ndarray)")
        q = np.asarray(query_vec, dtype="float32").flatten()

        if self.stub:
            rng = np.random.RandomState(abs(hash((modality, top_k, q.shape[0]))) % (2**32))
            scores = np.sort(rng.uniform(0.5, 0.95, size=top_k))[::-1]
            db_ids = rng.randint(1, 5000, size=top_k).tolist()
            cands = [{"db_id": int(d), "score": float(s), "faiss_id": i}
                     for i, (d, s) in enumerate(zip(db_ids, scores))]
            return ToolResult(success=True, data={"candidates": cands, "modality": modality},
                              meta={"top_k": top_k, "stub": True})

        backend = _FaissBackend.get(modality, self.index_dir)
        scores, ids = backend.search(q, top_k)
        cands = []
        for f_id, s in zip(ids, scores):
            if f_id < 0:
                continue
            db_id = backend.id_map[f_id]
            cands.append({"db_id": int(db_id), "score": float(s), "faiss_id": int(f_id)})
        return ToolResult(success=True,
                          data={"candidates": cands, "modality": modality},
                          meta={"top_k": top_k, "stub": False})
