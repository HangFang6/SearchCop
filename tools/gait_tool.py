"""
Gait Tool — wraps OpenGait BaselineDemo multi-window feature extraction.
Real implementation reuses `fix_fangh_3_gait_feature_26_0415.py` from Gait-Project.
"""
from __future__ import annotations

from typing import List
import numpy as np

from .base import BaseTool, ToolResult


class GaitEncodeTool(BaseTool):
    name = "gait_encode"
    description = (
        "Encode a sequence of silhouette frames into a multi-window gait feature vector. "
        "Use when the target is walking and the full body is visible across >= 30 frames."
    )

    @property
    def json_schema(self):
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "silhouette_dir": {"type": "string", "description": "Path to silhouette png folder"},
                    "windows": {"type": "array", "items": {"type": "integer"}, "default": [30, 45, 60]},
                },
                "required": ["silhouette_dir"],
            },
        }

    def __init__(self, model=None, device: str = "cuda"):
        self.model = model           # OpenGait model handle (lazy-load in setup)
        self.device = device

    def run(self, silhouette_dir: str, windows: List[int] = (30, 45, 60)) -> ToolResult:
        # TODO(Coder Agent): wire to OpenGait BaselineDemo
        # For now: deterministic stub so the rest of the pipeline is testable.
        feat = np.random.RandomState(abs(hash(silhouette_dir)) % (2**32)).randn(256).astype("float32")
        feat /= (np.linalg.norm(feat) + 1e-8)
        return ToolResult(success=True, data=feat, meta={"dim": 256, "windows": list(windows)})
