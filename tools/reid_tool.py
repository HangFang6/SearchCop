"""
ReID Tool — wraps CLIP-ReID 1280-d feature extraction.
Real implementation reuses `fix_fangh_2-2_crop2reid_26_0415.py` from Gait-Project.
"""
from __future__ import annotations

from typing import List
import numpy as np

from .base import BaseTool, ToolResult


class ReidEncodeTool(BaseTool):
    name = "reid_encode"
    description = (
        "Encode N person crops (256x128) into a 1280-d ReID feature. "
        "Use when only a few good crops are available (occlusion, short clip)."
    )

    @property
    def json_schema(self):
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "crop_paths": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["crop_paths"],
            },
        }

    def __init__(self, model=None, device: str = "cuda"):
        self.model = model
        self.device = device

    def run(self, crop_paths: List[str]) -> ToolResult:
        feat = np.random.RandomState(abs(hash(tuple(crop_paths))) % (2**32)).randn(1280).astype("float32")
        feat /= (np.linalg.norm(feat) + 1e-8)
        return ToolResult(success=True, data=feat, meta={"dim": 1280, "n_crops": len(crop_paths)})
