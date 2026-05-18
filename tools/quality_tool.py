"""
Quality Score Tool.

Cheap, opencv-only quality estimator for a single crop:
  - sharpness   (Laplacian variance, normalized)
  - bbox_size   (relative area)
  - edge_margin (distance to image border)
  - aspect_ok   (aspect ratio inside reasonable range)

Final `score` ∈ [0, 1] is a weighted combination. The agent uses this to
decide whether to bother running the heavier gait/reid encoders on a track.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

from .base import BaseTool, ToolResult


def laplacian_variance(image_gray: np.ndarray) -> float:
    """Approximate Laplacian variance without importing cv2 if not needed."""
    try:
        import cv2
        return float(cv2.Laplacian(image_gray, cv2.CV_64F).var())
    except Exception:
        # Pure numpy fallback (1D-separable Laplacian approximation)
        kx = np.array([1.0, -2.0, 1.0])
        gx = np.apply_along_axis(lambda r: np.convolve(r, kx, "same"), 1, image_gray.astype("f4"))
        gy = np.apply_along_axis(lambda r: np.convolve(r, kx, "same"), 0, image_gray.astype("f4"))
        return float(np.var(gx + gy))


class QualityScoreTool(BaseTool):
    name = "quality_score"
    description = (
        "Score the visual quality of a person crop in [0,1] from sharpness, "
        "bbox size, edge margin, and aspect ratio. Use to skip heavy encoders "
        "when a crop is hopeless."
    )

    @property
    def json_schema(self):
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path":     {"type": "string"},
                    "frame_w":        {"type": "integer"},
                    "frame_h":        {"type": "integer"},
                    "bbox":           {"type": "array", "items": {"type": "number"},
                                       "description": "[x1, y1, x2, y2]"},
                },
                "required": ["image_path"],
            },
        }

    def __init__(self, stub: Optional[bool] = None):
        if stub is None:
            stub = os.getenv("SEARCHCOP_STUB", "0") == "1"
        self.stub = stub

    def run(self,
            image_path: str,
            frame_w: int = 1920,
            frame_h: int = 1080,
            bbox=None) -> ToolResult:
        if self.stub:
            # deterministic 0.7 ± 0.2
            rng = np.random.RandomState(abs(hash(image_path)) % (2**32))
            score = float(np.clip(0.5 + rng.uniform(-0.2, 0.4), 0.0, 1.0))
            return ToolResult(success=True,
                              data={"score": score, "components": {"stub": True}},
                              meta={"stub": True})

        if not os.path.isfile(image_path):
            return ToolResult(success=False, error=f"image not found: {image_path}")
        try:
            import cv2
        except Exception as e:
            return ToolResult(success=False, error=f"opencv missing: {e}")

        img = cv2.imread(image_path)
        if img is None:
            return ToolResult(success=False, error="cv2.imread returned None")
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 1. sharpness — squash via tanh on log-variance
        var = laplacian_variance(gray)
        sharpness = float(np.tanh(np.log1p(var) / 6.0))   # 0..~1

        # 2. bbox size: prefer area >= 5% of frame
        if bbox and len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            area_rel = max(0.0, (x2 - x1) * (y2 - y1)) / max(1.0, frame_w * frame_h)
        else:
            area_rel = (w * h) / max(1.0, frame_w * frame_h)
        bbox_size = float(np.clip(area_rel / 0.10, 0.0, 1.0))     # 10% of frame -> 1

        # 3. edge margin (only meaningful if bbox provided)
        if bbox and len(bbox) == 4:
            margin = min(bbox[0], bbox[1], frame_w - bbox[2], frame_h - bbox[3])
            edge_margin = float(np.clip(margin / 20.0, 0.0, 1.0))   # 20px -> 1
        else:
            edge_margin = 1.0

        # 4. aspect (person ~2:1 H:W is healthy; allow 1.4..3.0)
        aspect = h / max(1.0, w)
        aspect_ok = 1.0 if 1.4 <= aspect <= 3.0 else max(0.0, 1.0 - abs(aspect - 2.0) / 2.0)

        score = float(np.clip(0.45 * sharpness + 0.25 * bbox_size
                              + 0.15 * edge_margin + 0.15 * aspect_ok, 0.0, 1.0))
        return ToolResult(
            success=True,
            data={"score": score,
                  "components": {"sharpness": sharpness, "bbox_size": bbox_size,
                                 "edge_margin": edge_margin, "aspect_ok": aspect_ok}},
            meta={"stub": False, "h": h, "w": w},
        )
