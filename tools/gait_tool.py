"""
Gait Tool — multi-window OpenGait feature extraction.

Real implementation reuses the inference logic of
`Gait-Project/fix_fangh_3_gait_feature_26_0415.py`. The heavy
modules (torch, opengait, baselineDemo) are imported lazily, so
on a machine without GPU/dependencies the tool can still be
imported and unit-tested in stub mode.

Design:
  - The tool takes a directory of 64x64 silhouette PNGs and returns:
      mean_feature (D,)            — L2-normalized mean over all windows
      window_features (N, D)       — per-window features (multi-vector rerank)
  - Window sizes [30, 45, 60] with stride 10 + a "full" window match the
    Gait-Project setup, ensuring feature compatibility with the existing
    FAISS index (so we can search the production gallery directly).
"""
from __future__ import annotations

import os
from glob import glob
from typing import List, Optional

import numpy as np

from .base import BaseTool, ToolResult


# ----- knobs (kept identical to fix_fangh_3) -----
WINDOW_SIZES = (30, 45, 60)
STRIDE = 10


# ----------------------------------------------------------------------
# Lazy backend (only imported when stub=False and run() is actually called)
# ----------------------------------------------------------------------
class _OpenGaitBackend:
    """Wraps OpenGait BaselineDemo. Loaded once, cached in process."""

    _instance = None

    def __init__(
        self,
        gait_project_dir: str,
        cfg_path: str = "./configs/gaitbase/gaitbase_da_gait3d.yaml",
        model_type: str = "BaselineDemo",
    ):
        # Heavy imports — only happen on the H20 server
        import sys
        if gait_project_dir not in sys.path:
            sys.path.insert(0, gait_project_dir)
        import torch                                                # noqa: F401
        from opengait.utils import config_loader                    # noqa: F401
        import model.baselineDemo as baselineDemo                    # noqa: F401

        cfgs = config_loader(os.path.join(gait_project_dir, cfg_path)
                             if not os.path.isabs(cfg_path) else cfg_path)
        Model = getattr(baselineDemo, model_type)
        m = Model(cfgs, training=False)
        m.requires_grad_(False)
        m.eval()
        if torch.cuda.is_available():
            m.to("cuda")
        self.model = m
        self._torch = torch

    @classmethod
    def get(cls, gait_project_dir: str, **kw) -> "_OpenGaitBackend":
        if cls._instance is None:
            cls._instance = cls(gait_project_dir, **kw)
        return cls._instance

    def encode(self, frames_paths: List[str]) -> Optional[np.ndarray]:
        """Reproduce fix_fangh_3.extract_feature_from_frames."""
        import cv2
        seqs = []
        for p in frames_paths:
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            if img.shape != (64, 64):
                img = cv2.resize(img, (64, 64))
            seqs.append(img)
        if not seqs:
            return None
        seqs_np = np.array([seqs])  # [1, T, 64, 64]
        ipts = ([seqs_np], ["temp_sid"], np.array([[len(seqs)]]))
        with self._torch.no_grad():
            delabeled = self.model.inputs_pretreament(ipts)
            _, embs = self.model.forward(delabeled)
        return embs.cpu().data.numpy().flatten()


# ----------------------------------------------------------------------
def generate_sliding_windows(frame_list: List[str],
                             window_sizes=WINDOW_SIZES,
                             stride: int = STRIDE) -> List[List[str]]:
    """Identical to fix_fangh_3.generate_sliding_windows for feature compatibility."""
    out: List[List[str]] = []
    n = len(frame_list)
    for w in window_sizes:
        if n >= w:
            for s in range(0, n - w + 1, stride):
                out.append(frame_list[s:s + w])
    if n > 0:
        out.append(frame_list)   # the "full" window
    return out


def list_silhouettes(silhouette_dir: str) -> List[str]:
    files = sorted(glob(os.path.join(silhouette_dir, "*.png"))
                   + glob(os.path.join(silhouette_dir, "*.jpg")))
    return files


# ----------------------------------------------------------------------
class GaitEncodeTool(BaseTool):
    name = "gait_encode"
    description = (
        "Encode a sequence of 64x64 silhouette frames into multi-window gait "
        "features and an L2-normalized mean vector. Use when the target is "
        "walking and the silhouette directory contains >= 30 frames."
    )

    @property
    def json_schema(self):
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "silhouette_dir": {
                        "type": "string",
                        "description": "Path to a directory with PNG silhouettes (any of the per-frame masks).",
                    },
                    "windows": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "default": list(WINDOW_SIZES),
                        "description": "Sliding-window sizes (frames). Default 30/45/60.",
                    },
                },
                "required": ["silhouette_dir"],
            },
        }

    def __init__(
        self,
        gait_project_dir: Optional[str] = None,
        cfg_path: str = "./configs/gaitbase/gaitbase_da_gait3d.yaml",
        stub: Optional[bool] = None,
    ):
        # Default: SearchCop/../Gait-Project
        if gait_project_dir is None:
            here = os.path.dirname(os.path.abspath(__file__))
            gait_project_dir = os.path.normpath(os.path.join(here, "..", "..", "Gait-Project"))
        self.gait_project_dir = gait_project_dir
        self.cfg_path = cfg_path
        # Stub auto-decision: enable stub if env var asks for it OR if the dir is missing
        if stub is None:
            stub = os.getenv("SEARCHCOP_STUB", "0") == "1" or not os.path.isdir(gait_project_dir)
        self.stub = stub

    def run(self,
            silhouette_dir: str,
            windows: Optional[List[int]] = None) -> ToolResult:
        if not os.path.isdir(silhouette_dir):
            return ToolResult(success=False, error=f"silhouette_dir not found: {silhouette_dir}")
        frames = list_silhouettes(silhouette_dir)
        if len(frames) == 0:
            return ToolResult(success=False, error="no silhouette frames found")

        win_sizes = tuple(windows) if windows else WINDOW_SIZES
        win_lists = generate_sliding_windows(frames, window_sizes=win_sizes)

        if self.stub:
            # Deterministic stub for local dev / unit tests
            rng = np.random.RandomState(abs(hash(silhouette_dir)) % (2**32))
            mean = rng.randn(256).astype("float32")
            mean /= (np.linalg.norm(mean) + 1e-8)
            wins = rng.randn(max(len(win_lists), 1), 256).astype("float32")
            wins /= (np.linalg.norm(wins, axis=1, keepdims=True) + 1e-8)
            return ToolResult(
                success=True,
                data={"mean_feature": mean, "window_features": wins},
                meta={"dim": 256, "n_windows": len(wins), "stub": True},
            )

        backend = _OpenGaitBackend.get(self.gait_project_dir, cfg_path=self.cfg_path)
        feats = []
        for win in win_lists:
            f = backend.encode(win)
            if f is not None:
                feats.append(f)
        if not feats:
            return ToolResult(success=False, error="all windows produced empty features")

        feats = np.stack(feats).astype("float32")
        norms = np.clip(np.linalg.norm(feats, axis=1, keepdims=True), 1e-12, None)
        feats = feats / norms
        mean = feats.mean(axis=0)
        mean /= (np.linalg.norm(mean) + 1e-8)
        return ToolResult(
            success=True,
            data={"mean_feature": mean, "window_features": feats},
            meta={"dim": feats.shape[1], "n_windows": feats.shape[0], "stub": False},
        )
