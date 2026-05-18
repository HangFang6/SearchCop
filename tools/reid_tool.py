"""
ReID Tool — CLIP-ReID 1280-d feature extraction over a list of crops.

Real implementation reuses the inference logic of
`Gait-Project/fix_fangh_2-2_crop2reid_26_0415.py`. Heavy modules
(torch, clip_reid) are lazily imported.

Output:
  mean_feature   (1280,)
  crop_features  (N, 1280)
  frame_ids      list[int]
"""
from __future__ import annotations

import os
import re
from glob import glob
from typing import List, Optional

import numpy as np

from .base import BaseTool, ToolResult


REID_FEAT_DIM = 1280
IMAGE_SIZE = (256, 128)  # H, W


# ---------------------------------------------------------------- backend
class _CLIPReidBackend:
    _instance = None

    def __init__(
        self,
        gait_project_dir: str,
        cfg_path: str = "clip_reid/configs/person/vit_clipreid.yml",
        weights: str = "clip_reid/MSMT17_clipreid_ViT-B-16_60.pth",
        num_class: int = 1041,
        camera_num: int = 8,
        view_num: int = 1,
    ):
        import sys
        if gait_project_dir not in sys.path:
            sys.path.insert(0, gait_project_dir)
        import torch                                              # noqa
        from clip_reid.config import cfg                          # noqa
        from clip_reid.model.make_model_clipreid import make_model  # noqa

        cfg.merge_from_file(os.path.join(gait_project_dir, cfg_path))
        cfg.freeze()
        m = make_model(cfg, num_class=num_class, camera_num=camera_num, view_num=view_num)
        m.load_param(os.path.join(gait_project_dir, weights))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        m.to(device).eval()
        self.model = m
        self.device = device
        self._torch = torch

    @classmethod
    def get(cls, gait_project_dir: str, **kw) -> "_CLIPReidBackend":
        if cls._instance is None:
            cls._instance = cls(gait_project_dir, **kw)
        return cls._instance

    def encode(self, crop_paths: List[str], batch_size: int = 64) -> Optional[np.ndarray]:
        import cv2
        torch = self._torch
        tensors = []
        for p in crop_paths:
            img = cv2.imread(p)
            if img is None:
                continue
            tensors.append(_bgr_to_tensor(img, torch))
        if not tensors:
            return None
        feats = []
        for i in range(0, len(tensors), batch_size):
            batch = torch.stack(tensors[i:i + batch_size], dim=0).to(self.device)
            with torch.no_grad():
                feats.append(self.model(batch).cpu().numpy())
        return np.concatenate(feats, axis=0).astype("float32")


def _bgr_to_tensor(image_bgr, torch):
    import cv2
    h, w = IMAGE_SIZE
    resized = cv2.resize(image_bgr, (w, h))
    rgb = resized[:, :, ::-1].astype(np.float32) / 255.0
    rgb = (rgb - 0.5) / 0.5
    return torch.from_numpy(rgb.transpose(2, 0, 1).copy())


# ---------------------------------------------------------------- helper
def _parse_frame_id(filename: str) -> Optional[int]:
    m = re.search(r"(\d+)", os.path.basename(filename))
    return int(m.group(1)) if m else None


def list_crops(crop_input) -> List[str]:
    """Accept a directory or an explicit list of file paths."""
    if isinstance(crop_input, list):
        return [p for p in crop_input if os.path.isfile(p)]
    if os.path.isdir(crop_input):
        files = sorted(glob(os.path.join(crop_input, "*.jpg"))
                       + glob(os.path.join(crop_input, "*.png")))
        return files
    if os.path.isfile(crop_input):
        return [crop_input]
    return []


# ---------------------------------------------------------------- tool
class ReidEncodeTool(BaseTool):
    name = "reid_encode"
    description = (
        "Encode N person crops (BGR jpgs, any size, will be resized to 256x128) "
        "into a 1280-d CLIP-ReID feature plus per-crop features for multi-vector "
        "rerank. Pass either a directory or a list of file paths."
    )

    @property
    def json_schema(self):
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "crop_paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of jpg paths.",
                    },
                    "crop_dir": {
                        "type": "string",
                        "description": "Directory containing the crops (alternative to crop_paths).",
                    },
                },
            },
        }

    def __init__(
        self,
        gait_project_dir: Optional[str] = None,
        stub: Optional[bool] = None,
    ):
        if gait_project_dir is None:
            here = os.path.dirname(os.path.abspath(__file__))
            gait_project_dir = os.path.normpath(os.path.join(here, "..", "..", "Gait-Project"))
        self.gait_project_dir = gait_project_dir
        if stub is None:
            stub = os.getenv("SEARCHCOP_STUB", "0") == "1" or not os.path.isdir(gait_project_dir)
        self.stub = stub

    def run(self,
            crop_paths: Optional[List[str]] = None,
            crop_dir: Optional[str] = None) -> ToolResult:
        # In stub mode trust any non-empty list/string the caller passes,
        # so we don't need real files on local dev machines.
        if self.stub:
            if crop_paths and isinstance(crop_paths, list):
                paths = list(crop_paths)
            elif crop_dir:
                paths = [os.path.join(crop_dir, f"stub_{i}.jpg") for i in range(5)]
            else:
                return ToolResult(success=False, error="no crops provided")
            frame_ids = [_parse_frame_id(p) for p in paths]
            rng = np.random.RandomState(abs(hash(tuple(paths))) % (2**32))
            crops = rng.randn(len(paths), REID_FEAT_DIM).astype("float32")
            crops /= (np.linalg.norm(crops, axis=1, keepdims=True) + 1e-8)
            mean = crops.mean(axis=0)
            mean /= (np.linalg.norm(mean) + 1e-8)
            return ToolResult(
                success=True,
                data={"mean_feature": mean, "crop_features": crops, "frame_ids": frame_ids},
                meta={"dim": REID_FEAT_DIM, "n_crops": len(paths), "stub": True},
            )

        # Real mode: must read files from disk
        paths: List[str] = []
        if crop_paths:
            paths = list_crops(crop_paths)
        elif crop_dir:
            paths = list_crops(crop_dir)
        if not paths:
            return ToolResult(success=False, error="no crops provided")
        frame_ids = [_parse_frame_id(p) for p in paths]

        backend = _CLIPReidBackend.get(self.gait_project_dir)
        feats = backend.encode(paths)
        if feats is None or feats.shape[0] == 0:
            return ToolResult(success=False, error="encoder returned empty features")
        # L2 normalize each row
        feats /= (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12)
        mean = feats.mean(axis=0)
        mean /= (np.linalg.norm(mean) + 1e-8)
        return ToolResult(
            success=True,
            data={"mean_feature": mean, "crop_features": feats, "frame_ids": frame_ids},
            meta={"dim": feats.shape[1], "n_crops": feats.shape[0], "stub": False},
        )
