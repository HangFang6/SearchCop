"""
MEVID dataset loader.

The real MEVID release (Davila et al., WACV 2023) ships with:
  - tracks/<pid>/<cam>/<seq>/*.jpg
  - meta/info.json   (cam topology, time stamps if available)
  - splits/{query,gallery}.csv

This loader exposes a uniform interface that any dataset can implement, so
that experiments don't depend on MEVID's exact layout:

    ds = build_dataset("MEVID", root="/data/MEVID")
    queries  = ds.queries()        # List[QueryRecord]
    galleries= ds.gallery()        # List[GalleryRecord]
    topo     = ds.topology()       # Dict
    gt       = ds.ground_truth()   # Dict[query_id -> set(positive_db_ids)]

A `MockDataset` is provided so unit tests + local dev can run without the
real download.
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


# ---------------------------------------------------------------- records
@dataclass
class QueryRecord:
    query_id:    str
    image_path:  str
    camera_id:   str
    time_seen:   float = 0.0      # epoch seconds (or normalized)
    text_desc:   str = ""         # optional natural-language description

    def as_agent_query(self) -> Dict:
        return {
            "query_id":  self.query_id,
            "image":     self.image_path,
            "cam_id":    self.camera_id,
            "t":         self.time_seen,
            "text":      self.text_desc or None,
        }


@dataclass
class GalleryRecord:
    db_id:        int
    person_id:    str
    image_paths:  List[str]
    camera_id:    str
    time_seen:    float = 0.0


# ---------------------------------------------------------------- abstract
class BaseDataset:
    name: str = "base"

    def queries(self) -> List[QueryRecord]:    raise NotImplementedError
    def gallery(self) -> List[GalleryRecord]:  raise NotImplementedError
    def topology(self) -> Dict:                return {}
    def ground_truth(self) -> Dict[str, Set[int]]:  raise NotImplementedError


# ---------------------------------------------------------------- MEVID
class MEVIDDataset(BaseDataset):
    name = "MEVID"

    def __init__(self, root: str, split: str = "test"):
        self.root = root
        self.split = split
        self._queries: Optional[List[QueryRecord]] = None
        self._gallery: Optional[List[GalleryRecord]] = None
        self._topo:    Optional[Dict] = None
        self._gt:      Optional[Dict[str, Set[int]]] = None

    # ----- file discovery -----
    def _split_csv(self, name: str) -> str:
        return os.path.join(self.root, "splits", f"{name}_{self.split}.csv")

    # ----- public API -----
    def queries(self) -> List[QueryRecord]:
        if self._queries is None:
            self._queries = self._load_queries()
        return self._queries

    def gallery(self) -> List[GalleryRecord]:
        if self._gallery is None:
            self._gallery = self._load_gallery()
        return self._gallery

    def topology(self) -> Dict:
        if self._topo is None:
            tpath = os.path.join(self.root, "meta", "topology.json")
            if os.path.isfile(tpath):
                with open(tpath, "r", encoding="utf-8") as f:
                    self._topo = json.load(f)
            else:
                self._topo = {}
        return self._topo

    def ground_truth(self) -> Dict[str, Set[int]]:
        if self._gt is None:
            self._gt = self._load_gt()
        return self._gt

    # ----- loaders (defensive: tolerate missing pieces) -----
    def _load_queries(self) -> List[QueryRecord]:
        csv = self._split_csv("query")
        if not os.path.isfile(csv):
            return []
        out = []
        with open(csv, "r", encoding="utf-8") as f:
            f.readline()  # header
            for line in f:
                parts = [x.strip() for x in line.strip().split(",")]
                if len(parts) < 4:
                    continue
                qid, rel_path, cam, t = parts[0], parts[1], parts[2], float(parts[3] or 0.0)
                out.append(QueryRecord(
                    query_id=qid,
                    image_path=os.path.join(self.root, rel_path),
                    camera_id=cam,
                    time_seen=t,
                ))
        return out

    def _load_gallery(self) -> List[GalleryRecord]:
        csv = self._split_csv("gallery")
        if not os.path.isfile(csv):
            return []
        out, db_id = [], 0
        with open(csv, "r", encoding="utf-8") as f:
            f.readline()
            for line in f:
                parts = [x.strip() for x in line.strip().split(",")]
                if len(parts) < 4:
                    continue
                pid, dir_rel, cam, t = parts[0], parts[1], parts[2], float(parts[3] or 0.0)
                seq_dir = os.path.join(self.root, dir_rel)
                if not os.path.isdir(seq_dir):
                    continue
                imgs = sorted(os.path.join(seq_dir, x) for x in os.listdir(seq_dir)
                              if x.lower().endswith((".jpg", ".png")))
                out.append(GalleryRecord(db_id=db_id, person_id=pid,
                                         image_paths=imgs, camera_id=cam, time_seen=t))
                db_id += 1
        return out

    def _load_gt(self) -> Dict[str, Set[int]]:
        gpath = os.path.join(self.root, "splits", f"gt_{self.split}.json")
        if not os.path.isfile(gpath):
            return {}
        with open(gpath, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return {qid: set(map(int, ids)) for qid, ids in raw.items()}


# ---------------------------------------------------------------- mock
@dataclass
class MockDataset(BaseDataset):
    """Tiny in-memory dataset for unit tests and local sanity runs."""
    n_queries: int = 5
    n_gallery: int = 50
    seed:      int = 0
    name:      str = "MOCK"
    _q: list = field(default_factory=list)
    _g: list = field(default_factory=list)
    _gt: dict = field(default_factory=dict)
    _topo: dict = field(default_factory=dict)

    def __post_init__(self):
        rng = random.Random(self.seed)
        cams = ["cam01", "cam02", "cam03", "cam04"]
        for i in range(self.n_queries):
            qid = f"q{i:03d}"
            cam = rng.choice(cams)
            t = float(rng.randint(0, 3600))
            self._q.append(QueryRecord(query_id=qid, image_path=f"/mock/q/{qid}.jpg",
                                       camera_id=cam, time_seen=t))
        for j in range(self.n_gallery):
            pid = f"p{j % max(1, self.n_queries):03d}"
            cam = rng.choice(cams)
            t = float(rng.randint(0, 3600))
            self._g.append(GalleryRecord(db_id=j, person_id=pid,
                                         image_paths=[f"/mock/g/{j}.jpg"],
                                         camera_id=cam, time_seen=t))
        # GT: same person_id is positive
        pid_to_ids: Dict[str, Set[int]] = {}
        for g in self._g:
            pid_to_ids.setdefault(g.person_id, set()).add(g.db_id)
        for q in self._q:
            pid = f"p{int(q.query_id[1:]) % max(1, self.n_queries):03d}"
            self._gt[q.query_id] = pid_to_ids.get(pid, set())
        # tiny topology
        self._topo = {c: {"neighbors": [x for x in cams if x != c],
                          "transit_seconds": {x: {"p50": 30, "p90": 60} for x in cams if x != c}}
                      for c in cams}

    def queries(self):       return list(self._q)
    def gallery(self):       return list(self._g)
    def topology(self):      return dict(self._topo)
    def ground_truth(self):  return {k: set(v) for k, v in self._gt.items()}


# ---------------------------------------------------------------- factory
def build_dataset(name: str, root: Optional[str] = None,
                  split: str = "test", **kw) -> BaseDataset:
    name_l = name.lower()
    if name_l in ("mock", "_mock", "mock_dataset"):
        return MockDataset(**kw)
    if name_l in ("mevid",):
        if not root:
            raise ValueError("MEVID requires root=...")
        return MEVIDDataset(root=root, split=split)
    raise ValueError(f"unknown dataset: {name}")
