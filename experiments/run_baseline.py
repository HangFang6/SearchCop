"""
Baseline runner: classical Gait+ReID fusion, NO LLM agent.

Pipeline per query:
  1. reid_encode(query crop or short crop list)  -> mean_feature
  2. faiss_search(modality='reid', q=mean)        -> coarse Top-50
  3. (optional) gait_encode + faiss_search(gait)  -> coarse Top-50
  4. evidence_fuse(reid_rank, gait_rank)          -> final Top-K

This is the baseline against which SearchCop's `agent + tools` is compared.
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from tools import build_default_registry
from data import QueryRecord
from experiments.runner import run_experiment


def make_baseline_predictor(stub: bool = False, top_k: int = 10):
    reg = build_default_registry(stub=stub)
    reid = reg.get("reid_encode")
    faiss_t = reg.get("faiss_search")
    fuse = reg.get("evidence_fuse")
    gait = reg.get("gait_encode")

    def predictor(q: QueryRecord):
        # Try reid path
        rid = reid.run(crop_paths=[q.image_path])
        if not rid.success:
            # nothing to match against
            return []
        reid_q = rid.data["mean_feature"]
        rid_search = faiss_t.run(modality="reid", query_vec=reid_q, top_k=50)
        reid_list = rid_search.data.get("candidates", []) if rid_search.success else []

        # Try gait path if a silhouette dir was hinted in text_desc — kept simple here
        gait_list = []
        sil_dir = q.text_desc if q.text_desc and os.path.isdir(q.text_desc) else None
        if sil_dir:
            ge = gait.run(silhouette_dir=sil_dir)
            if ge.success:
                gs = faiss_t.run(modality="gait", query_vec=ge.data["mean_feature"], top_k=50)
                if gs.success:
                    gait_list = gs.data.get("candidates", [])

        # Fuse via RRF
        lists = [l for l in (reid_list, gait_list) if l]
        if not lists:
            return []
        out = fuse.run(ranked_lists=lists, strategy="rrf", top_k=top_k)
        return out.data["ranked"] if out.success else reid_list[:top_k]

    return predictor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--stub", action="store_true",
                    help="Force stub mode (no GPU / no FAISS / mock data).")
    args = ap.parse_args()

    if args.stub:
        os.environ["SEARCHCOP_STUB"] = "1"

    predictor = make_baseline_predictor(stub=args.stub)
    run_experiment(args.config, predictor, args.output_dir)


if __name__ == "__main__":
    main()
