"""
Experiment runner — shared between baseline and SearchCop.

Reads a YAML config, builds a dataset, evaluates a `predictor` (callable
that takes a QueryRecord and returns a list of {db_id, score, ...}),
computes metrics, and writes results to OUTPUT_DIR.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from typing import Callable, Dict, List

import yaml

from data import build_dataset, QueryRecord
from eval.metrics import person_map, cmc_at_k, tool_call_efficiency, reasoning_faithfulness


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_experiment(cfg_path: str,
                   predictor: Callable[[QueryRecord], List[Dict]],
                   output_dir: str,
                   trace_dump: List = None) -> Dict:
    cfg = load_config(cfg_path)
    ds_cfg = cfg["dataset"]
    ds = build_dataset(ds_cfg["name"], root=ds_cfg.get("root"),
                       split=ds_cfg.get("split", "test"))
    queries = ds.queries()
    gt      = ds.ground_truth()
    n_query_cap = ds_cfg.get("num_query") or len(queries)
    queries = queries[:n_query_cap]

    os.makedirs(output_dir, exist_ok=True)

    predictions: List[Dict] = []
    t_start = time.time()
    for i, q in enumerate(queries):
        ranked = predictor(q)
        predictions.append({
            "query_id":      q.query_id,
            "ranked_db_ids": [int(r["db_id"]) for r in ranked],
            "scores":        [float(r["score"]) for r in ranked],
        })
        if (i + 1) % max(1, len(queries) // 10) == 0:
            print(f"[{i+1}/{len(queries)}] {(i+1)*100/len(queries):.1f}% "
                  f"({(time.time()-t_start):.1f}s)")

    elapsed = time.time() - t_start

    # ----- metrics -----
    metrics = {
        "person_map":   person_map(predictions, gt),
        "cmc_at_k":     cmc_at_k(predictions, gt, ks=(1, 5, 10)),
        "n_queries":    len(queries),
        "elapsed_sec":  elapsed,
    }
    if trace_dump is not None:
        # ground-truth correctness flags for tool-call efficiency
        correct = []
        for pred in predictions:
            pos = gt.get(pred["query_id"], set())
            correct.append(bool(pred["ranked_db_ids"] and pred["ranked_db_ids"][0] in pos))
        metrics["tool_call_efficiency"] = tool_call_efficiency(trace_dump, correct)
        metrics["reasoning_faithfulness"] = reasoning_faithfulness(trace_dump)

    # ----- dump -----
    with open(os.path.join(output_dir, "predictions.jsonl"), "w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("\n===== METRICS =====")
    print(json.dumps(metrics, indent=2))
    return metrics
