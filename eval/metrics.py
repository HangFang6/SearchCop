"""
SearchCop evaluation metrics:
  - person_mAP            (person-level mean Average Precision)
  - cmc_at_k              (rank-1 / rank-5 / rank-10)
  - reasoning_faithfulness  (LLM-as-judge or rule-based on tool-result alignment)
  - tool_call_efficiency  (tools per correct answer)
"""
from __future__ import annotations

from typing import List, Dict
import numpy as np


def person_map(predictions: List[Dict], ground_truth: Dict) -> float:
    """Standard mAP aggregated at the person level.
    predictions: [{query_id, ranked_db_ids: [...]}, ...]
    ground_truth: {query_id: set_of_positive_db_ids}
    """
    aps = []
    for pred in predictions:
        qid = pred["query_id"]
        ranked = pred["ranked_db_ids"]
        positives = ground_truth.get(qid, set())
        if not positives:
            continue
        hits, score = 0, 0.0
        for i, db_id in enumerate(ranked, 1):
            if db_id in positives:
                hits += 1
                score += hits / i
        aps.append(score / max(len(positives), 1))
    return float(np.mean(aps)) if aps else 0.0


def cmc_at_k(predictions: List[Dict], ground_truth: Dict, ks=(1, 5, 10)) -> Dict[int, float]:
    out = {k: 0.0 for k in ks}
    n = 0
    for pred in predictions:
        positives = ground_truth.get(pred["query_id"], set())
        if not positives:
            continue
        n += 1
        for k in ks:
            if any(d in positives for d in pred["ranked_db_ids"][:k]):
                out[k] += 1
    return {k: (v / n if n else 0.0) for k, v in out.items()}


def tool_call_efficiency(traces, correct_flags) -> float:
    """Avg #tool-calls per correctly answered query (lower is better)."""
    correct_calls = [len(t.steps) for t, ok in zip(traces, correct_flags) if ok]
    return float(np.mean(correct_calls)) if correct_calls else 0.0


def reasoning_faithfulness(traces, judge=None) -> float:
    """Placeholder: ratio of steps whose stated rationale matches the actual tool output.
    Real impl can use an LLM judge or rule-based string overlap.
    """
    if not traces:
        return 0.0
    # TODO(Experiment Agent): implement either LLM-judge or programmatic check
    return float(np.mean([0.0 for _ in traces]))
