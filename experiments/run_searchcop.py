"""
SearchCop runner: full LLM Agent + Tool Library pipeline.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Make repo root importable when invoked as a script
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from agent import LLMClient, SearchCopAgent
from tools import build_default_registry
from data import build_dataset, QueryRecord
from experiments.runner import run_experiment, load_config


def make_searchcop_predictor(cfg: dict, stub: bool = False, mock_llm: bool = False):
    """Build a (predictor, trace_collector) pair."""
    if mock_llm:
        llm = LLMClient(mock=True)
    else:
        llm = LLMClient.from_env()

    # Topology comes from the dataset
    ds_cfg = cfg["dataset"]
    ds = build_dataset(ds_cfg["name"], root=ds_cfg.get("root"),
                       split=ds_cfg.get("split", "test"))
    topology = ds.topology()

    registry = build_default_registry(stub=stub, llm=llm, topology=topology)
    agent_cfg = cfg.get("agent", {})

    agent = SearchCopAgent(
        llm=llm,
        registry=registry,
        max_steps=agent_cfg.get("max_steps", 8),
        confidence_threshold=agent_cfg.get("confidence_threshold", 0.85),
        planner_tier=agent_cfg.get("llm_planner_tier", "pro"),
        critic_tier=agent_cfg.get("llm_critic_tier", "pro"),
        aggregator_tier=agent_cfg.get("llm_aggregator_tier", "lite"),
    )

    traces = []

    def predictor(q: QueryRecord):
        trace = agent.search(q.as_agent_query(), camera_topology=topology)
        traces.append(trace)
        # Top-K is already a list of {db_id, score, why}
        return trace.top_k

    return predictor, traces


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--cache_dir", default="cache/")
    ap.add_argument("--stub", action="store_true",
                    help="Force tool stub mode (CI/local run).")
    ap.add_argument("--mock_llm", action="store_true",
                    help="Use deterministic mock LLM (fast smoke test).")
    args = ap.parse_args()

    if args.stub:
        os.environ["SEARCHCOP_STUB"] = "1"
    if args.cache_dir:
        os.environ["LLM_CACHE_DIR"] = args.cache_dir

    cfg = load_config(args.config)
    predictor, traces = make_searchcop_predictor(cfg, stub=args.stub, mock_llm=args.mock_llm)

    metrics = run_experiment(args.config, predictor, args.output_dir, trace_dump=traces)

    # Dump trace summaries for paper figures / qualitative analysis
    trace_path = os.path.join(args.output_dir, "traces.jsonl")
    with open(trace_path, "w", encoding="utf-8") as f:
        for t in traces:
            f.write(json.dumps({
                "query":       t.query,
                "stop_reason": t.stop_reason,
                "n_steps":     len(t.steps),
                "steps":       [{"tool": s.tool_name, "args_keys": list(s.tool_args.keys()),
                                  "thought": s.planner_thought,
                                  "verdict": s.critic_verdict,
                                  "meta": s.tool_meta} for s in t.steps],
                "top_k":       t.top_k,
                "reasoning":   t.reasoning,
            }, ensure_ascii=False) + "\n")
    print(f"[*] traces -> {trace_path}")


if __name__ == "__main__":
    main()
