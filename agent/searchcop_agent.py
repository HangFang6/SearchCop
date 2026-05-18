"""
SearchCop Agent — Planner / Executor / Critic / Aggregator loop, real Doubao calls.

Loop:
    while not stop and steps < max_steps:
        plan   = planner(query, history)        # tier=pro
        if plan.stop: break
        result = executor(plan.tool_call)
        verdict= critic(result, history)        # tier=pro
        history.append((plan, result, verdict))
        if verdict.confidence >= threshold: break
        if same tool+args called >= dedup_limit: stop with reason
    final = aggregator(history)                  # tier=lite
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .llm_client import LLMClient
from .prompts import (
    build_planner_messages, build_critic_messages, build_aggregator_messages,
    parse_json_block,
)
from tools import ToolRegistry, build_default_registry


# ----- traces -----
@dataclass
class AgentStep:
    tool_name: str
    tool_args: Dict[str, Any]
    tool_result: Any = None
    tool_meta: Dict[str, Any] = field(default_factory=dict)
    critic_verdict: Optional[Dict] = None
    planner_thought: str = ""


@dataclass
class AgentTrace:
    query: Dict
    steps: List[AgentStep] = field(default_factory=list)
    candidates: Dict[int, Dict] = field(default_factory=dict)   # db_id -> evidence
    top_k: List[Dict] = field(default_factory=list)
    reasoning: str = ""
    stop_reason: str = ""

    def summary_text(self) -> str:
        if not self.steps:
            return "(empty)"
        lines = []
        for i, s in enumerate(self.steps, 1):
            v = s.critic_verdict or {}
            lines.append(
                f"[{i}] tool={s.tool_name} args={_short(s.tool_args)} "
                f"meta={_short(s.tool_meta)} "
                f"verdict={{useful={v.get('useful')},conf={v.get('confidence')}}}"
            )
        return "\n".join(lines)

    def candidates_summary(self, k: int = 20) -> str:
        if not self.candidates:
            return "(none)"
        ranked = sorted(self.candidates.items(),
                        key=lambda kv: kv[1].get("score", 0.0), reverse=True)[:k]
        return json.dumps(
            [{"db_id": d, "score": v.get("score", 0.0),
              "evidence": v.get("evidence", [])} for d, v in ranked],
            ensure_ascii=False,
        )


def _short(obj, n: int = 80) -> str:
    s = json.dumps(obj, ensure_ascii=False, default=str)
    return s if len(s) <= n else s[:n] + "..."


# ----- agent -----
class SearchCopAgent:
    def __init__(
        self,
        llm: Optional[LLMClient] = None,
        registry: Optional[ToolRegistry] = None,
        max_steps: int = 8,
        confidence_threshold: float = 0.85,
        dedup_limit: int = 2,
        planner_tier: str = "pro",
        critic_tier: str = "pro",
        aggregator_tier: str = "lite",
    ):
        self.llm = llm or LLMClient(mock=True)
        self.registry = registry or build_default_registry()
        self.max_steps = max_steps
        self.confidence_threshold = confidence_threshold
        self.dedup_limit = dedup_limit
        self.planner_tier = planner_tier
        self.critic_tier = critic_tier
        self.aggregator_tier = aggregator_tier

    # -------------- entry --------------
    def search(self, query: Dict, camera_topology: Optional[Dict] = None) -> AgentTrace:
        trace = AgentTrace(query=query)
        topo = json.dumps(camera_topology, ensure_ascii=False) if camera_topology else ""
        seen_calls: Dict[str, int] = {}

        for step_idx in range(self.max_steps):
            plan = self._plan_next_step(query, trace, topo)
            if plan is None or plan.get("stop"):
                trace.stop_reason = "planner-stop"
                break

            tc = plan.get("tool_call") or {}
            name = tc.get("name") or tc.get("tool_name")
            args = tc.get("arguments") or tc.get("tool_args") or {}
            if not name or name not in self.registry.names():
                trace.stop_reason = f"bad tool name: {name}"
                break

            sig = name + "::" + json.dumps(args, sort_keys=True, default=str)
            seen_calls[sig] = seen_calls.get(sig, 0) + 1
            if seen_calls[sig] > self.dedup_limit:
                trace.stop_reason = f"dedup: {name} repeated"
                break

            step = self._execute(name, args)
            step.planner_thought = plan.get("thought", "")
            verdict = self._criticize(step, trace)
            step.critic_verdict = verdict
            trace.steps.append(step)

            self._merge_candidates(trace, step)

            if (verdict.get("confidence") or 0.0) >= self.confidence_threshold:
                trace.stop_reason = "confidence-threshold"
                break
        else:
            trace.stop_reason = trace.stop_reason or "max-steps-reached"

        trace.top_k, trace.reasoning = self._aggregate(trace)
        return trace

    # -------------- planner --------------
    def _plan_next_step(self, query: Dict, trace: AgentTrace, topo: str) -> Optional[Dict]:
        msgs = build_planner_messages(
            query=query,
            tool_schemas=self.registry.list_schemas(),
            trace_summary=trace.summary_text(),
            topology_summary=topo,
            confidence_threshold=self.confidence_threshold,
        )
        out = self.llm.chat(msgs, tier=self.planner_tier)
        parsed = parse_json_block(out)
        if parsed is None:
            return {"thought": "(unparsable planner output)", "tool_call": None, "stop": True}
        return parsed

    # -------------- executor --------------
    def _execute(self, name: str, args: Dict[str, Any]) -> AgentStep:
        tool = self.registry.get(name)
        result = tool.run(**args)
        # AgentStep stores both the typed result and a serializable summary
        return AgentStep(
            tool_name=name,
            tool_args=args,
            tool_result=result,
            tool_meta={"success": result.success, "error": result.error, **result.meta},
        )

    # -------------- critic --------------
    def _criticize(self, step: AgentStep, trace: AgentTrace) -> Dict:
        last = (
            f"tool={step.tool_name}\n"
            f"args={_short(step.tool_args)}\n"
            f"success={step.tool_meta.get('success')}\n"
            f"meta={_short(step.tool_meta)}\n"
            f"top_candidates={trace.candidates_summary(5)}"
        )
        msgs = build_critic_messages(
            last_step_summary=last,
            trace_summary=trace.summary_text(),
        )
        out = self.llm.chat(msgs, tier=self.critic_tier)
        parsed = parse_json_block(out) or {}
        return {
            "useful": bool(parsed.get("useful", False)),
            "confidence": float(parsed.get("confidence", 0.0) or 0.0),
            "needs_correction": bool(parsed.get("needs_correction", False)),
            "note": str(parsed.get("note", "")),
        }

    # -------------- aggregator --------------
    def _aggregate(self, trace: AgentTrace) -> tuple[List[Dict], str]:
        # Rank by accumulated score (may be empty)
        ranked = sorted(trace.candidates.items(),
                        key=lambda kv: kv[1].get("score", 0.0), reverse=True)[:10]
        seed_topk = [{"db_id": d, "score": v.get("score", 0.0),
                      "why": ", ".join(v.get("evidence", []))} for d, v in ranked]

        # Skip the LLM call entirely if there is literally nothing to summarize
        if not trace.steps:
            return seed_topk, ""

        msgs = build_aggregator_messages(
            trace_summary=trace.summary_text(),
            candidates_summary=trace.candidates_summary(20),
        )
        out = self.llm.chat(msgs, tier=self.aggregator_tier)
        parsed = parse_json_block(out)
        if parsed and isinstance(parsed.get("top_k"), list) and parsed["top_k"]:
            return parsed["top_k"], str(parsed.get("reasoning", ""))
        # Fallback: programmatic ranking + empty reasoning
        return seed_topk, "Programmatic fallback ranking by accumulated score."

    # -------------- evidence merge --------------
    @staticmethod
    def _merge_candidates(trace: AgentTrace, step: AgentStep) -> None:
        """If the step is a faiss_search, fold its candidates into trace.candidates."""
        if step.tool_name != "faiss_search":
            return
        result = step.tool_result
        if not getattr(result, "success", False):
            return
        modality = result.data.get("modality")
        for c in result.data.get("candidates", []):
            db_id = int(c["db_id"])
            score = float(c["score"])
            slot = trace.candidates.setdefault(db_id, {"score": 0.0, "evidence": []})
            slot["score"] = max(slot["score"], score)         # take best per modality
            tag = f"{modality}:{score:.3f}"
            if tag not in slot["evidence"]:
                slot["evidence"].append(tag)
