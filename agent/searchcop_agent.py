"""
SearchCop Agent main loop — Planner / Executor / Critic.

Pseudocode:
    plan = planner(query, camera_topology, time_window)
    while not done and steps < MAX_STEPS:
        tool_call = planner.decide_next(history)
        result   = executor.run(tool_call)
        verdict  = critic.evaluate(result, history)
        history.append((tool_call, result, verdict))
        if verdict.confidence > THRESH: break
        if verdict.needs_correction: plan = planner.replan(history)
    return top_k(history)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .llm_client import LLMClient
from tools import ToolRegistry, build_default_registry


@dataclass
class AgentStep:
    tool_name: str
    tool_args: Dict[str, Any]
    tool_result: Any = None
    critic_verdict: Optional[Dict] = None


@dataclass
class AgentTrace:
    query: Dict
    steps: List[AgentStep] = field(default_factory=list)
    top_k: List[Dict] = field(default_factory=list)
    reasoning: str = ""


class SearchCopAgent:
    def __init__(
        self,
        llm: Optional[LLMClient] = None,
        registry: Optional[ToolRegistry] = None,
        max_steps: int = 8,
        confidence_threshold: float = 0.85,
    ):
        self.llm = llm or LLMClient(mock=True)
        self.registry = registry or build_default_registry()
        self.max_steps = max_steps
        self.confidence_threshold = confidence_threshold

    # ----- Entry -----
    def search(self, query: Dict, camera_topology: Optional[Dict] = None) -> AgentTrace:
        trace = AgentTrace(query=query)
        for _ in range(self.max_steps):
            tool_call = self._plan_next_step(query, trace, camera_topology)
            if tool_call is None:
                break
            step = self._execute(tool_call)
            verdict = self._criticize(step, trace)
            step.critic_verdict = verdict
            trace.steps.append(step)
            if verdict.get("confidence", 0.0) >= self.confidence_threshold:
                break
        trace.top_k = self._aggregate(trace)
        trace.reasoning = self._explain(trace)
        return trace

    # ----- Internal stubs (to be implemented by Coder Agent) -----
    def _plan_next_step(self, query, trace, topo) -> Optional[Dict]:
        # TODO: prompt LLM with available tool schemas + history, ask which tool to call next.
        return None

    def _execute(self, tool_call: Dict) -> AgentStep:
        tool = self.registry.get(tool_call["tool_name"])
        result = tool.run(**tool_call.get("tool_args", {}))
        return AgentStep(tool_name=tool_call["tool_name"],
                         tool_args=tool_call.get("tool_args", {}),
                         tool_result=result)

    def _criticize(self, step: AgentStep, trace: AgentTrace) -> Dict:
        # TODO: ask LLM critic to score the step's usefulness and overall confidence.
        return {"confidence": 0.0, "needs_correction": False}

    def _aggregate(self, trace: AgentTrace) -> List[Dict]:
        # TODO: fuse multi-source evidence into a final Top-K list.
        return []

    def _explain(self, trace: AgentTrace) -> str:
        # TODO: render an auditable reasoning chain for the paper figures.
        return ""
