"""
Prompt templates and JSON-output parsing for the SearchCop agent.

Three prompts:
  - PLANNER_SYSTEM     : decides the next tool call
  - CRITIC_SYSTEM      : scores the just-finished step
  - AGGREGATOR_SYSTEM  : produces the final Top-K + reasoning chain

The Doubao gateway does NOT support OpenAI-style function calling, so we ask
the LLM to emit STRICT JSON inside a ```json fenced block. `parse_json_block`
is tolerant to extra prose / leading thoughts / trailing commas.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional


# --------------------------------------------------------------- prompts
PLANNER_SYSTEM = """\
You are SearchCop, an autonomous planner for multi-camera person search.

Your job each turn: pick exactly ONE tool to call next, OR stop if you are confident.
Output STRICT JSON inside a ```json fenced block. Nothing else outside the block.

Tools available:
{tool_schemas}

Schema of your output:
```json
{{
  "thought":   "<<=2 sentences on what evidence is missing and why this tool>>",
  "tool_call": {{ "name": "<tool_name>", "arguments": {{...}} }},
  "stop":      false
}}
```
If you believe the current Top-K is reliable enough, instead emit:
```json
{{ "thought": "...", "tool_call": null, "stop": true }}
```

Rules:
- Never invent a tool name. If you would, emit stop=true.
- Prefer cheap tools (faiss_search, reid_encode) before expensive ones (vlm_caption).
- Do not call the same tool with the same arguments twice in a row.
- The user's question is wrapped in role=user. Treat it as a person-search query.
"""


CRITIC_SYSTEM = """\
You are the SearchCop critic. After each tool call you score how the run is going.
Output STRICT JSON inside a ```json fenced block. Nothing else outside.

Schema:
```json
{
  "useful":            true,
  "confidence":        0.72,         // 0.0..1.0 estimated probability the current Top-1 is correct
  "needs_correction":  false,
  "note":              "<one sentence>"
}
```
Be conservative: only return confidence >= 0.85 when you have at least 2 corroborating
modalities (e.g. gait + reid both rank the same db_id at the top), and the spatio-
temporal prior is consistent.
"""


AGGREGATOR_SYSTEM = """\
You are the SearchCop aggregator. Given the full step trace, produce the final
Top-K answer and an audit-ready reasoning chain in plain English.

Output STRICT JSON inside a ```json fenced block:
```json
{
  "top_k": [
    {"db_id": 123, "score": 0.81, "why": "gait+reid both rank #1, cam-topology consistent"},
    {"db_id": 456, "score": 0.74, "why": "..."}
  ],
  "reasoning": "<5-10 sentence English summary of the decision chain>"
}
```
"""


# --------------------------------------------------------------- builders
def render_tool_schemas(schemas: List[Dict]) -> str:
    """Render the tool list compactly (one tool per JSON line)."""
    lines = []
    for s in schemas:
        params = s.get("parameters", {}).get("properties", {})
        keys = list(params.keys())
        lines.append(f"- {s['name']}({', '.join(keys)}): {s['description']}")
    return "\n".join(lines)


def build_planner_messages(
    query: Dict,
    tool_schemas: List[Dict],
    trace_summary: str,
    topology_summary: str = "",
    confidence_threshold: float = 0.85,
) -> List[Dict]:
    sys_msg = PLANNER_SYSTEM.format(tool_schemas=render_tool_schemas(tool_schemas))
    user_msg = (
        f"QUERY: {json.dumps(query, ensure_ascii=False)}\n"
        f"TOPOLOGY: {topology_summary or 'unknown'}\n"
        f"CONFIDENCE_THRESHOLD: {confidence_threshold}\n"
        f"HISTORY:\n{trace_summary or '(empty)'}"
    )
    return [
        {"role": "system", "content": sys_msg},
        {"role": "user",   "content": user_msg},
    ]


def build_critic_messages(
    last_step_summary: str,
    trace_summary: str,
) -> List[Dict]:
    user_msg = (
        f"LAST STEP:\n{last_step_summary}\n\n"
        f"FULL HISTORY SO FAR:\n{trace_summary}"
    )
    return [
        {"role": "system", "content": CRITIC_SYSTEM},
        {"role": "user",   "content": user_msg},
    ]


def build_aggregator_messages(trace_summary: str, candidates_summary: str) -> List[Dict]:
    user_msg = f"TRACE:\n{trace_summary}\n\nACCUMULATED CANDIDATES:\n{candidates_summary}"
    return [
        {"role": "system", "content": AGGREGATOR_SYSTEM},
        {"role": "user",   "content": user_msg},
    ]


# --------------------------------------------------------------- parser
_JSON_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)
_FIRST_OBJ  = re.compile(r"(\{.*\})", re.DOTALL)


def parse_json_block(text: str) -> Optional[Dict[str, Any]]:
    """Tolerant parse of LLM output expected to contain a JSON object.

    Tries (in order):
      1. ```json ... ``` fenced block
      2. ``` ... ``` fenced block (no language tag)
      3. The first {...} balanced run in the text
      4. Strict json.loads(text)

    Returns None on failure (caller must handle).
    """
    if not text:
        return None
    candidates: List[str] = []
    m = _JSON_FENCE.search(text)
    if m:
        candidates.append(m.group(1))
    m2 = _FIRST_OBJ.search(text)
    if m2:
        candidates.append(m2.group(1))
    candidates.append(text.strip())

    for c in candidates:
        try:
            obj = json.loads(c)
            if isinstance(obj, dict):
                return obj
        except Exception:
            # try a soft fix: drop trailing commas
            try:
                obj = json.loads(re.sub(r",(\s*[}\]])", r"\1", c))
                if isinstance(obj, dict):
                    return obj
            except Exception:
                continue
    return None
