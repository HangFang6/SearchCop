# SearchCop · Architecture & Design

> Author: Architect Agent
> Status: Draft v0.1 (Day 1 of 2-week sprint)
> Purpose: single source of truth for the system design that Coder Agent, Experiment Agent, and Writer Agent all consume.

---

## 1. System overview

```
                ┌───────────────────────────────────────────────────────┐
                │                       USER QUERY                      │
                │  (image | text desc | short clip)  +  cam_id, t_start │
                └─────────────────────────┬─────────────────────────────┘
                                          │
                                          ▼
                          ┌────────────────────────────┐
                          │      PROMPT BUILDER        │
                          │  query → structured spec   │
                          │  + camera topology         │
                          │  + time window prior       │
                          └─────────────────────────┬──┘
                                                    │
                          ┌─────────────────────────▼─────────────────────────┐
                          │                 PLANNER (LLM)                     │
                          │  pick next tool with rationale (function-calling) │
                          └───────────┬──────────────────────────────┬────────┘
                                      │                              │ replan
                                      ▼                              ▲
                          ┌─────────────────────┐                    │
                          │      EXECUTOR       │                    │
                          │  invoke selected    │                    │
                          │  tool with args     │                    │
                          └──────────┬──────────┘                    │
                                     │                               │
                                     ▼                               │
                          ┌─────────────────────┐                    │
                          │     TOOL LIBRARY    │                    │
                          │  (see §3)           │                    │
                          └──────────┬──────────┘                    │
                                     │ result                        │
                                     ▼                               │
                          ┌─────────────────────┐                    │
                          │     CRITIC (LLM)    │── needs_correction ┘
                          │  score + confidence │
                          └──────────┬──────────┘
                                     │ confidence ≥ τ
                                     ▼
                          ┌─────────────────────┐
                          │     AGGREGATOR      │
                          │  fuse evidence,     │
                          │  emit Top-K + chain │
                          └─────────────────────┘
```

**Three LLM roles, one agent process:**
- **Planner** — sees query + history + tool schemas → emits a function call.
- **Executor** — pure Python, runs the selected tool, returns `ToolResult`.
- **Critic** — sees the executor's result + history → outputs `{confidence, useful, needs_correction}`.

---

## 2. State object

A single `AgentTrace` is passed through the loop:

```python
AgentTrace = {
  "query":   {image|text|clip, cam_id, t_start, t_end},
  "topology": {cam_id: {neighbors, transit_seconds_p50_p90}},
  "steps":   [AgentStep, ...],
  "candidates": {db_id: {evidence: [...], score_partial: {gait, reid, vlm}}},
  "top_k":   [{db_id, score, why}, ...],
  "reasoning": "audit-ready CoT string",
}

AgentStep = {
  "tool_name": str,
  "tool_args": dict,
  "tool_result": ToolResult,
  "critic_verdict": {confidence: float, useful: bool, needs_correction: bool, note: str},
}
```

---

## 3. Tool Library (8-10 tools)

| # | Tool name | Wraps | When the LLM should call it |
|---|---|---|---|
| 1 | `gait_encode` | OpenGait BaselineDemo (multi-window 30/45/60) | Full body visible across ≥30 consecutive frames |
| 2 | `reid_encode` | CLIP-ReID 1280-d | Only a handful of good crops (occlusion / short clip) |
| 3 | `vlm_caption` | GPT-4o vision / InternVL | Need a text description ("man, red shirt, backpack") |
| 4 | `quality_score` | Laplacian-var + occlusion mask | Decide whether to skip a track segment |
| 5 | `spatio_temporal_filter` | Topology + time-window prior | Prune candidates that can't physically be at cam_j by t_j |
| 6 | `faiss_search` | gait / reid FAISS index | Coarse Top-50 retrieval over the gallery |
| 7 | `multivector_rerank` | All-to-all multi-window/multi-crop rerank | After coarse Top-50 → refined Top-K |
| 8 | `track_extract` | YOLOX + BYTETracker | When the query is a raw video segment, not a feature |
| 9 | `sam_silhouette` | SAM ViT-L → 64×64 silhouette | Required before `gait_encode` |
| 10 | `evidence_fuse` | Weighted-rank fusion (Borda / RRF) | Final aggregation step |

Each tool follows the schema in `tools/base.py::BaseTool.json_schema` and is exposed
to the LLM via OpenAI function-calling.

### 3.1 Tool dependency graph

```
  raw video ─► track_extract ─► sam_silhouette ─► gait_encode ─┐
                              ─► (crops) ──────► reid_encode ──┤
                              ─► (crops) ──────► vlm_caption ──┤
                                                               ├──► evidence_fuse ──► Top-K
   gallery ──► faiss_search ─► multivector_rerank ─────────────┤
                                                               │
   topology ─► spatio_temporal_filter ─────────────────────────┘
```

---

## 4. Reasoning Chain spec (CoT prompt template)

```text
You are SearchCop, an autonomous multi-camera person-search planner.

GOAL: find the same person as the query across the camera network.

CONTEXT:
  - Query: {{query_spec}}
  - Camera topology: {{topology_summary}}
  - Time window: [{{t_start}}, {{t_end}}]
  - Tools available: {{tool_schemas_json}}
  - History so far: {{trace_summary}}

INSTRUCTIONS:
  1. Think step by step. State which evidence is missing.
  2. Choose ONE tool to call next. Justify in <=2 sentences.
  3. Output strict JSON: { "thought": "...", "tool_call": { "name": "...", "arguments": {...} } }
  4. If you believe Top-K is already reliable (confidence > {{thresh}}), return
     { "thought": "...", "tool_call": null, "stop": true }
```

The Critic uses a sibling prompt that scores the just-finished step on
`{useful: bool, confidence_delta: float, needs_correction: bool, note: str}`.

---

## 5. Self-correcting loop

Triggered when ANY of:
- Critic returns `needs_correction = true`
- Top-1 confidence drops by ≥0.15 after fusion
- Same tool called 3× consecutively with no Top-K change

Action options the LLM can pick from (encoded as additional pseudo-tools):
- `expand_time_window(delta_seconds)`
- `expand_cameras(neighbors_k)`
- `switch_encoder(from→to)`        # e.g. gait→reid when motion is poor
- `request_human_in_the_loop()`     # optional; off in CVPR submission

---

## 6. Data flow with the existing Gait-Project

```
Gait-Project (already running)            SearchCop (this repo)
─────────────────────────────────────     ─────────────────────────────────────
fix_fangh_1 (track + crop)         ◄──── reused as `track_extract` tool
fix_fangh_2-1 (SAM silhouette)     ◄──── reused as `sam_silhouette` tool
fix_fangh_2-2 (CLIP-ReID feat)     ◄──── reused as `reid_encode` tool
fix_fangh_3 (OpenGait feat)        ◄──── reused as `gait_encode` tool
fix_fangh_5 (ReID search engine)   ◄──── reused as `faiss_search`+`multivector_rerank`
fix_fangh_8 (FastAPI)              ◄──── optional: SearchCop calls REST instead of in-proc

NEW in SearchCop:
  + LLM Planner / Executor / Critic
  + Spatio-temporal reasoner
  + VLM captioner
  + Evidence fuser
  + Reasoning-chain emitter
```

---

## 7. Metrics (paper-ready)

| Metric | Definition | Goal |
|---|---|---|
| **Person-mAP** | Person-level mean Average Precision across queries | Headline number |
| **CMC@1 / @5 / @10** | Rank hit rate | Standard ReID metric |
| **Reasoning Faithfulness** | Fraction of steps whose rationale matches the tool result | Novel, supports interpretability claim |
| **Tool-call Efficiency** | Avg #tool-calls per correct query | Novel, supports efficiency claim |
| **Cross-camera Recall@K** | Recall on queries that span ≥2 cameras | Highlights Multi-Camera contribution |

---

## 8. Implementation order (for Coder Agent)

1. `tools/base.py` ✅ (done)
2. `agent/llm_client.py` ✅ (done)
3. `agent/searchcop_agent.py` skeleton ✅ (done; planner/critic still stubs)
4. **Next**: implement real `gait_encode` + `reid_encode` by importing the
   existing `fix_fangh_*` modules and exposing their inference functions.
5. Implement `faiss_search` + `multivector_rerank` (port `fix_fangh_5` cleanly).
6. Implement `vlm_caption` (GPT-4o vision call).
7. Implement `spatio_temporal_filter` (camera topology JSON + simple geometry).
8. Wire `_plan_next_step` and `_criticize` to real GPT-4o function-calling.
9. Implement `evidence_fuse` and `_aggregate`.
10. End-to-end smoke run on 10 MEVID queries.
