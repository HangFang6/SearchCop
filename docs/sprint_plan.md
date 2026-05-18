# SearchCop · 2-Week Sprint Plan

Sprint window: **2026-05-18 (Mon) → 2026-05-31 (Sun)**
Sub-Agent team: Researcher · Architect · Coder · Experiment · Writer (parallel)

| Day | Date | Local (this machine) | Remote (H20) | Owner |
|----|------|----------------------|--------------|-------|
| D1 | 05-18 Mon | Project skeleton, env files, docs, deployment guide | (prep) provision H20, clone repo, run setup_remote.sh | Architect+Coder / User |
| D2 | 05-19 Tue | Tool wrappers (gait/reid real impl), planner prompt v1 | Download MEVID + MARS + DukeMTMC datasets | Coder / User |
| D3 | 05-20 Wed | faiss_search + multivector_rerank port from fix_fangh_5 | Build FAISS indexes for MEVID; smoke run baseline | Coder / Experiment |
| D4 | 05-21 Thu | vlm_caption + spatio_temporal_filter; critic prompt v1 | Download PRCC + LTCC; convert to common format | Coder / Experiment |
| D5 | 05-22 Fri | evidence_fuse + agent loop wired end-to-end | Pilot: 50 MEVID queries through full pipeline | Coder / Experiment |
| D6 | 05-23 Sat | Eval scripts (Person-mAP, Faithfulness, Tool-call-Eff) | Run 4 baselines × 4 datasets (matrix start) | Experiment |
| D7 | 05-24 Sun | Bug fixes from D6 results; refine prompts | Continue baseline matrix | Experiment |
| D8 | 05-25 Mon | Ablation scripts (tool-leave-one-out / CoT off / self-correct off) | Run SearchCop full on MEVID + PRCC | Experiment |
| D9 | 05-26 Tue | Visualization (reasoning-chain figure, qualitative cases) | Run ablations; in-the-wild on自有数据 | Experiment |
| D10| 05-27 Wed | Paper sec 1 Intro + sec 2 Related Work (Writer) | Re-run any flaky experiment | Writer |
| D11| 05-28 Thu | Paper sec 3 Method (Architect feeds Writer) | Aggregate results to xlsx | Writer / Experiment |
| D12| 05-29 Fri | Paper sec 4 Experiments + main table + main figure | Final ablation re-runs if needed | Writer / Experiment |
| D13| 05-30 Sat | Proposal (15-20p) finalize; Paper draft full pass | (idle / standby) | Writer |
| D14| 05-31 Sun | Demo notebook + demo video; submit to advisor | (idle) | Writer / User |

**Daily ritual**: append a line to `D:\program\All-in-One-Gait-main\fh_gait\工作记录.md`.

**Risk hotspots**:
- D2-D3: dataset download bandwidth — if MEVID is slow, parallel-fetch PRCC/LTCC first.
- D5: first end-to-end run will reveal prompt fragility — budget half a day for prompt tuning.
- D8-D9: GPU contention on 100×H20 — pre-book slots if cluster shared.

**Cost guardrails**:
- All GPT-4o calls go through `diskcache` (`agent/llm_client.py`).
- Hard cap per experiment: 5000 LLM calls (set in `experiments/configs/*.yaml::agent.max_calls`).
