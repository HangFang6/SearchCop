# SearchCop · LLM Agent for Multi-Camera Person Search

> Target venue: **CVPR 2027 / ICCV 2027 / NeurIPS 2026**
> Author: [Your Name], PhD Candidate
> Start date: 2026-05-18 · Sprint: 2 weeks (D-Day = 2026-05-31)

---

## 1. What is SearchCop?

**SearchCop** reformulates multi-camera person search from a *single-shot feature-matching* problem into a *multi-step tool-using decision* problem driven by an LLM Agent (GPT-4o).

Given a query (image / text description / short clip) and a multi-camera video gallery, SearchCop autonomously decides:

- **Which camera to inspect first** (spatio-temporal prior reasoning)
- **Which encoder to invoke** (Gait when full-body / ReID when frontal / VLM-caption when occluded)
- **When to fall back, expand, or self-correct** (low confidence triggers re-search)
- **How to fuse multi-source evidence** into a Top-K list with an auditable reasoning chain

## 2. Five core contributions

1. **Task reformulation** — Person Search as Tool-using Agent (not a one-shot retrieval)
2. **Tool Library** — 8-10 callable tools wrapping Gait / ReID / VLM / Spatio-Temporal Filter / Quality Scorer
3. **Reasoning Chain** — CoT-style auditable retrieval decision trace
4. **Spatio-Temporal Reasoning** — Camera topology + time window prior baked into the prompt
5. **Self-Correcting Loop** — Confidence-aware re-search and tool switching

## 3. Repository layout

```
SearchCop/
├── agent/              # LLM Agent main loop (planner / executor / critic)
├── tools/              # Tool wrappers (gait_tool / reid_tool / vlm_tool / ...)
├── data/               # Dataset loaders (MEVID / PRCC / LTCC / MARS / DukeMTMC-VideoReID)
├── eval/               # Person-mAP / Reasoning-Faithfulness / Tool-call-Efficiency
├── experiments/        # configs/ + scripts/ for each experiment
├── paper/              # LaTeX + figures + sections
├── proposal/           # Research Proposal (15-20 pages)
├── docs/               # Architecture diagrams, design notes
├── scripts/            # setup_remote.sh, run_exp_*.sh
├── results/            # Experiment outputs (gitignored except summary tables)
├── logs/               # Run logs (gitignored)
├── cache/              # GPT-4o response cache (gitignored)
├── tests/              # Unit tests (mock LLM + mock GPU)
├── environment.yml     # Remote conda environment spec
├── requirements.txt    # Pip-only fallback
├── README.md           # This file
└── README_REMOTE.md    # Remote H20 deployment guide
```

## 4. Workflow: local dev + remote exec

| Stage | Where | What |
|---|---|---|
| Design / write / unit-test (mock) | **Local (this machine)** | Code, paper, mock-LLM tests |
| Real inference / training / large-scale eval | **Remote H20 server** | `git pull && bash scripts/run_exp_xxx.sh` |
| Result review | **Local** | Pull `results/` summary, write paper |

See `README_REMOTE.md` for the remote setup playbook.

## 5. Quickstart (local, mock mode)

```bash
conda env create -f environment.yml
conda activate searchcop
pytest tests/ -v          # all tests use mock LLM, runnable on CPU
```

## 6. Quickstart (remote H20)

```bash
ssh user@h20-server
git clone <private-repo-url> SearchCop && cd SearchCop
bash scripts/setup_remote.sh
export OPENAI_API_KEY=sk-...
bash scripts/run_exp_mevid_baseline.sh
```

## 7. Status (Day-by-Day)

See `docs/sprint_log.md` (mirror of `fh_gait/工作记录.md`).
