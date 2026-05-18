# SearchCop · Related Work (Draft v0.1)

> Author: Researcher Agent
> Status: Draft outline — citations need verification on Day 2-3 with live arXiv search.
> Purpose: feed `paper/sections/02_related_work.tex` and the Proposal §3.

---

## 1. Multi-Camera Person Search & Re-Identification

### 1.1 Classical video ReID
- **MARS** (Zheng et al., ECCV 2016) — first large-scale video-based ReID benchmark.
- **DukeMTMC-VideoReID** (Wu et al., 2018) — multi-camera video ReID with tracklets.
- **AP3D / GRL / STMN** — temporal-aggregation methods (CNN + temporal pooling).
- **TransReID** (He et al., ICCV 2021) — Transformer for image ReID.
- **CLIP-ReID** (Li et al., AAAI 2023) — text-image alignment as ReID auxiliary.

> **Gap**: All are single-step retrievers. None model the multi-camera **decision process** (which camera to search first, when to switch modality).

### 1.2 Multi-camera person search
- **PRW** (Zheng et al., CVPR 2017), **CUHK-SYSU** (Xiao et al., CVPR 2017) — joint detection + ReID, but single camera or aggregated gallery.
- **MEVID** (Davila et al., WACV 2023) — multi-camera multi-environment video ReID. **Our main testbed.**
- **MTA** (Kohl et al., 2020) — synthetic multi-target multi-camera.

> **Gap**: Multi-camera benchmarks exist, but the standard pipeline still treats each gallery item independently — no agent-style planning, no cross-camera prior.

### 1.3 Clothing-change ReID (motivates Gait branch)
- **PRCC** (Yang et al., TPAMI 2019), **LTCC** (Qian et al., ACM MM 2020), **DeepChange** — same person, different outfits.
- **CAL** (Gu et al., CVPR 2022) — clothes-agnostic adversarial learning.

> **Gap**: Pure ReID degrades; Gait helps but is rarely combined dynamically — usually offline late fusion. SearchCop fuses on-demand via Agent decision.

### 1.4 Gait recognition
- **GaitSet** (Chao et al., AAAI 2019), **GaitPart** (Fan et al., CVPR 2020), **GaitBase / GaitGL / DeepGaitV2** (OpenGait family).
- **OpenGait** (Fan et al., CVPR 2023) — the framework we reuse.

> **Gap**: Gait works on silhouettes assuming clean walking sequences — fragile in real surveillance. An Agent can decide *when* gait is trustworthy.

---

## 2. LLM Agents & Tool Use

### 2.1 Agent foundations
- **ReAct** (Yao et al., ICLR 2023) — reasoning + acting interleaved.
- **Toolformer** (Schick et al., NeurIPS 2023) — self-supervised tool calling.
- **MM-ReAct** / **HuggingGPT** (Shen et al., NeurIPS 2023) — LLM orchestrates vision models.
- **Visual ChatGPT / Visual Programming (VisProg, Gupta & Kembhavi, CVPR 2023)** — composing vision tools via LLM.
- **AssistGPT** (Gao et al., 2023), **Chameleon** (Lu et al., NeurIPS 2023) — plan-then-execute compositional reasoning.

### 2.2 Vision-language agents for surveillance / search
- **VideoChat / Video-LLaMA / Video-ChatGPT** — video understanding via LLM, **but no retrieval over a gallery**.
- **TimeChat** (Ren et al., CVPR 2024) — time-sensitive video QA.
- **VLM4Surveillance** — anomaly detection style, not person search.

> **Gap (our white-space)**: To our knowledge, **no prior work** formulates *Person Search* as an *LLM Agent task* with a tool library and an auditable reasoning chain over a multi-camera gallery. This is SearchCop's primary contribution.

### 2.3 Agent evaluation
- **AgentBench** (Liu et al., ICLR 2024), **ToolBench** (Qin et al., ICLR 2024) — agent capability eval.
- **WebArena / VisualAgentBench** — embodied / web agents.

> **Gap**: No benchmark scores agents on *retrieval faithfulness*; SearchCop introduces **Reasoning Faithfulness** + **Tool-call Efficiency** as new metrics.

---

## 3. Positioning matrix

| Method                | Multi-cam | Modality fusion | Decision making | Interpretable | Eval'd on MEVID |
|----------------------|:---------:|:---------------:|:---------------:|:-------------:|:---------------:|
| TransReID            | ◐         | ✗               | ✗               | ✗             | ✗               |
| CLIP-ReID            | ◐         | text aux        | ✗               | ◐             | ✗               |
| CAL (clothes-change) | ✗         | clothes-aware   | ✗               | ✗             | ✗               |
| OpenGait family      | ✗         | gait only       | ✗               | ✗             | ✗               |
| VisProg / Chameleon  | ✗         | LLM-tool        | ✓               | ✓             | ✗               |
| **SearchCop (ours)** | **✓**    | **gait+reid+vlm**| **✓ planner**  | **✓ CoT**     | **✓**           |

---

## 4. Citation backlog (verify Day 2-3)

The following must be confirmed (year, venue, arXiv id) before paper submission:

- [ ] AP3D, GRL, STMN exact venues
- [ ] CAL CVPR 2022 paper id
- [ ] LTCC final venue (was ACM MM 2020 IIRC)
- [ ] MEVID WACV 2023 official cite
- [ ] OpenGait CVPR 2023 author list
- [ ] ReAct ICLR 2023 vs arXiv version
- [ ] HuggingGPT NeurIPS 2023 cite
- [ ] Visual Programming CVPR 2023 (Best Paper)
- [ ] Chameleon NeurIPS 2023 author list

> **TODO**: Researcher Agent on remote (with internet) should run `arxiv_search` and produce `paper/citations.bib`.
