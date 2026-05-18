# SearchCop · Related Work 中文草案 v0.1

> 用途：给老板/合作者评审；与 `paper/sections/02_related_work.tex` 严格对应（4 个子节 + Positioning Matrix 表）。
> 写作策略：每个子节末尾必有一句 **"Gap → 我们怎么补"**，让审稿人在 30 秒内抓到 SearchCop 的差异化卖点。
> 写作日期：2026-05-18 深夜

---

## §1 视频 Re-ID（Video Re-Identification）

**主流路线**：把人员检索看成 tracklet 上的"表征学习"问题。
- 图像级：TransReID（Transformer backbone）、CLIP-ReID（视觉-语言预训练）
- 视频级：AP3D（3D 卷积时序聚合）、GRL（全局-局部互引导）、STMN（时空记忆网络）
- 标准 benchmark：MARS、DukeMTMC-VideoReID

**Gap**：
这类方法只优化**单一相似度函数**，不提供**决策策略**——比如不会因为 query 被遮挡了就主动切换模态，也不会解释 Top-1 是怎么选出来的。

**SearchCop 怎么补**：把这些 encoder 当成 **tool 保留下来**（gait_encode、reid_encode），但把"用哪个模态"和"什么时候停止检索"这两个决策权**升级到 LLM Planner 手里**。

---

## §2 跨摄像头检索 + 时空推理（Multi-Camera Person Search & Spatio-Temporal）

**主流路线**：联合检测 + Re-ID 跨多个摄像头。
- 经典 benchmark：PRW、CUHK-SYSU
- 多环境长时间跨度：MEVID（**我们的主战场**）
- 用时空先验（拓扑、转移时间、图传播）做精化：Lv2018、Wang2019

**Gap**：
- 即便用了时空先验，也是当成**事后过滤器**——固定权重、不可调
- 没有任何工作把"何时调用拓扑"、"用哪个 encoder"、"答案够不够好"这些决策**升格为系统级 action**

**SearchCop 怎么补**：把"调用拓扑过滤"、"切换编码器"、"宣布完成"全部变成 **LLM agent 工具空间里的 first-class action**——agent 自己决定每一步要不要调它们。

---

## §3 换衣 Re-ID + 步态识别（Clothing-Change & Gait）

**两条互补路线**：
1. **Clothing-aware 表征学习**（外观感知不变）：CAL（CVPR 2022）on PRCC / LTCC
2. **步态识别**（用走路动力学，不依赖外观）：GaitSet、GaitPart、OpenGait 家族

**Gap**：
两条路都要求**入库时就锁定模态**。一旦 query 来了不知道是该走 ReID 还是 gait，离线后期融合也只能用固定权重做加权和。

**真实场景的痛点**：
- 正面遮挡的 crop → 应该走 ReID（gait 没有完整剪影）
- 背面干净的 tracklet → 应该走 gait（ReID 看不到正脸特征）
- 但同一个查询往往两种情况都有

**SearchCop 怎么补**：两个 encoder 都保留为按需调用的 tool，**Planner 每个 query 自己决定信谁**；当 Critic 检测到两个模态打架（rank 严重不一致），系统会主动 invalidate 当前结论并 replan。

---

## §4 LLM Agent 与 Tool Use

**主流路线**：
- 起点：ReAct（推理+行动交替）、Toolformer（自监督学习调工具）
- 多模态编排：HuggingGPT（调度 HF 模型库）、Visual Programming（视觉工具组合做 VQA）、Chameleon（即插即用组合推理）
- 评测：AgentBench、ToolBench

**Gap**：
- 现有 agent benchmark 关注的是**能力**（agent 能不能解决任务）
- 没有人测过**检索忠实度**（agent 的推理链 vs 工具结果是否一致）
- 也没人在**大规模多摄像头 gallery 检索**上做 agent

**SearchCop 怎么补**：把 tool-using 范式带进 person search，**新增两个评测 primitive**：
- **Reasoning Faithfulness**（推理链与工具输出是否对得上）
- **Tool-call Efficiency**（每个正确答案平均要调多少工具）

这两个指标可以被未来所有"agent + 检索"工作复用，是**通用方法论贡献**而不是只针对 SearchCop。

---

## §5 视频/监控领域的视觉-语言 Agent

**最近的视频多模态 LLM**：
- 视频理解：VideoChat、Video-LLaMA、Video-ChatGPT、TimeChat
- 视频 QA + 工具调用：AssistGPT
- 监控异常描述：少量 LLM-driven 工作

**Gap**：
- 这些都在做**描述视频**或**回答关于视频的问题**
- **没有人在多摄像头 gallery 里"找同一个人"** + 加时空约束 + 自纠正

**SearchCop 怎么补**：据我们所知，是**第一个**专门面向跨摄像头人员检索任务的 LLM Agent。

---

## §6 Positioning Matrix（论文里的 Table 1）

5 个维度对比：

| 方法 | 多摄像头 | 模态融合 | 决策能力 | 可解释 | MEVID 评测 |
|---|:---:|:---:|:---:|:---:|:---:|
| TransReID | ◐ | ✗ | ✗ | ✗ | ✗ |
| CLIP-ReID | ◐ | text aux | ✗ | ◐ | ✗ |
| CAL（换衣） | ✗ | clothes-aware | ✗ | ✗ | ✗ |
| OpenGait 系 | ✗ | gait only | ✗ | ✗ | ✗ |
| HuggingGPT | ✗ | LLM-tool | ✓ | ✓ | ✗ |
| Visual Programming | ✗ | LLM-tool | ✓ | ✓ | ✗ |
| Chameleon | ✗ | LLM-tool | ✓ | ✓ | ✗ |
| AssistGPT | ◐ | LLM-tool | ✓ | ✓ | ✗ |
| **SearchCop（本文）** | **✓** | **gait+reid+vlm** | **✓** | **✓** | **✓** |

**关键观察**：5 维全 ✓ 的只有 SearchCop 一行。任何审稿人质疑"和 X 工作有什么区别"，直接指这张表就行。

---

## 写作笔记（不进 paper）

### 这版 Related Work 的 5 个隐藏卖点

1. **每段都有 Gap → Ours 收尾**：训练审稿人形成"读完一段就联想到 SearchCop"的反射
2. **Positioning Matrix 用 5 维**：不是常见的 3 维表，避免被人轻易"补一格"反驳
3. **MEVID 列单列出来**：直接在表里抢"first to evaluate on MEVID with agent"的 niche
4. **§4 末尾把 Faithfulness + Efficiency 包装成"通用方法论贡献"**：审稿人即使觉得 SearchCop 本身不够新，也会给"提了新评测指标"加分
5. **§5 一句话定位"first-of-its-kind"**：但**不放第 1 段**，避免 over-claim 引发审稿人反弹；放在第 5 段作为收尾自然得多

### 老板可能问的 7 个硬问题（提前准备）

#### Q1: 跟 HuggingGPT、Visual Programming 差别只是任务换成 person search？
**A**：不止换任务。三个具体差异：
- 他们的 tool library 是通用 NLP/CV 模型 API；我们是 **gallery 检索专用工具**（FAISS、multivector_rerank、spatio_temporal_filter）
- 他们没有 Critic + 自纠正循环
- 他们没有"基于工具结果的 Faithfulness 度量"

#### Q2: 既然有 OpenGait + CLIP-ReID 等强 encoder，为什么不直接做 ensemble？
**A**：ensemble 是 SearchCop 的 baseline 之一（"无 Agent 的 RRF 融合"）。我们的实验会显示：在干净查询上 ensemble 和 SearchCop 持平，但在**遮挡 + 换衣 + 跨摄像头**这三类硬 query 上，SearchCop 显著超过 ensemble，因为它会主动调 VLM、调时空过滤、做 self-correction。

#### Q3: AssistGPT 也调了 modality-specific 模型，区别在哪？
**A**：AssistGPT 是 video QA（一个视频一个问题）；我们是 **gallery 检索**（一个 query 对几千个候选），任务结构、评测指标完全不同。AssistGPT 也没有 spatio-temporal 推理，没有 multi-vector rerank，没有 confidence-based 自纠正。

#### Q4: GPT-4o 已经有 vision；为什么不让它直接看所有 crop？
**A**：成本和延迟。一个 MEVID query 配一个 1000 人的 gallery，让 GPT-4o 看 10000 张图 = 单次查询 $50+ 美元 + 几分钟。SearchCop 用 GPT/Doubao 只做**决策**，重模型推理交给本地 GPU 跑，每个 query 平均 4-6 次 LLM 调用 + ~10 次本地推理。

#### Q5: 论文里为什么不引一些 GAN 重建 / face Re-ID 的工作？
**A**：scope 控制。我们专注于 person search 这个明确的设定；face / GAN-based 重建是相邻 niche，引了反而稀释 contribution。

#### Q6: 这三个新指标（Faithfulness / Efficiency / Cross-cam Recall）算"contribution"吗？
**A**：算。CVPR 评审里 "evaluation protocol contribution" 是被认可的一类贡献（参考 NeRF 引入的 PSNR-aware 评测、AVA 引入的时序定位评测）。我们会在 paper §4 单独有 0.5 column 写这部分。

#### Q7: 为什么不做 unified 模型而要做 agent？端到端不香吗？
**A**：end-to-end 在大 vocabulary（千人以上 gallery）下还没人做出来过；而且 agent 路线的可解释性优势是端到端不可替代的。后续 future work 可以提"用 agent 产生的 trace 反过来做 distillation 训 unified 模型"。

### v0.2 想做的优化（D10 时回炉）

- [ ] §1 加一句"统计上 X% 的现有方法都没在 MEVID 上 evaluate"——量化 MEVID gap
- [ ] §2 加 1-2 个最新（2024）的 multi-camera 工作引用
- [ ] §4 把 ReAct/Toolformer 引用顺序与原文一致
- [ ] Positioning Matrix 加一列"agent trace 公开"——对所有 baseline 都是 ✗，强化我们卖点
- [ ] 全部 cite 在 D2-D3 联网时核对 venue/年份/作者列表
