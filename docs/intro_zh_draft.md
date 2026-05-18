# SearchCop · Introduction 中文草案 v0.1

> 用途：给老板/合作者评审用；与 `paper/sections/01_introduction.tex` 严格对应（5 段 + 5 个贡献），便于互相核对。
> 状态：Day 1 初稿，所有具体数字用 `[TODO]` 占位，等实验跑完再回填。
> 写作日期：2026-05-18

---

## 第 1 段 · 痛点与场景（Why this problem matters）

在公共安全、走失人员侦查、大规模监控取证等场景中，**跨摄像头的人员检索（multi-camera person search）** 是一项基础能力。它和图像层面的行人重识别（Re-ID）不同：分析师真正面对的问题不是"两张图像是不是同一个人"，而是"在一个由几十到上百个摄像头组成的网络中，**这个人何时出现、被哪个摄像头捕获**，能否给出一个排序的候选列表 + 一个**人能验证**的解释"。

实际场景中查询输入高度异构：可能是一张正面 crop、一段从背后拍的 tracklet、一段只能描述衣着的目击证言文本。这种条件对系统提出了三个新要求：

- **自适应（adaptive）**：不同查询要走不同检索路径
- **成本感知（cost-aware）**：不要对每个查询都把所有重模型跑一遍
- **可审计（auditable）**：必须给出可解释的决策链，让分析师采信结果

---

## 第 2 段 · 现有方法的三大结构性缺陷（What's wrong with the status quo）

主流做法是把多摄像头检索退化为一次性的特征匹配：用 Re-ID 网络（TransReID、CLIP-ReID）或步态网络（GaitSet、OpenGait）把 query 编码成一条向量，到一个固定模态的 FAISS 索引里取 Top-K，再用人工调好的权重把多模态做后期融合。这种范式忽略了三个**结构性事实**：

1. **查询质量是变量，编码器选择不应是常量**
   一段干净的步行序列适合 gait 编码器；一张严重遮挡的 crop 反而 ReID 更靠谱。但现有系统在**入库时就锁定模态**，无法根据查询情况切换。

2. **多摄像头网络自带时空先验，而它们被丢掉了**
   人在 t 时刻出现在摄像头 A，物理上不可能在几秒后出现在不相邻的摄像头 B。这种拓扑+时间窗约束几乎能砍掉一半误报，但现有 pipeline 几乎都没用。

3. **换衣场景的根本问题不是编码器不够强**
   PRCC、LTCC 等换衣 benchmark 暴露的痛点是：任何**单一**模态都会在某些 query 上崩溃。解决方案不是再训一个更强的 encoder，而是要有一个**决策策略**，懂得"什么时候该换工具"。

---

## 第 3 段 · 我们的核心洞察（Insight）

我们的核心观点：**跨摄像头人员检索本质上不是"一次性匹配"问题，而是"多步工具调用决策"问题**。

LLM Agent 领域近两年的进展（ReAct、Toolformer、Visual Programming、Chameleon）已经证明：当把 LLM 接入一个**结构化的工具库**时，它能为复杂任务规划合理的工具调用顺序，并产出**可追溯的推理链**。

我们把这个范式引入人员检索：

- 既有的 gait/ReID/VLM 编码器，从"单体 pipeline"变成**LLM 动作空间里的可调用工具**
- 既有的相机拓扑先验，从"被丢弃的元数据"变成**一个时空过滤工具**
- LLM 的工作不是看像素，而是**决定下一步该调哪个工具，并判断何时已经能回答**

---

## 第 4 段 · 方法概述 + 五大贡献（What we propose）

我们提出 **SearchCop**——首个面向跨摄像头人员检索的 LLM Agent。SearchCop 的核心是一个 **Planner / Executor / Critic 循环**：

- **Planner**（GPT-4o / Doubao-pro 等前沿 LLM 驱动）每步发出一个工具调用，从 10 个工具中选一个：gait 编码、CLIP-ReID 编码、多向量重排、VLM 描述、时空过滤、质量评分、FAISS 粗筛、证据融合、SAM 剪影、跟踪抽取
- **Executor**（纯 Python）执行该工具，返回类型化结果
- **Critic** 评估当前步骤的有用性 + 当前 Top-1 的置信度，触发**自纠正 replan**——当多模态相互矛盾或置信度长时间停滞时

我们在 MEVID（主战场）+ MARS + DukeMTMC-VideoReID（经典视频 Re-ID）+ PRCC + LTCC（换衣 Re-ID）上评估，使用标准 mAP/CMC + **三个新指标**：Reasoning Faithfulness、Tool-call Efficiency、Cross-camera Recall。

### 五大贡献

| # | 贡献 | 一句话 |
|---|---|---|
| 1 | **任务范式重构** | 首次把人员检索从"单步特征匹配"重构为"LLM Agent 工具调用任务"，引入显式的 plan-act-criticize 循环 |
| 2 | **工具库与 Schema 设计** | 设计 10 个带严格 JSON schema 的工具，让 Planner 在不依赖 LLM 网关原生 function-calling 的情况下也能可靠组合 |
| 3 | **时空推理 prompt** | 把相机拓扑图 + 时间窗先验编码进 Planner prompt，让 Agent 在调用重模型之前先剪掉物理上不可能的候选 |
| 4 | **自纠正 Agent 循环** | Critic 输出校准过的置信度 + needs-correction 旗标，让系统在证据弱时主动扩窗、扩相机或换编码器；用工具参数去重防止死循环 |
| 5 | **新评估协议** | 提出 Reasoning Faithfulness（陈述理由与工具结果的一致性）+ Tool-call Efficiency（每个正确答案平均调用多少工具），把"Agent 行为"变成可测量的论文产物 |

---

## 第 5 段 · 实验亮点 + 路线图（Headline + roadmap）

实验上，SearchCop 在 MEVID 上取得 [TODO X.X]% 的 mAP，相对最强的"步态+ReID 人工融合"基线提升 [TODO +Δ]%；在 PRCC / LTCC 上 rank-1 提升 [TODO X.X]%。Agent 平均每个 query 调用 [TODO N] 个工具，Critic 标记其中 [TODO Y]% 是有用调用；推理链被人工评估为"可信"的比例为 [TODO Z]%。

论文余下部分：第 2 节回顾相关工作（视频 Re-ID、跨摄像头检索、LLM Agent）；第 3 节详述工具库、Planner/Critic prompt、自纠正循环；第 4 节给出实验和 ablation；第 5 节总结。

---

## 写作笔记（不进 paper）

### 这版 intro 的"卖点"分布

- **P1**：场景具体 + 三个新要求（adaptive / cost-aware / auditable）—— 暗示后面三个核心贡献的对应
- **P2**：用 Numbered list（结构性事实 1/2/3）抢占审稿人记忆 —— 这三条直接对应贡献 1/3/4
- **P3**：把 LLM Agent 范式接到 Person Search，是全文最关键的一段，必须 motivation 强
- **P4**：5 个 itemized contribution，CVPR 评审最爱看
- **P5**：headline 数字 + roadmap

### 老板可能会问的问题（提前准备）

1. **"为什么不用 OpenAI function-calling？"**
   答：我们用的内部 Doubao 网关不支持 function-calling，所以用"指令式 JSON 输出 + 容错解析"，反而更通用——不依赖任何特定 LLM 厂商的接口扩展，可移植到 Qwen、InternLM 等开源 LLM。

2. **"Reasoning Faithfulness 怎么量化？"**
   答：两种方案：① 程序化——检查"Planner 的 thought 字段中提到的工具结果"是否真的出现在 ToolResult.meta 里；② LLM-as-judge——用另一个 LLM 给推理链打分。两个一起报，互为佐证。

3. **"Tool-call Efficiency 越低越好吗？"**
   答：是的，且这是 SearchCop 相对穷举 baseline 的关键优势——baseline 必然把每个工具都跑一遍，效率分母固定 = N_tools；SearchCop 通常 4-6 步就停。

4. **"为什么不用 GPT-4o？"**
   答：内部代码评审 / 公司合规要求；同时 Doubao-pro 在中文场景下表现更好，且推理成本对长期运行更友好。我们已设计了双模型路由（pro/lite）来控制成本。

5. **"和 Visual Programming 等 LLM-Agent 视觉工作的区别？"**
   答：他们是 single-image VQA 场景；我们是 **多摄像头 + 时间维度 + 大规模 gallery 检索**，工具库结构、时空先验、自纠正循环都是新设计。

### 下一步可改进项（v0.2 时考虑）

- [ ] P3 加一句"为什么 Re-ID 社区还没人这么做"——卡位 first-of-its-kind
- [ ] P4 itemized list 太长，camera-ready 时可能要压成 4 项
- [ ] 加一张 teaser figure（Fig. 1）：展示 Agent 的一次完整 trace
- [ ] 把 [TODO] 改用 \todo{} LaTeX 宏，编译时自动高亮红色
