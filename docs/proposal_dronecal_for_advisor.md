# DroneCal · 博士论文 Chapter 3 研究方案（给导师的 1 页执行摘要）

> **学生**：方航
> **博士论文 Chapter 3 拟定研究**
> **拟投目标**：CVPR 2027 / TPAMI
> **预计开题答辩**：2026 年 7-8 月（中科大）
> **本摘要日期**：2026-05-23

---

## 一、一句话研究问题

> **导师，我想做这件事**：用园区已有的无人机倾斜摄影作为 dense 监督，让广角监控摄像头**学会自己持续校准自己**——把传统的"一次性人工棋盘标定"重构为"无人机 + 行人双源监督的 lifelong calibration"。

---

## 二、为什么这是一个真正的博士级问题

| 现有方法 | 真实部署中的失败 | 我们要解决 |
|---|---|---|
| 人工棋盘格标定 | 1 个摄像头 1 小时，无法规模化 | 自动化 |
| SfM / 学习方法 | 大 FoV 鱼眼下精度崩溃 | dense supervision |
| 一次性标定 | 1-2 年后机械漂移失效 | lifelong online |

**国家产业背景**：城市级 6 亿摄像头，几乎全部依赖人工标定，2 年后失效需要重做——这是真痛点，不是 paper 数字游戏。

---

## 三、核心方法（两层创新点）

### 创新点 1：**Implicit Neural Calibration Field (INCF)**

把校正参数从"显式多项式"换成**隐式神经表征**：

```
传统多项式：  BEV(u,v) = Σ a_ij × u^i × v^j        (4-6 阶，离散，per-camera 独立)

INCF (本文)： BEV(u,v,k) = MLP([PE(u,v); embedding_k])
              MLP 全园区共享，每个摄像头只学一个 64-dim embedding
              UAV 倾斜摄影 → DSM/正射图 → 提供 dense pixel-level 监督
```

**学术新颖性**：第一次把 Implicit Neural Field（NeRF 系列 2020-2024 顶会霸主）引入 surveillance camera calibration。

### 创新点 2：**Test-Time Calibration Adaptation (TTCA)**

把摄像头校正从"一次性"升级成"终身在线"：

```
Stage 1 (offline)：用 UAV 数据训出初始 INCF
Stage 2 (online)：每个时段，用园区行走的人作为 update signal
                  跨摄像头一致性约束：同一个人的 BEV 坐标应重合
                  极轻量 MLP 微调，无需重飞 UAV
```

**学术新颖性**：第一个把 Test-Time Adaptation 用于 calibration 域，做出"摄像头自愈"系统。

---

## 四、3 个 contribution（CVPR/TPAMI 标准结构）

1. **新表征**：Implicit Neural Calibration Field（INCF），把多项式投影升级为可学习隐式场
2. **新算法**：Test-Time Calibration Adaptation（TTCA），用行人轨迹做 lifelong self-healing
3. **真实落地**：在公司园区 N 个摄像头长期部署 + 评测，建立 DroneCal-Bench 评测协议

---

## 五、和博士论文整体的关系

```
Chapter 3 (本提案)         Chapter 4 (在做)              Chapter 5 (后续)
─────────────────         ────────────────              ─────────────────
DroneCal:                 SearchCop:                    Distillation:
INCF + TTCA              LLM Agent + Tool Lib          End-to-End Edge
"感知统一"                 "决策智能"                     "落地可信"
                                                        
TPAMI / CVPR             CVPR 2027                     NeurIPS / TPAMI
                                                        
       │                       ▲                              │
       │                       │                              │
       └──── 提供 BEV 坐标 ────┘                              │
            + 全局 ID                                         │
                                                              │
                   完整闭环：感知 → 决策 → 部署 ◄──────────────┘
```

**博士整体故事**："用 LLM Agent 在数字孪生世界里做监控视频检索"——三章环环相扣。

---

## 六、风险评估（坦诚）

| 风险 | 严重度 | Plan B |
|---|---|---|
| **我没训过 NeRF/SIREN，怕做不出 INCF** | 🟡 中 | 已列 4 周入门路线（见附录 A），主流 INR 框架成熟，开源代码可复用 |
| INCF 在新场景泛化失败 | 🟡 中 | 退化为 per-camera MLP（仍优于多项式） |
| 数据脱敏耗时长 | 🟡 中 | 先用合成数据 + 内部数据写 paper，发布版本可延后 |
| 公司同意发表但不同意发数据集 | 🟡 中 | 改为发布"评测协议 + 100 张测试图"的 mini-bench |
| 12 周做不完（开题前 2-3 月） | 🟢 低 | 第 8 周如果只完成 INCF，可投 CVPR poster；TTCA 留 Chapter 5 |

---

## 七、开题前 12 周关键 milestone（从 5/24 起算）

| 周 | 时间 | 关键产出 | 风险检查点 |
|---|---|---|---|
| W1-2 | 5/24 - 6/06 | NeRF/INR 入门 + SIREN/NGP code reproduce | 能否独立跑通 NGP demo？ |
| W3-4 | 6/07 - 6/20 | INCF v0：单摄像头 + UAV GT 训出第一个能用的 MLP | 能否复现多项式精度？ |
| W5-6 | 6/21 - 7/04 | INCF v1：N 摄像头共享 backbone + per-camera embedding | 是否优于多项式？ |
| W7-8 | 7/05 - 7/18 | TTCA：用行人轨迹做 online refinement | TTCA 是否带来明显增益？ |
| W9-10 | 7/19 - 8/01 | 全实验 + ablation + 下游 ReID/检测验证 | 数字够不够投顶会？ |
| W11-12 | 8/02 - 8/15 | Paper 初稿 + 开题 PPT | 投稿 + 开题答辩 |

**第 4 周末是关键决策点**：如果 INCF v0 跑不通，立即触发 Plan B（退化为多项式 + 学习的 refinement，仍可投 CVPR poster）。

---

## 八、需要导师确认的 3 件事

1. ✅ **方向是否认可**："Implicit Neural Calibration + Lifelong TTA" 这个 framing 您觉得合适吗？
2. ✅ **资源支持**：能否安排 1-2 张 H20 给我跑 INR 训练？（INCF 训练量小，1 张就够）
3. ✅ **数据策略**：您倾向"完整发布数据集"还是"只发评测协议"？

---

## 附录 A · 技术入门路线（4 周）

> 解决 q2 的"没训过 implicit MLP 怕做不出来"问题。

### Week 1：理论入门
- [ ] NeRF (Mildenhall et al., ECCV 2020) — 实操跑一遍 nerf-pytorch
- [ ] SIREN (Sitzmann et al., NeurIPS 2020) — 理解 sinusoidal activation
- [ ] Positional Encoding 数学原理
- 资源：[课程 CS231n implicit representation lecture](https://cs231n.stanford.edu/) + [NeRF 综述](https://arxiv.org/abs/2210.00379)

### Week 2：代码 reproduce
- [ ] Run [NeRFStudio](https://github.com/nerfstudio-project/nerfstudio) 跑通一个 demo scene
- [ ] Run [Instant-NGP](https://github.com/NVlabs/instant-ngp) 训出第一个 hash-grid MLP
- [ ] 改造：把 NeRF 的 `(x,y,z) → (rgb,σ)` 改成 `(u,v) → (x_w, y_w)`（这就是 INCF 的 toy 版）

### Week 3：领域迁移
- [ ] 读 implicit-based geometry：DeepSDF / IDR / NeuS
- [ ] 读 implicit-based calibration（少量但有）：[NeRF-- (CVPR 2023)] / [BARF (ICCV 2021)]（联合优化相机参数）
- [ ] 自己写一个最小化的 INCF：1 个摄像头 + UAV GT，看能不能拟合出 BEV

### Week 4：进阶
- [ ] 引入 per-camera embedding（multi-condition INR）
- [ ] 引入 hash encoding 加速
- [ ] 第一版 baseline 跑通 → 进入 W3 主线

**关键 codebase**：
- `nerfstudio` (主框架)
- `instant-ngp` 的 hash encoding 模块
- `kornia` (用其 calibration baseline 做对比)

---

## 附录 B · 我已经具备的基础（不用从头学）

- ✅ 完整园区 UAV 倾斜摄影（DSM + 正射图）
- ✅ N 个广角摄像头的实时数据流
- ✅ 多项式拟合 baseline（已实现）
- ✅ 特征金字塔多视图拼接（已实现）
- ✅ 行人检测 + 跨摄像头跟踪（公司系统已部署）
- ✅ Chapter 4 的 SearchCop 框架（已完成 50%）

**结论**：核心数据 + baseline 都齐，主要工作量在"把多项式升级成 INCF"+"把离线标定升级成 TTCA"上。

---

## 九、一句话结论

> 这是一个**用我已有的工程资产**（UAV + 园区监控）撬动**学术界正在关注的两个热点**（Implicit Neural Field + Test-Time Adaptation）的 niche。如果 12 周内做出来，开题答辩时我能交出：
> 
> - **1 篇投了 CVPR / TPAMI 的 Chapter 3 paper**
> - **1 个真实部署的数字孪生 + DroneCal 系统**
> - **博士论文三章纵深叙事的清晰路线图**

请导师拍板。

---

## 十、如果导师有疑虑的备选方案

| 备选 | 适用情形 | 调整 |
|---|---|---|
| Plan B1 | 导师觉得 INCF 风险高 | 只做 TTCA，多项式仍用，但加入 lifelong update。仍可投 TPAMI |
| Plan B2 | 导师觉得 TTCA 不够 | 只做 INCF + 完整 benchmark 发布，靠数据集 + 方法双 contribution 投 CVPR |
| Plan B3 | 导师建议先做 Chapter 4 | 先 SearchCop（已 50%）冲 CVPR 2027，Chapter 3 推到 2027 春季 |
