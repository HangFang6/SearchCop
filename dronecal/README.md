# DroneCal · 工程目录

> 博士论文 Chapter 3：**Drone-Supervised Self-Calibrating BEV Perception for Wide-FoV Surveillance Cameras**
> 主要方法：**INCF (Implicit Neural Calibration Field) + TTCA (Test-Time Calibration Adaptation)**
> 投稿目标：CVPR 2027 / TPAMI

---

## 项目状态

- ✅ **2026-05-23**：方向定下，proposal 给老板看
- ✅ **2026-05-30**：老板拍板，Week 1 开始
- ⏳ **2026-08-22**：Week 12 末，实验 + paper 初稿
- ⏳ **2026-11-15**：CVPR 2027 投稿

## 目录结构

```
dronecal/
├── README.md                  # 本文件
├── data_status.md             # 真实数据资产清单（Day 6 填）
├── early_competitors.md       # 早期竞品调研（Day 6 填）
├── week1_self_report.md       # Week 1 自我复盘（Day 7 填）
├── week2_plan.md              # Week 2 计划（Day 7 填）
├── model/                     # 网络结构（INCF / TTCA）
│   ├── __init__.py
│   ├── toy_incf.py            # Day 5 写：toy 版 INCF
│   └── (后续) incf.py / ttca.py
├── data/                      # 数据加载器
│   └── __init__.py
├── experiments/               # 实验脚本
│   └── __init__.py
└── notebooks/                 # 探索性 jupyter（不进 paper 但便于调试）
```

## 相关文档

- 给老板的提案：`../docs/proposal_dronecal_for_advisor.md`
- 给另一个 AI 的 prompt：`../docs/dronecal_handoff_prompt.md`
- 你 Week 1 day-by-day 清单：`../docs/dronecal_week1_kickoff.md`

## 开发约定

- 不依赖 GPU 时跑在本机 CPU
- 真实训练在 H20 服务器（git pull → bash run.sh）
- 所有大文件（数据/checkpoint）`.gitignore` 不进 Git
- 所有 commit message 加前缀：`dronecal: ...`
