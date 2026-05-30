# DroneCal · Week 1 启动清单（2026-05-30 → 2026-06-05）

> **状态**：✅ 老板已拍板（5/23）→ 今天正式 Week 1 开工
> **你的目标**：7 天后能**独立读懂 NeRF 论文 + 跑通一个 toy implicit MLP**
> **时间投入**：每天 2 小时（不影响 SearchCop 主线）
> **验收标准**：第 7 天末能用一句话回答"INCF 和 NeRF 的对偶关系是什么"

---

## 一、Week 1 双线并行

```
本周双线并行（互不冲突）：

  你（人）                              另一个 AI
  ──────                              ──────────
  Day 1-7：NeRF 入门 + 跑 toy demo        Day 1-2：交付 task1_kickoff.md
                                          Day 3-7：交付 related_work_dronecal.md

  本周末汇合：                            Week 1 末：
    - 你能看懂 INR 的核心数学              - 文献综述初稿到手
    - 另一个 AI 文献综述完成               - 你能基于综述提关键问题

  Week 2 开始：你能和另一个 AI 在同一个频道讨论 INCF 设计了
```

---

## 二、Day-by-Day 清单（每天 2 小时）

### 🌟 Day 1（周日，5/31）· NeRF 论文精读

**理论目标**：理解 implicit neural representation 的核心思想

**任务**（共 2 小时）：
- [ ] **45 分钟**：精读 NeRF 论文（Mildenhall et al., ECCV 2020）
  - 核心数学：Volume Rendering equation、Positional Encoding、Hierarchical Sampling
  - 论文：[NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](https://arxiv.org/abs/2003.08934)
- [ ] **30 分钟**：看 5 分钟概念视频
  - YouTube 搜 "NeRF explained" 的几个 5 分钟视频
  - 推荐：Two Minute Papers / AI Coffee Break with Letitia
- [ ] **30 分钟**：在自己笔记里画图
  - **手画一遍 NeRF 的输入输出**：(x, y, z, θ, φ) → (RGB, σ)
  - **再手画一遍 INCF 的输入输出**：(u, v, k) → (x_w, y_w)
  - 看出对偶性：NeRF 是"3D 点 → 渲染像素"；INCF 是"图像像素 → 3D 地面点"
- [ ] **15 分钟**：打开 GitHub bookmarks 这两个仓库
  - https://github.com/yenchenlin/nerf-pytorch（最干净的 NeRF 实现）
  - https://github.com/nerfstudio-project/nerfstudio（生产级框架）

**Day 1 验收**：能用一句话回答"NeRF 为什么需要 positional encoding"。
> 提示：因为 MLP 对低频信号有偏置，positional encoding 把 (x,y,z) 升维到高频空间，让 MLP 能拟合细节。

---

### Day 2（周一，6/01）· SIREN + Sinusoidal Activations

**理论目标**：理解为什么 implicit field 偏好 sin/Fourier 类激活

**任务**：
- [ ] **40 分钟**：精读 SIREN 论文（Sitzmann et al., NeurIPS 2020）
  - [Implicit Neural Representations with Periodic Activation Functions](https://arxiv.org/abs/2006.09661)
  - 核心：用 sin 替代 ReLU，能拟合任意复杂信号 + 一阶导
- [ ] **30 分钟**：跑一个 SIREN toy demo（你的电脑 CPU 就够）
  - 仓库：https://github.com/vsitzmann/siren
  - 跑 `experiment_scripts/train_img.py` 拟合一张图（10 分钟出结果）
  - 看 loss 曲线 + 拟合效果
- [ ] **30 分钟**：思考一个问题
  - **如果用 SIREN 替代 NeRF 的 ReLU MLP，效果会怎样？**（有人做过：NeRV、Mip-NeRF 360）
  - 这对 INCF 有什么启发？（INCF 的几何映射是连续光滑的，SIREN 可能比 ReLU 好）
- [ ] **20 分钟**：在你的笔记本里写一段话
  - "我对 INCF 的初步直觉：MLP 架构选 X / Positional encoding 选 Y / 激活选 Z"
  - 不需要正确，建立你自己的方法直觉

**Day 2 验收**：能解释"sin 激活相比 ReLU 在 implicit field 里好在哪"。

---

### Day 3（周二，6/02）· Instant-NGP + Hash Encoding

**理论目标**：理解为什么 NeRF 训练慢，以及 Hash encoding 怎么加速 100 倍

**任务**：
- [ ] **40 分钟**：精读 Instant-NGP（Müller et al., SIGGRAPH 2022）
  - [Instant Neural Graphics Primitives with a Multiresolution Hash Encoding](https://nvlabs.github.io/instant-ngp/)
  - 核心：用 hash table + multi-resolution grid 替代 positional encoding，训练快 100 倍
- [ ] **30 分钟**：理解 hash encoding 的 trade-off
  - 内存换计算：每个空间点 → hash → table lookup（O(1)）
  - hash collision 怎么处理（让 MLP 学会区分）
- [ ] **30 分钟**：选定 INCF 的 backbone 候选
  - **选项 A**：纯 MLP + Fourier PE（最简单，跑得动）
  - **选项 B**：MLP + SIREN（中等，光滑）
  - **选项 C**：Hash-grid + tiny MLP（快，但 INCF 数据量小可能 overkill）
  - 你的初步决策（写在笔记里）：__________
- [ ] **20 分钟**：搜 "implicit field calibration" / "implicit homography" 看有没有人做过
  - 如果有 → 重要 baseline，告诉另一个 AI
  - 如果没有 → 你的 niche 更稳

**Day 3 验收**：能在 30 秒内解释"为什么 NeRF 的 hash encoding 是个 trade-off"。

---

### Day 4（周三，6/03）· 跑通第一个 NeRF demo

**理论目标**：从纸面进入代码，把 implicit field 真正在你电脑上跑起来

**任务**（这天可能多花 1 小时，无所谓）：
- [ ] **30 分钟**：环境准备
  ```cmd
  conda create -n nerf_study python=3.10
  conda activate nerf_study
  pip install torch torchvision tqdm numpy matplotlib imageio configargparse
  ```
- [ ] **60 分钟**：clone + 跑通 nerf-pytorch
  ```cmd
  cd D:\program\All-in-One-Gait-main
  git clone https://github.com/yenchenlin/nerf-pytorch
  cd nerf-pytorch
  bash download_example_data.sh  # 用 Git Bash 或 WSL
  python run_nerf.py --config configs/lego.txt
  ```
  - 第一次跑：500 iterations 大概 10 分钟出第一张渲染图（Lego 玩具）
  - **如果跑挂了**：换跑 `tiny-nerf` 单文件版本（CPU 也行）：
    https://github.com/bmild/nerf/blob/master/tiny_nerf.ipynb
- [ ] **30 分钟**：调试 + 看 tensorboard
  ```cmd
  tensorboard --logdir logs/
  ```
  看 loss 曲线 + 看不同 iteration 的渲染图
- [ ] **15 分钟**：截一张你跑出来的 NeRF 渲染图，发给自己/老板/记进笔记
  - **这是你"踏入 implicit field"的第一个交付物**

**Day 4 验收**：你电脑里有一个 NeRF checkpoint + 一张渲染图。

---

### Day 5（周四，6/04）· 改造 NeRF → 第一个 toy INCF

**理论目标**：把 NeRF 的核心代码"剪枝"成 INCF 的 toy 版

**任务**（这天最有意思，也最关键）：
- [ ] **30 分钟**：理解 NeRF 代码骨架
  - `run_nerf.py` 主入口
  - `run_nerf_helpers.py` 里的 `NeRF` class（这就是核心 MLP）
  - `get_embedder()` 函数（positional encoding）
- [ ] **60 分钟**：写一个 toy_incf.py（在 SearchCop/dronecal/ 目录下，我会建好空目录）
  ```python
  # 目标：给定合成数据 (u, v) → (x_w, y_w)，用 MLP 学出映射
  
  import torch
  import torch.nn as nn
  
  class ToyINCF(nn.Module):
      def __init__(self, n_freqs=8, hidden=128, depth=4):
          super().__init__()
          self.n_freqs = n_freqs
          in_dim = 2 + 2 * 2 * n_freqs  # (u,v) + sin/cos × n_freqs × 2
          layers = [nn.Linear(in_dim, hidden), nn.ReLU()]
          for _ in range(depth - 1):
              layers += [nn.Linear(hidden, hidden), nn.ReLU()]
          layers += [nn.Linear(hidden, 2)]  # 输出 (x_w, y_w)
          self.mlp = nn.Sequential(*layers)
      
      def positional_encoding(self, uv):
          freqs = 2.0 ** torch.arange(self.n_freqs).to(uv)
          encoded = [uv]
          for f in freqs:
              encoded += [torch.sin(uv * f * torch.pi),
                          torch.cos(uv * f * torch.pi)]
          return torch.cat(encoded, dim=-1)
      
      def forward(self, uv):
          x = self.positional_encoding(uv)
          return self.mlp(x)
  
  # 合成数据：模拟"广角图像像素 (u,v) → 地面 BEV (x_w, y_w)"
  # 用一个非线性多项式作为 GT
  def synthetic_gt(uv):
      u, v = uv[:, 0], uv[:, 1]
      x = u + 0.3 * u**2 - 0.1 * v**2
      y = v + 0.2 * u * v
      return torch.stack([x, y], dim=-1)
  
  # 训练循环
  if __name__ == "__main__":
      model = ToyINCF()
      opt = torch.optim.Adam(model.parameters(), lr=1e-3)
      for step in range(5000):
          uv = torch.rand(1024, 2) * 2 - 1   # [-1, 1]
          gt = synthetic_gt(uv)
          pred = model(uv)
          loss = ((pred - gt) ** 2).mean()
          opt.zero_grad(); loss.backward(); opt.step()
          if step % 500 == 0:
              print(f"step={step} loss={loss.item():.6f}")
  ```
- [ ] **30 分钟**：在 CPU 上跑通 toy_incf.py，看 loss 曲线
  - 如果 loss 5000 步内降到 1e-4 以下 → 你已经独立写出第一个 INCF 雏形
- [ ] **20 分钟**：思考扩展方向（写笔记）
  - "如果加 per-camera embedding，怎么改？" → 加一个 nn.Embedding(N, embed_dim)
  - "如果用 SIREN 激活，怎么改？" → 把 ReLU 替换成 sin
  - "Loss 应该用什么？dense pixel 监督是什么意思？" → 准备明天的 Day 6

**Day 5 验收**：toy_incf.py 在你电脑里能跑通，loss 曲线正常下降。

---

### Day 6（周五，6/05）· 关联到真实数据 + 看相关工作

**理论目标**：从 toy 迈向真实 DroneCal 设计

**任务**：
- [ ] **40 分钟**：列出你已有的真实数据
  - UAV 倾斜摄影：DSM 多大？正射图分辨率多少？
  - 广角监控：摄像头数量？FoV？分辨率？帧率？
  - 你已有的多项式标定：精度多少 cm？哪几个失效场景？
  - 写在 `dronecal/data_status.md`（我会建好空文件）
- [ ] **40 分钟**：搜文献查"有没有人用 UAV 当 GT 标定地面摄像头"
  - 搜索关键词：
    - "drone supervised camera calibration"
    - "uav-ground cross-view"
    - "aerial-to-ground homography"
    - "wide-fov surveillance calibration"
  - 找到的工作记到 `dronecal/early_competitors.md`
  - 期望结果：**几乎没有人做过**（这是好消息，也是 paper 卖点）
- [ ] **40 分钟**：和今天另一个 AI 给的 task1_kickoff.md（Day 1-2 应该已交付）对照
  - 它对你项目的"200 字自述"对吗？
  - 它问的问题你能回答吗？
  - 它的 7 天计划是不是和你的 Week 1 节奏对得上？
  - 把回答整理成一段话，发给另一个 AI

**Day 6 验收**：你能用 5 句话向陌生人讲清"为什么 UAV-ground 标定是个新问题"。

---

### Day 7（周六，6/06）· Week 1 复盘 + Week 2 计划

**目标**：闭环 Week 1，写一份给自己看的 progress report

**任务**：
- [ ] **40 分钟**：写 `dronecal/week1_self_report.md`
  - 我学会了什么？（NeRF / SIREN / NGP 大概懂了）
  - 我跑通了什么？（toy_incf.py + nerf-pytorch lego demo）
  - 我对 INCF 的初步设计直觉是什么？（MLP 架构 / PE / 激活）
  - 我下周需要解决的 3 个问题？
- [ ] **30 分钟**：审阅另一个 AI 的 Week 1 产出
  - related_work_dronecal.md 是否覆盖了主要工作？
  - 数据 pipeline 设计是否合理？
  - 有没有它没看到的问题？
- [ ] **30 分钟**：写 `dronecal/week2_plan.md`（粗略草稿，week 2 第一天再细化）
  - Week 2 主线：基于 Week 1 综述定下 INCF 架构 + 真实数据 pipeline
  - Week 2 我和另一个 AI 各自做什么
- [ ] **20 分钟**：周报（给老板用，可选）
  - 一段 200 字总结这周进度，老板下周一一看就清楚

**Day 7 验收**：你已经从"完全没碰过 NeRF"升级为"能独立讨论 INCF 架构选型的人"。

---

## 三、Week 1 装备清单

### 软件（你电脑要装的）
```cmd
conda activate opengait      # 已有
pip install matplotlib seaborn ipykernel jupyterlab  # 看 loss 曲线
git clone https://github.com/yenchenlin/nerf-pytorch  # 学习用
git clone https://github.com/vsitzmann/siren           # 学习用
```

### 电脑要求
- Day 1-3：纯看论文，**任何电脑都行**
- Day 4：跑 NeRF lego demo 推荐有 GPU（无 GPU 用 tiny_nerf 也能跑）
- Day 5：toy_incf.py CPU 即可（数据是合成的，几秒一个 epoch）

### 论文 PDF（推荐用 Notion / Zotero 做笔记）
- [x] NeRF (ECCV 2020) — Day 1
- [x] SIREN (NeurIPS 2020) — Day 2
- [x] Instant-NGP (SIGGRAPH 2022) — Day 3
- [ ] BARF (ICCV 2021)（联合优化相机参数，相关性强）— Week 2
- [ ] DeepSDF (CVPR 2019)（implicit shape）— Week 2 选读

---

## 四、Week 1 之后的 Week 2 预告

下周（6/06-6/12）的主要工作：

| 谁 | Week 2 任务 |
|---|---|
| 你 | 在 toy_incf.py 上加 per-camera embedding；接入真实 UAV+广角数据；跑出第一版 vs 多项式 baseline |
| 另一个 AI | 交付 method_dronecal.md（INCF + TTCA 完整伪代码 + 数学）+ data_pipeline.md |

---

## 五、Plan B 早期信号

如果 Week 1 末出现以下信号，立刻和我说 → 触发 Plan B：

1. ⚠️ NeRF demo 在你电脑上反复跑挂 → 改用 tiny_nerf，砍掉 GPU 依赖
2. ⚠️ toy_incf.py 5000 步 loss 不收敛 → 说明合成数据有问题，重新生成
3. ⚠️ 另一个 AI 文献综述发现"已有人做过完全相同的工作" → 立刻调整 framing
4. ⚠️ 你每天 2 小时 NeRF 学习无法坚持 → 砍 Day 5 toy_incf 改成"读论文笔记"模式

---

## 六、给 CodeBuddy 的话（也就是给我）

我（CodeBuddy）这周仍然专注 SearchCop 主线（fix2_ 升级），**不会主动 push 你 NeRF 学习**。但每天你跑完任务，**简短发我一句话**就行：

> "Day X 完成 / 没完成 / 卡在 X 上"

我会立刻判断要不要触发 Plan B 调整。

DroneCal 的另一个 AI 那边，**我不替你管理**——它的产出由你审核，有问题告诉我帮你判断。

---

## 七、立即行动（今天 5/30）

```
☐ 1. 把 docs/dronecal_handoff_prompt.md 整段复制给 GPT-4o / Claude
     （提示已加：项目时钟已激活、Week 1 起算）

☐ 2. 阅读这份 Week 1 清单 5 分钟，确认能跟着做

☐ 3. 进入 Day 1 任务（NeRF 论文精读，今天 2 小时）

☐ 4. 周日睡前用一句话回答："NeRF 为什么需要 positional encoding？"
```

加油。Week 1 是 12 周中最容易 chickening out 的一周，但只要 Day 1 跑起来，后面就靠惯性了。
