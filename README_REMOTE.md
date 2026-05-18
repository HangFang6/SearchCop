# Remote H20 Deployment Playbook

> Audience: you, when you SSH into the H20 server for the first time.
> Goal: zero-friction setup so experiments can start within 30 minutes.

---

## 0. Pre-flight checklist (on remote)

```bash
nvidia-smi                   # confirm H20 visible, CUDA driver version
echo $CUDA_VISIBLE_DEVICES   # confirm GPU allocation
df -h /data                  # confirm dataset disk has enough space (>= 500GB)
which conda                  # confirm conda available; if not, install Miniconda
```

---

## 1. One-shot setup

```bash
# 1.1 Clone repo (private)
git clone git@github.com:<your-user>/SearchCop.git
cd SearchCop

# 1.2 Build conda env (~5-10 min)
bash scripts/setup_remote.sh

# 1.3 Configure secrets (NEVER COMMIT)
cp .env.example .env
vim .env                     # fill OPENAI_API_KEY and MYSQL creds

# 1.4 Sanity test (mock LLM, no API call)
conda activate searchcop
pytest tests/ -v
```

---

## 2. Dataset preparation

Datasets are NOT in Git. Download on remote directly:

| Dataset | Where to get | Target path |
|---|---|---|
| MEVID | https://github.com/Kitware/MEVID | `/data/MEVID/` |
| PRCC | http://www.isee-ai.cn/~yangqize/clothing.html | `/data/PRCC/` |
| LTCC | https://naiq.github.io/LTCC_Perosn_ReID.html | `/data/LTCC/` |
| MARS | http://zheng-lab.cecs.anu.edu.au/Project/project_mars.html | `/data/MARS/` |
| DukeMTMC-VideoReID | https://exposing.ai/duke_mtmc/ | `/data/DukeMTMC-VideoReID/` |

After download, create symlinks inside the repo:

```bash
ln -s /data/MEVID                data/MEVID
ln -s /data/PRCC                 data/PRCC
ln -s /data/LTCC                 data/LTCC
ln -s /data/MARS                 data/MARS
ln -s /data/DukeMTMC-VideoReID   data/DukeMTMC-VideoReID
```

---

## 3. Model weights

```bash
mkdir -p weights/{yolox,sam,opengait,clip_reid}
# Reuse Gait-Project weights if available:
# scp -r local:D:/program/All-in-One-Gait-main/Gait-Project/weights/* weights/

# Or download fresh:
# - YOLOX-x: https://github.com/Megvii-BaseDetection/YOLOX
# - SAM ViT-L: https://github.com/facebookresearch/segment-anything
# - OpenGait BaselineDemo: https://github.com/ShiqiYu/OpenGait
# - CLIP-ReID: https://github.com/Syliz517/CLIP-ReID
```

---

## 4. Run experiments

### 4.1 Baseline (no Agent)
```bash
bash scripts/run_exp_mevid_baseline.sh
```

### 4.2 SearchCop full system
```bash
bash scripts/run_exp_mevid_searchcop.sh
```

### 4.3 Ablations
```bash
bash scripts/run_ablation_tools.sh        # remove each tool one at a time
bash scripts/run_ablation_reasoning.sh    # CoT on/off
bash scripts/run_ablation_selfcorrect.sh  # self-correcting loop on/off
```

Results land in `results/<exp_name>/` and are auto-summarized into `results/summary/all_runs.xlsx`.

---

## 5. Bring results back to local

```bash
# On local
cd D:\program\All-in-One-Gait-main\SearchCop
git pull                      # if remote committed summary tables
# Or rsync large result trees:
rsync -avz user@h20:~/SearchCop/results/summary/ ./results/summary/
```

---

## 6. Troubleshooting

| Symptom | Fix |
|---|---|
| `faiss-gpu` import fails | Reinstall matching CUDA: `conda install -c pytorch faiss-gpu=1.7.4` |
| `torch.cuda.is_available()` False | Re-export `CUDA_VISIBLE_DEVICES` after SSH reconnect |
| OpenAI rate limit | `cache/` will retry with exponential backoff (tenacity) |
| Out of memory on H20 | Reduce `batch_size` in `experiments/configs/*.yaml` |
