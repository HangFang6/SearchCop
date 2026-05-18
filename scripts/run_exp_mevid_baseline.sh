#!/usr/bin/env bash
# Run baseline (no Agent, classical Gait+ReID fusion) on MEVID
set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate searchcop

CFG=experiments/configs/mevid_baseline.yaml
EXP_NAME=mevid_baseline_$(date +%Y%m%d_%H%M)
OUT=results/${EXP_NAME}
mkdir -p "${OUT}"

echo "[*] Running ${EXP_NAME}"
python -m experiments.run_baseline \
    --config ${CFG} \
    --output_dir ${OUT} \
    2>&1 | tee logs/${EXP_NAME}.log

echo "[OK] Results -> ${OUT}"
