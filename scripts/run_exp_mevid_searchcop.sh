#!/usr/bin/env bash
# Run SearchCop full pipeline (LLM Agent + Tool Library) on MEVID
set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate searchcop

# Load secrets
set -a
[ -f .env ] && source .env
set +a

if [ -z "${DOUBAO_APP_ID:-}" ] || [ -z "${DOUBAO_APP_KEY:-}" ]; then
    echo "[ERR] DOUBAO_APP_ID / DOUBAO_APP_KEY not set. Run: cp .env.example .env" >&2
    exit 1
fi

CFG=experiments/configs/mevid_searchcop.yaml
EXP_NAME=mevid_searchcop_$(date +%Y%m%d_%H%M)
OUT=results/${EXP_NAME}
mkdir -p "${OUT}"

echo "[*] Running ${EXP_NAME}"
python -m experiments.run_searchcop \
    --config ${CFG} \
    --output_dir ${OUT} \
    --cache_dir cache/ \
    2>&1 | tee logs/${EXP_NAME}.log

echo "[OK] Results -> ${OUT}"
