#!/usr/bin/env bash
# Run SearchCop full pipeline (LLM Agent + Tool Library) on MEVID
set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate searchcop

# Load secrets
set -a
[ -f .env ] && source .env
set +a

if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "[ERR] OPENAI_API_KEY not set. Edit .env first." >&2
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
