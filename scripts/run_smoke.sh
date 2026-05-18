#!/usr/bin/env bash
# Local smoke test: end-to-end SearchCop on the MOCK dataset, no GPU/API.
# Useful sanity check before pushing to remote.
set -euo pipefail

cd "$(dirname "$0")/.."

OUT=results/local_smoke_$(date +%Y%m%d_%H%M%S)
mkdir -p "${OUT}"

python -m experiments.run_searchcop \
    --config experiments/configs/mock_smoke.yaml \
    --output_dir "${OUT}" \
    --stub --mock_llm

echo ""
echo "[OK] Smoke output -> ${OUT}"
echo "    - metrics.json"
echo "    - predictions.jsonl"
echo "    - traces.jsonl"
