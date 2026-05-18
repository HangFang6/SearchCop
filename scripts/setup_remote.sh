#!/usr/bin/env bash
# SearchCop · Remote H20 setup script
# Usage: bash scripts/setup_remote.sh
set -euo pipefail

echo "=========================================="
echo " SearchCop remote bootstrap"
echo "=========================================="

# --- 0. Sanity check ---
if ! command -v conda >/dev/null 2>&1; then
    echo "[ERR] conda not found. Install Miniconda first." >&2
    exit 1
fi

# --- 1. Create conda env ---
if conda env list | grep -q '^searchcop\s'; then
    echo "[*] conda env 'searchcop' already exists, updating..."
    conda env update -f environment.yml --prune
else
    echo "[*] Creating conda env 'searchcop' ..."
    conda env create -f environment.yml
fi

# --- 2. Activate ---
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate searchcop

# --- 3. Sanity: GPU + torch ---
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available(), 'n_gpu:', torch.cuda.device_count())"

# --- 4. Directories ---
mkdir -p weights results logs cache data
touch results/.gitkeep

# --- 5. .env hint ---
if [ ! -f .env ]; then
    if [ -f .env.example ]; then
        echo "[*] .env not found. Run: cp .env.example .env  (Doubao credentials are pre-filled)"
    fi
fi

echo ""
echo "=========================================="
echo "[DONE] Setup complete."
echo "Next steps:"
echo "  1. cp .env.example .env && vim .env"
echo "  2. Download datasets per README_REMOTE.md §2"
echo "  3. Download/symlink model weights per §3"
echo "  4. pytest tests/ -v"
echo "=========================================="
