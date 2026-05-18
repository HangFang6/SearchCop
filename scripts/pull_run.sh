#!/usr/bin/env bash
# Remote-side helper: pull latest code, then optionally run a script.
# Usage:
#   bash scripts/pull_run.sh                                  # pull only
#   bash scripts/pull_run.sh pytest tests/ -v                 # pull then test
#   bash scripts/pull_run.sh bash scripts/run_exp_mevid_baseline.sh
set -euo pipefail

echo "[*] git pull --rebase ..."
git pull --rebase

if [ $# -gt 0 ]; then
    echo "[*] Running: $*"
    "$@"
fi
