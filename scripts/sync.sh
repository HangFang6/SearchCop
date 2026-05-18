#!/usr/bin/env bash
# Local-side helper: stage all + commit + push.
# Usage:
#   bash scripts/sync.sh "wip: tweak planner prompt"
#   bash scripts/sync.sh                # auto-message with timestamp
set -euo pipefail

MSG="${1:-wip: $(date +%Y-%m-%d_%H:%M)}"
git add -A
if git diff --cached --quiet; then
    echo "[*] Nothing to commit."
    exit 0
fi
git commit -m "${MSG}"
git push
echo "[OK] Pushed: ${MSG}"
