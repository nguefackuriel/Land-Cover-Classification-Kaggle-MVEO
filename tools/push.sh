#!/bin/bash
# Copy the working folder into the repo and push it. usage: tools/push.sh ["commit message"]
set -e
cd "$(dirname "$0")/.."
python3 tools/sync_from_work.py > /dev/null
git add -A
if git diff --cached --quiet; then echo "nothing new"; exit 0; fi
git commit -q -m "${1:-Sync logs, checkpoints and notes from the working folder}"
git push -q origin main
git log --oneline -1
