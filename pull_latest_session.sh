#!/usr/bin/env bash
# Pull the latest session_<id>/ directory AND the asymmetric_games/
# from bob to the local Mac so:
#   - live_server.py serves the session in the pitches.html Stream tab
#   - the Engine Games tab shows the streamed games
#
# Usage:
#   ./pull_latest_session.sh          # pulls newest session + all games
#   ./pull_latest_session.sh <id>     # pulls a specific session

set -euo pipefail

BOB_USER="${BOB_USER:-ubuntu}"
BOB_HOST="${BOB_HOST:-129.80.114.39}"
BOB_KEY="${BOB_KEY:-$HOME/.ssh/oracle_key}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/ubuntu/stream}"
LOCAL_ROOT="${LOCAL_ROOT:-/Users/billnye/Desktop/video essays 2}"

SSH_OPTS=(-i "$BOB_KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null)

SID="${1:-}"
if [[ -z "$SID" ]]; then
  SID=$(ssh "${SSH_OPTS[@]}" "$BOB_USER@$BOB_HOST" \
    "ls -1d $REMOTE_ROOT/narration_library/session_* 2>/dev/null | sort | tail -1 | xargs -n1 basename")
fi
[[ -z "$SID" ]] && { echo "No session found on bob"; exit 1; }

echo "── Pulling session $SID ──"
mkdir -p "$LOCAL_ROOT/narration_library"
rsync -az --info=progress2 \
  -e "ssh ${SSH_OPTS[*]}" \
  "$BOB_USER@$BOB_HOST:$REMOTE_ROOT/narration_library/$SID/" \
  "$LOCAL_ROOT/narration_library/$SID/"

echo "── Pulling asymmetric_games ──"
LOCAL_GAMES="$LOCAL_ROOT/research_output/chess/asymmetric_games"
mkdir -p "$LOCAL_GAMES"
rsync -az --info=progress2 \
  -e "ssh ${SSH_OPTS[*]}" \
  "$BOB_USER@$BOB_HOST:$REMOTE_ROOT/research_output/chess/asymmetric_games/" \
  "$LOCAL_GAMES/"

echo "── Rebuilding _index.json ──"
python3 - "$LOCAL_GAMES" << 'PY'
import json, os, sys, glob
out_dir = sys.argv[1]
files = sorted(os.path.basename(p) for p in glob.glob(os.path.join(out_dir, "game_*.json")))
with open(os.path.join(out_dir, "_index.json"), "w") as f:
    json.dump({"games": files}, f, indent=2)
print(f"Wrote _index.json with {len(files)} games")
PY

echo "Done. Refresh pitches.html — Stream tab plays $SID, Engine Games tab lists the new games."

