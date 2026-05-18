#!/usr/bin/env bash
# Pull the latest session_<id>/ directory from bob to the local Mac
# so live_server.py can serve it in the pitches.html Stream tab.
#
# Usage:
#   ./pull_latest_session.sh          # pulls newest session
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

echo "Pulling $SID from $BOB_HOST …"
mkdir -p "$LOCAL_ROOT/narration_library"
rsync -az --info=progress2 \
  -e "ssh ${SSH_OPTS[*]}" \
  "$BOB_USER@$BOB_HOST:$REMOTE_ROOT/narration_library/$SID/" \
  "$LOCAL_ROOT/narration_library/$SID/"

echo "Done. Open the Stream tab in pitches.html — live_server.py auto-picks the newest session."
