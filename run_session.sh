#!/usr/bin/env bash
# Daily chess-stream orchestrator (runs on GitHub Actions, drives bob).
#
# 1. Start bob (Oracle Ampere "tuna" instance) via OCI CLI.
# 2. rsync the build scripts from this repo to bob.
# 3. Generate N asymmetric games on bob (Stockfish vs capped Stockfish).
# 4. Build per-ply Reginald commentary via Gemini Pro + Gemini TTS.
# 5. Align words via faster-whisper.
# 6. Activate the new session in /home/ubuntu/stream/.env and restart
#    tuna-display / tuna-producer / tuna-stream → Twitch goes live.
# 7. Sleep until the session's total speech time has elapsed.
# 8. Stop tuna-stream + tuna-producer (Twitch goes offline).
# 9. Stop the bob instance to save money.
#
# Required env (passed in by the workflow):
#   GEMINI_API_KEY        — for commentary + TTS
#   OCI CLI configured    — via ~/.oci/config + key file written from secrets
#   SSH key at ~/.ssh/oracle_key — written from secret BOB_SSH_PRIVATE_KEY
#
# Optional env:
#   NUM_GAMES (default 40)
#   VOICE     (default Enceladus)
#   CONCURRENCY (default 8)
#   SKILL     (default 6)
#   DEPTH     (default 3)
#   BOB_SSH_IP (default 129.80.114.39)

set -euo pipefail

NUM_GAMES="${NUM_GAMES:-40}"
VOICE="${VOICE:-Enceladus}"
CONCURRENCY="${CONCURRENCY:-8}"
SKILL="${SKILL:-6}"
DEPTH="${DEPTH:-3}"

SESSION_ID="$(date +%Y%m%d_%H%M%S)"
STRONG_WHITE=$(( (NUM_GAMES + 1) / 2 ))
STRONG_BLACK=$(( NUM_GAMES - STRONG_WHITE ))

INSTANCE_ID="ocid1.instance.oc1.iad.anuwcljtyqvmqyacssu6wqlq6ebzrm6dqfxsqxvjdhz2bvfibdxgltm5xqqq"
SSH_KEY="$HOME/.ssh/oracle_key"
SSH_USER="ubuntu"
SSH_IP="${BOB_SSH_IP:-129.80.114.39}"
SSH_OPTS="-i $SSH_KEY -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30"
REMOTE_ROOT="/home/ubuntu/stream"
REMOTE_GAMES_DIR="research_output/chess/asymmetric_games"
PY="$REMOTE_ROOT/.venv/bin/python"

export SUPPRESS_LABEL_WARNING=True

echo "════════════════════════════════════════════════════════════════════"
echo " Daily chess stream — session $SESSION_ID"
echo "   $NUM_GAMES games  (white=$STRONG_WHITE  black=$STRONG_BLACK)"
echo "   voice=$VOICE concurrency=$CONCURRENCY skill=$SKILL depth=$DEPTH"
echo "════════════════════════════════════════════════════════════════════"

T0=$(date +%s)

# ── 1. Start bob ──────────────────────────────────────────────────────
STATE=$(oci compute instance get --instance-id "$INSTANCE_ID" \
    --query 'data."lifecycle-state"' --raw-output 2>/dev/null)
echo "[1/9] bob state: $STATE"
case "$STATE" in
    STOPPED)  oci compute instance action --instance-id "$INSTANCE_ID" --action START > /dev/null ;;
    STOPPING) while [[ "$STATE" != "STOPPED" ]]; do sleep 5; STATE=$(oci compute instance get --instance-id "$INSTANCE_ID" --query 'data."lifecycle-state"' --raw-output); done
              oci compute instance action --instance-id "$INSTANCE_ID" --action START > /dev/null ;;
    RUNNING|STARTING) ;;
    *) echo "Unexpected state $STATE"; exit 1 ;;
esac
echo -n "      waiting for RUNNING"
while true; do
    STATE=$(oci compute instance get --instance-id "$INSTANCE_ID" --query 'data."lifecycle-state"' --raw-output)
    [[ "$STATE" == "RUNNING" ]] && { echo " ✓"; break; }
    echo -n "."; sleep 3
done

# IP may have rotated if instance was stopped + reassigned.
FRESH_IP=$(oci compute instance list-vnics --instance-id "$INSTANCE_ID" \
    --query 'data[0]."public-ip"' --raw-output 2>/dev/null || echo "")
if [[ -n "$FRESH_IP" && "$FRESH_IP" != "$SSH_IP" ]]; then
    echo "      IP rotated: $SSH_IP → $FRESH_IP"
    SSH_IP="$FRESH_IP"
fi

echo -n "      waiting for SSH"
for i in $(seq 1 60); do
    if ssh $SSH_OPTS $SSH_USER@$SSH_IP true 2>/dev/null; then echo " ✓"; break; fi
    echo -n "."; sleep 3
    if [[ $i -eq 60 ]]; then echo " ✗ ssh never came up"; exit 1; fi
done

# ── 2. Push build scripts to bob ─────────────────────────────────────
echo "[2/9] pushing code to bob"
RSYNC="rsync -az -e \"ssh $SSH_OPTS\""
eval $RSYNC _chess_asymmetric_games.py \
    "$SSH_USER@$SSH_IP:$REMOTE_ROOT/_chess_asymmetric_games.py"
eval $RSYNC production/live/build_commentary_gemini.py \
    production/live/build_session_commentary.py \
    production/live/stockfish_analyzer.py \
    production/live/align_words.py \
    production/live/fix_word_silence_offset.py \
    "$SSH_USER@$SSH_IP:$REMOTE_ROOT/production/live/"
eval $RSYNC research_output/chess/openings_db.json \
    "$SSH_USER@$SSH_IP:$REMOTE_ROOT/research_output/chess/openings_db.json"

# ── 3. Archive previous batch + generate new games on bob ────────────
echo "[3/9] archiving previous batch + generating $NUM_GAMES games"
ssh $SSH_OPTS $SSH_USER@$SSH_IP "
set -e
cd $REMOTE_ROOT/$REMOTE_GAMES_DIR
mkdir -p _played
PREV_SID=\$(ls game_*.json 2>/dev/null | head -1 \
    | xargs -I{} $PY -c \"import json,sys; d=json.load(open(sys.argv[1])); print(d.get('session_id','archived'))\" {} 2>/dev/null \
    || true)
if [ -z \"\$PREV_SID\" ]; then PREV_SID=archived_\$(date +%Y%m%d_%H%M%S); fi
if ls game_*.json >/dev/null 2>&1; then
    mkdir -p _played/\$PREV_SID
    mv game_*.json _played/\$PREV_SID/
fi
cd $REMOTE_ROOT
$PY -u _chess_asymmetric_games.py \
    --num-games $NUM_GAMES \
    --strong-as-white $STRONG_WHITE \
    --strong-as-black $STRONG_BLACK \
    --weak-elo $((1000 + SKILL * 100)) \
    --strong-time 8.0 \
    --threads-per-game 6 \
    --hash-mb 1024 \
    --parallel 8 \
    --weak-engine-kind stockfish_capped \
    --weak-skill $SKILL \
    --weak-depth $DEPTH \
    --opening-book $REMOTE_ROOT/research_output/chess/openings_db.json \
    --output-dir $REMOTE_ROOT/$REMOTE_GAMES_DIR 2>&1 | tail -50
"

# ── 4. Build commentary on bob ───────────────────────────────────────
echo "[4/9] building Gemini commentary + TTS for session $SESSION_ID"
ssh $SSH_OPTS $SSH_USER@$SSH_IP "
set -e
cd $REMOTE_ROOT/narration_library
ARCH=_archive_\$(date +%Y%m%d_%H%M%S)
SESSIONS=\$(ls -d session_* 2>/dev/null || true)
if [ -n \"\$SESSIONS\" ]; then
    mkdir -p \$ARCH
    mv \$SESSIONS \$ARCH/
    if [ -f library.json ]; then mv library.json \$ARCH/library.json; fi
fi
"
ssh $SSH_OPTS $SSH_USER@$SSH_IP "
set -e
cd $REMOTE_ROOT
set -a
. /home/ubuntu/stream/.env
export GEMINI_API_KEY='$GEMINI_API_KEY'
set +a
$PY production/live/build_session_commentary.py \
    --games-dir $REMOTE_GAMES_DIR \
    --session-id $SESSION_ID \
    --voice $VOICE \
    --concurrency $CONCURRENCY 2>&1 | tail -150
"

# ── 5. Word alignment ────────────────────────────────────────────────
echo "[5/9] aligning words via faster-whisper"
ssh $SSH_OPTS $SSH_USER@$SSH_IP "
set -e
cd $REMOTE_ROOT
$PY production/live/align_words.py \
    --session-dir narration_library/session_$SESSION_ID \
    --voice $VOICE 2>&1 | tail -20
$PY production/live/fix_word_silence_offset.py \
    --session-dir narration_library/session_$SESSION_ID \
    --voice $VOICE 2>&1 | tail -5
"

# ── 6. Activate session + go live ────────────────────────────────────
echo "[6/9] activating session and restarting stream services"
ssh $SSH_OPTS $SSH_USER@$SSH_IP "
sed -i 's|^SESSION_ID=.*|SESSION_ID=$SESSION_ID|' /home/ubuntu/stream/.env
grep -q '^STREAM_MODE=' /home/ubuntu/stream/.env \
    && sed -i 's|^STREAM_MODE=.*|STREAM_MODE=chess|' /home/ubuntu/stream/.env \
    || echo 'STREAM_MODE=chess' >> /home/ubuntu/stream/.env
sudo systemctl restart tuna-display tuna-producer tuna-stream
sleep 8
systemctl is-active tuna-display tuna-producer tuna-stream
"

# ── 7. Pull session length, then sleep ───────────────────────────────
TOTAL_SECONDS=$(ssh $SSH_OPTS $SSH_USER@$SSH_IP \
    "$PY -c \"import json,pathlib; d=json.loads(pathlib.Path('$REMOTE_ROOT/narration_library/session_$SESSION_ID/_session.json').read_text()); print(int(d.get('total_seconds',0)))\"")
SLEEP_SECONDS=$((TOTAL_SECONDS * 110 / 100 + 60))
T1=$(date +%s)
BUILD_ELAPSED=$((T1 - T0))
echo "[7/9] build elapsed ${BUILD_ELAPSED}s; stream will run ~${SLEEP_SECONDS}s ($(echo "$SLEEP_SECONDS/60" | bc) min)"
sleep "$SLEEP_SECONDS"

# ── 8. Stop streaming services ───────────────────────────────────────
echo "[8/9] stopping tuna-stream + tuna-producer (Twitch goes offline)"
ssh $SSH_OPTS $SSH_USER@$SSH_IP \
    'sudo systemctl stop tuna-stream tuna-producer; sudo systemctl is-active tuna-stream tuna-producer || true'

# ── 9. Stop bob ──────────────────────────────────────────────────────
echo "[9/9] stopping bob"
oci compute instance action --instance-id "$INSTANCE_ID" --action STOP > /dev/null

T2=$(date +%s)
echo "✓ session complete — total wall time $((T2 - T0))s ($(echo "($T2 - $T0)/60" | bc) min)"
echo "  session_id = $SESSION_ID"
