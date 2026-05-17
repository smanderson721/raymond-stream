# raymond-stream

Daily 3-hour automated chess broadcast on Twitch.

## What it does

Every day at 22:00 UTC the workflow:
1. Starts the Oracle "bob" instance.
2. Generates 40 asymmetric Stockfish games on bob.
3. Builds per-ply Reginald commentary via Gemini Pro + Gemini TTS, then
   word-aligns the audio with faster-whisper.
4. Restarts the `tuna-display / tuna-producer / tuna-stream` systemd
   units pointing at the new session — Twitch goes live.
5. Sleeps until the session's total speech length has played out
   (~3 hours).
6. Stops the stream services + the bob instance.

All heavy work runs on bob; this repo just orchestrates.

## Repo layout

```
run_session.sh                            ← orchestrator
.github/workflows/daily-stream.yml        ← cron trigger
_chess_asymmetric_games.py                ← uploaded to bob
production/live/build_session_commentary.py
production/live/build_commentary_gemini.py
production/live/stockfish_analyzer.py
production/live/align_words.py
production/live/fix_word_silence_offset.py
research_output/chess/openings_db.json
```

## Required GitHub secrets

| Secret | Source |
|---|---|
| `GEMINI_API_KEY` | Google AI Studio |
| `OCI_CONFIG` | contents of `~/.oci/config` from the local Mac |
| `OCI_API_KEY` | contents of `~/.oci/oci_api_key.pem` |
| `BOB_SSH_PRIVATE_KEY` | contents of `~/.ssh/oracle_key` |

The Twitch stream key already lives in `/home/ubuntu/stream/.env` on bob,
so it is **not** a GitHub secret.

## Manual trigger

```bash
gh workflow run daily-stream.yml -f num_games=20 -f voice=Enceladus
```
