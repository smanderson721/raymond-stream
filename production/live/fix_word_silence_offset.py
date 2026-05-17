"""Fix whisper word-timestamp drift caused by TTS leading silence.

Gemini TTS prepends ~250 ms of natural silence to each clip. faster-whisper
snaps `words[0].start` to 0.0 instead of to the actual first phoneme,
which makes the jaw animator open the mouth ~250 ms before any sound is
heard.

This script measures the leading silence in each `<line>__<voice>.mp3`
via ffmpeg `silencedetect` and shifts every word timestamp in the
matching `<line>__<voice>.words.json` by that amount.

Idempotent: a `silence_offset` field is written to the sidecar so that
repeat runs detect the existing shift and skip.

Usage:
    python production/live/fix_word_silence_offset.py \\
        --session-dir narration_library/session_test_20260430_185628
    python production/live/fix_word_silence_offset.py --session-dir ... --dry-run
    python production/live/fix_word_silence_offset.py --session-dir ... --revert
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

_SILENCE_END_RE = re.compile(r"silence_end:\s*([0-9.]+)")


def _leading_silence(mp3: Path, threshold_db: int = -40) -> float:
    """Return seconds of leading silence at the start of the clip,
    or 0.0 if no leading-silence segment found.
    """
    out = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-nostats", "-i", str(mp3),
            "-af", f"silencedetect=noise={threshold_db}dB:d=0.05",
            "-f", "null", "-",
        ],
        capture_output=True, text=True,
    ).stderr
    # The silencedetect output emits lines like:
    #   silence_start: 0
    #   silence_end:   0.248542 | silence_duration: 0.248542
    # The leading-silence end is the FIRST silence_end whose preceding
    # silence_start was at 0 (or near 0).
    first_block = out.split("silence_start")[1] if "silence_start" in out else ""
    if not first_block.lstrip().startswith(":"):
        return 0.0
    # Check the start time of the first silence segment:
    try:
        first_start = float(first_block.split("\n")[0].strip(": ").split()[0])
    except (ValueError, IndexError):
        return 0.0
    if first_start > 0.05:  # leading silence must START at (or near) 0
        return 0.0
    m = _SILENCE_END_RE.search(first_block)
    if not m:
        return 0.0
    return float(m.group(1))


def fix_session(session_dir: Path, voice: str,
                dry_run: bool = False, revert: bool = False) -> None:
    word_files = sorted(session_dir.rglob(f"L_*__{voice}.words.json"))
    if not word_files:
        sys.exit(f"no {voice} words.json files under {session_dir}")

    shifted = 0
    skipped = 0
    reverted = 0
    no_silence = 0
    for wj in word_files:
        data = json.loads(wj.read_text())
        prev_offset = float(data.get("silence_offset") or 0.0)

        if revert:
            if prev_offset <= 0:
                skipped += 1
                continue
            for w in data.get("words", []):
                w["start"] = round(float(w["start"]) - prev_offset, 3)
                w["end"] = round(float(w["end"]) - prev_offset, 3)
            data.pop("silence_offset", None)
            if not dry_run:
                wj.write_text(json.dumps(data, indent=2))
            print(f"  reverted {-prev_offset*1000:+.0f}ms  {wj.name}")
            reverted += 1
            continue

        if prev_offset > 0:
            skipped += 1
            continue

        mp3 = wj.with_name(wj.name.replace(".words.json", ".mp3"))
        if not mp3.exists():
            skipped += 1
            continue

        offset = _leading_silence(mp3)
        if offset <= 0.02:  # < 20 ms is just noise — don't bother
            no_silence += 1
            continue

        for w in data.get("words", []):
            w["start"] = round(float(w["start"]) + offset, 3)
            w["end"] = round(float(w["end"]) + offset, 3)
        data["silence_offset"] = round(offset, 3)
        if not dry_run:
            wj.write_text(json.dumps(data, indent=2))
        print(f"  +{offset*1000:>4.0f}ms  {wj.name}")
        shifted += 1

    print()
    if revert:
        print(f"[fix] reverted {reverted}, skipped {skipped}")
    else:
        print(f"[fix] shifted {shifted}, skipped {skipped} (already shifted), "
              f"no-silence {no_silence}")
        if dry_run:
            print("[fix] DRY RUN — no files modified")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--session-dir", required=True)
    p.add_argument("--voice", default="Enceladus")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--revert", action="store_true",
                   help="undo a previous shift (uses sidecar's silence_offset field)")
    args = p.parse_args()

    session_dir = Path(args.session_dir).resolve()
    if not session_dir.is_dir():
        sys.exit(f"not a directory: {session_dir}")

    fix_session(session_dir, voice=args.voice,
                dry_run=args.dry_run, revert=args.revert)


if __name__ == "__main__":
    main()
