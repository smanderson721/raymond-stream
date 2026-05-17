"""Run faster-whisper word-timestamp transcription over a session's
commentary MP3s.

For every `<line_id>.txt` + `<line_id>__<voice>.mp3` pair under
`narration_library/session_<id>/`, write a `<line_id>__<voice>.words.json`
sidecar containing per-word start/end timestamps (relative to the clip).

The word timings are used by the live stream frontend (`chess_live.html`)
to drive the fish's mouth in sync with the spoken commentary.

Usage:
    python production/live/align_words.py --session-dir narration_library/session_xxx
    python production/live/align_words.py --session-dir ... --voice Enceladus
    python production/live/align_words.py --session-dir ... --force   # overwrite

Sidecar format:
    {
      "duration": 4.231,
      "language": "en",
      "voice": "Enceladus",
      "text": "I'll snap that off, thank you very much.",
      "words": [
        {"word": "I'll",  "start": 0.04,  "end": 0.21},
        ...
      ]
    }

faster-whisper transcribes the audio and returns word timestamps via
`word_timestamps=True`. The known TTS text is passed as `initial_prompt`
to bias the decoder toward the correct vocabulary.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_MODEL = "base.en"

_MODEL_CACHE = {}


def _load_model(model_name: str, device: str = "cpu", compute_type: str = "int8"):
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]
    from faster_whisper import WhisperModel
    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    _MODEL_CACHE[model_name] = model
    return model


def _align_one(model, audio_path: Path, text: str) -> dict:
    segments, info = model.transcribe(
        str(audio_path),
        word_timestamps=True,
        initial_prompt=text.strip(),
        beam_size=1,
        vad_filter=False,
    )
    words: list[dict] = []
    for seg in segments:
        for w in (seg.words or []):
            wd = (w.word or "").strip()
            if not wd:
                continue
            if w.start is None or w.end is None:
                continue
            words.append({
                "word": wd,
                "start": round(float(w.start), 3),
                "end": round(float(w.end), 3),
            })
    # Compensate for TTS leading silence: faster-whisper snaps word[0].start
    # to 0.0 even when the actual first phoneme begins ~250 ms in (Gemini TTS
    # adds natural lead silence). Without this shift the jaw animator opens
    # the mouth before any sound is heard. See fix_word_silence_offset.py.
    silence_offset = _leading_silence_seconds(audio_path)
    if silence_offset > 0.02:
        for w in words:
            w["start"] = round(w["start"] + silence_offset, 3)
            w["end"] = round(w["end"] + silence_offset, 3)
    return {
        "duration": round(float(info.duration), 3) if info.duration else 0.0,
        "language": info.language or "en",
        "text": text.strip(),
        "silence_offset": round(silence_offset, 3) if silence_offset > 0.02 else 0.0,
        "words": words,
    }


_SILENCE_END_RE = __import__("re").compile(r"silence_end:\s*([0-9.]+)")


def _leading_silence_seconds(mp3: Path, threshold_db: int = -40) -> float:
    """Return seconds of leading silence at the start of `mp3`. 0.0 if none."""
    import subprocess
    out = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-nostats", "-i", str(mp3),
            "-af", f"silencedetect=noise={threshold_db}dB:d=0.05",
            "-f", "null", "-",
        ],
        capture_output=True, text=True,
    ).stderr
    first_block = out.split("silence_start")[1] if "silence_start" in out else ""
    if not first_block.lstrip().startswith(":"):
        return 0.0
    try:
        first_start = float(first_block.split("\n")[0].strip(": ").split()[0])
    except (ValueError, IndexError):
        return 0.0
    if first_start > 0.05:
        return 0.0
    m = _SILENCE_END_RE.search(first_block)
    if not m:
        return 0.0
    return float(m.group(1))


def align_session(
    session_dir: Path,
    voice: str = "Enceladus",
    model_name: str = DEFAULT_MODEL,
    force: bool = False,
) -> tuple[int, int, int]:
    txt_files = sorted(session_dir.rglob("L_*.txt"))
    if not txt_files:
        print(f"[align] no L_*.txt files under {session_dir}")
        return (0, 0, 0)

    pairs: list[tuple[Path, Path, Path]] = []
    for txt in txt_files:
        line_id = txt.stem
        mp3 = txt.with_name(f"{line_id}__{voice}.mp3")
        words_json = txt.with_name(f"{line_id}__{voice}.words.json")
        if not mp3.exists():
            continue
        if words_json.exists() and not force:
            continue
        pairs.append((txt, mp3, words_json))

    if not pairs:
        print(f"[align] nothing to do (use --force to re-align)")
        return (0, len(txt_files), 0)

    print(f"[align] loading faster-whisper model={model_name} (CPU, int8)…")
    model = _load_model(model_name)
    print(f"[align] aligning {len(pairs)} clip(s) for voice={voice}")

    ok = 0
    failed = 0
    for i, (txt, mp3, words_json) in enumerate(pairs, start=1):
        text = txt.read_text().strip()
        try:
            sidecar = _align_one(model, mp3, text)
        except Exception as exc:
            print(f"  [{i}/{len(pairs)}] {mp3.name}: ✗ {exc}")
            failed += 1
            continue
        sidecar["voice"] = voice
        words_json.write_text(json.dumps(sidecar, indent=2))
        ok += 1
        n_words = len(sidecar["words"])
        print(
            f"  [{i}/{len(pairs)}] {mp3.name}: ✓ {n_words} words "
            f"in {sidecar['duration']:.2f}s"
        )
    skipped = len(txt_files) - len(pairs)
    return (ok, skipped, failed)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--session-dir", required=True)
    p.add_argument("--voice", default="Enceladus")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    session_dir = Path(args.session_dir).resolve()
    if not session_dir.is_dir():
        sys.exit(f"not a directory: {session_dir}")

    ok, skipped, failed = align_session(
        session_dir, voice=args.voice, model_name=args.model, force=args.force
    )
    print(f"\n[align] done: {ok} aligned, {skipped} skipped, {failed} failed")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
