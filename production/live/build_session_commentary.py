"""Generate per-game Reginald commentary for a full session of asymmetric games.

Reads game JSON files produced by `_chess_asymmetric_games.py` from a
directory, then for each game extracts the PGN movetext + book ply count
and invokes `build_commentary_gemini.write_comments` + TTS to produce
per-ply audio under `narration_library/<session_category>/game_<NN>__<id>/`.

The session position (first | middle | last | only) is computed
automatically from each game's index in the session, controlling whether
Reginald greets / signs off.

Usage:
    python production/live/build_session_commentary.py \\
        --games-dir research_output/chess/asymmetric_games \\
        --session-id 20260430_capped \\
        --voice Enceladus

Requires GEMINI_API_KEY in env. Idempotent: existing mp3s are skipped.
"""
from __future__ import annotations

import argparse
import asyncio
import io
import json
import os
import sys
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import chess.pgn  # type: ignore

from google import genai
from google.genai import types  # noqa: F401

from production.live import build_commentary_gemini as bcg


def extract_pgn_movetext(game: dict) -> str:
    """Return the PGN movetext (no headers) suitable for write_comments."""
    pgn = game.get("pgn") or ""
    if not pgn.strip():
        # Build movetext from `moves` list
        sans = [m["san"] for m in game.get("moves", [])]
        out = []
        for i, san in enumerate(sans):
            if i % 2 == 0:
                out.append(f"{i // 2 + 1}.{san}")
            else:
                out.append(san)
        return " ".join(out) + (" " + game.get("result", "") if game.get("result") else "")

    pg = chess.pgn.read_game(io.StringIO(pgn))
    if pg is None:
        return pgn
    exporter = chess.pgn.StringExporter(headers=False, variations=False, comments=False)
    return pg.accept(exporter).strip()


def book_ply_count(game: dict) -> int:
    """Count plies marked book=True in the moves array (preferred), else
    fall back to opening.plies, else 0."""
    moves = game.get("moves") or []
    n = sum(1 for m in moves if m.get("book"))
    if n > 0:
        return n
    op = game.get("opening") or {}
    return int(op.get("plies") or 0)


def session_position(idx: int, total: int) -> str:
    if total == 1:
        return "only"
    if idx == 0:
        return "first"
    if idx == total - 1:
        return "last"
    return "middle"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--games-dir",
        default="research_output/chess/asymmetric_games",
        help="Directory of game_*.json files",
    )
    p.add_argument(
        "--session-id",
        required=True,
        help="Session label, used in category names",
    )
    p.add_argument(
        "--voice",
        default="Enceladus",
        help="Single Gemini TTS voice for the session",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit to first N games (debug)",
    )
    p.add_argument(
        "--game-id-filter",
        default=None,
        help="Only process games whose game_id contains this substring",
    )
    p.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help=(
            "Maximum simultaneous Gemini TTS calls per game (default 8). "
            "Set to 1 for purely sequential (legacy) behavior."
        ),
    )
    args = p.parse_args()

    if not os.environ.get("GEMINI_API_KEY"):
        sys.exit("GEMINI_API_KEY not set in environment")

    games_dir = (REPO_ROOT / args.games_dir).resolve()
    if not games_dir.exists():
        sys.exit(f"Games dir does not exist: {games_dir}")

    game_paths = sorted(p for p in games_dir.glob("game_*.json"))
    if args.game_id_filter:
        game_paths = [p for p in game_paths if args.game_id_filter in p.name]
    if args.limit:
        game_paths = game_paths[: args.limit]
    if not game_paths:
        sys.exit(f"No games found in {games_dir}")

    print(f"▶ Session {args.session_id}: {len(game_paths)} games, voice={args.voice}")

    client = genai.Client()

    session_root = bcg.NARRATION_DIR / f"session_{args.session_id}"
    session_root.mkdir(parents=True, exist_ok=True)

    lib_path = bcg.NARRATION_DIR / "library.json"
    lib = json.loads(lib_path.read_text()) if lib_path.exists() else {}
    lib.setdefault(
        "_meta",
        {
            "voice": "Gemini 3.1 Flash TTS — Enceladus",
            "personality": (
                "Reginald T. — British gentleman chess streamer. RP accent. "
                "Pensive, intellectual, mildly amused, faintly condescending."
            ),
            "audio": {
                "speedup": bcg.SPEEDUP,
                "channels": "stereo (mono signal duplicated to L+R)",
                "max_clip_seconds": bcg.MAX_CLIP_SECONDS,
                "target_wpm_after_speedup": bcg.WPM_AFTER_SPEEDUP,
            },
        },
    )

    sessions_summary: list[dict] = []

    for idx, gpath in enumerate(game_paths):
        game = json.loads(gpath.read_text())
        gid = game.get("game_id", gpath.stem.replace("game_", ""))
        position = session_position(idx, len(game_paths))
        movetext = extract_pgn_movetext(game)
        book_plies = book_ply_count(game)
        opening = (game.get("opening") or {}).get("name") or "(no book)"
        eco = (game.get("opening") or {}).get("eco") or ""

        category = f"session_{args.session_id}/game_{idx + 1:02d}__{gid}"
        cat_dir = bcg.NARRATION_DIR / category
        cat_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"\n══ Game {idx + 1}/{len(game_paths)} ({position}) — "
            f"{gid} — {opening} [{eco}] — {book_plies} book plies ══"
        )

        # Resume support: if this game already has a fully-built library entry
        # with audio for every line, skip it. Avoids regenerating fresh UUIDs
        # (which orphans the previous mp3s) when retrying a partial session.
        existing = lib.get(category)
        if existing and existing.get("lines"):
            all_have_audio = all(
                (cat_dir / f"{ln['id']}__{args.voice}.mp3").exists()
                for ln in existing["lines"]
            )
            if all_have_audio:
                print(f"  ⏭  already built ({len(existing['lines'])} lines), skipping")
                sessions_summary.append(
                    {
                        "game_index": idx + 1,
                        "game_id": gid,
                        "category": category,
                        "session_position": position,
                        "opening": opening,
                        "eco": eco,
                        "result": game.get("result"),
                        "lines": len(existing["lines"]),
                        "total_seconds": round(
                            sum(
                                d
                                for ln in existing["lines"]
                                for d in (ln.get("durations") or {}).values()
                            ),
                            2,
                        ),
                    }
                )
                continue

        # Pre-analyze every ply with Stockfish so the LLM has real chess
        # ground truth (eval, classification, alternatives, hanging pieces)
        # rather than inventing justifications. See stockfish_analyzer.py.
        try:
            from production.live.stockfish_analyzer import analyze_game
            print("  running Stockfish analysis (depth 18)…")
            engine_analysis = analyze_game(
                movetext, depth=18, multipv=3, threads=4, hash_mb=256,
            )
            print(f"  analyzed {len(engine_analysis)} plies")
        except Exception as exc:
            print(f"  ⚠ Stockfish analysis failed: {exc} — falling back to LLM-only")
            engine_analysis = None

        try:
            comments = bcg.write_comments(
                client,
                movetext,
                book_plies,
                position,
                streamer_color=game.get("strong_color", "white"),
                engine_analysis=engine_analysis,
            )
        except Exception as exc:
            print(f"  ✗ Failed to write comments for {gid}: {exc}")
            continue

        print(f"  got {len(comments)} comments")

        # Build all line metadata up-front so we can launch TTS in parallel.
        line_jobs: list[dict] = []
        for cidx, c in enumerate(comments, start=1):
            line_id = f"L_{uuid.uuid4().hex[:8]}"
            text = c["text"].strip()
            ply = int(c["ply"])
            is_book = ply <= book_plies
            (cat_dir / f"{line_id}.txt").write_text(text)
            n_words = len(text.split())
            wav = cat_dir / f"{line_id}__{args.voice}.wav"
            mp3 = cat_dir / f"{line_id}__{args.voice}.mp3"
            line_jobs.append(
                {
                    "idx": cidx,
                    "line_id": line_id,
                    "text": text,
                    "ply": ply,
                    "is_book": is_book,
                    "n_words": n_words,
                    "wav": wav,
                    "mp3": mp3,
                }
            )

        # ── Concurrent TTS synthesis ───────────────────────────────────
        # Each clip is an HTTP round-trip to Google (~5-7s).  Sequential
        # this is the dominant cost; with semaphore-bounded asyncio we
        # collapse it to ~ceil(N/concurrency) × per-call time.
        sem = asyncio.Semaphore(max(1, args.concurrency))
        total_jobs = len(line_jobs)

        async def _synth_job(job: dict) -> tuple[dict, float | None, str | None]:
            mp3 = job["mp3"]
            wav = job["wav"]
            if mp3.exists():
                return job, round(bcg.mp3_duration(mp3), 3), "skip"
            async with sem:
                try:
                    await bcg.synth_one_async(client, args.voice, job["text"], wav)
                    # ffmpeg conversion is sync but cheap (<1s).  Run in
                    # default executor to avoid blocking the event loop.
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(None, bcg.wav_to_mp3, wav, mp3)
                    if wav.exists():
                        try:
                            wav.unlink()
                        except Exception:
                            pass
                    dur = round(bcg.mp3_duration(mp3), 3)
                    return job, dur, None
                except Exception as exc:
                    if wav.exists():
                        try:
                            wav.unlink()
                        except Exception:
                            pass
                    return job, None, str(exc)

        if total_jobs:
            print(
                f"  ▶ synthesizing {total_jobs} lines "
                f"with concurrency={args.concurrency}…"
            )

            async def _run_all():
                return await asyncio.gather(*[_synth_job(j) for j in line_jobs])

            results = asyncio.run(_run_all())
        else:
            results = []

        lib_lines: list[dict] = []
        ok_count = 0
        for job, dur, err in results:
            durations: dict[str, float] = {}
            tag = ""
            if err:
                tag = f"  ✗ FAILED: {err}"
            elif dur is None:
                tag = "  ✗ no audio"
            else:
                durations[args.voice] = dur
                ok_count += 1
                tag = f"  ✓ {dur:.2f}s"
            print(
                f"  [{job['idx']}/{total_jobs}] ply {job['ply']}"
                f"{' (book)' if job['is_book'] else ''} "
                f"({job['n_words']}w): {job['text'][:60]}…{tag}"
            )
            lib_lines.append(
                {
                    "id": job["line_id"],
                    "ply": job["ply"],
                    "is_book": job["is_book"],
                    "word_count": job["n_words"],
                    "durations": durations,
                    "text": job["text"],
                }
            )
        print(f"  → {ok_count}/{total_jobs} clips OK")

        lib[category] = {
            "_doc": f"Game {idx + 1} of session {args.session_id} — {opening} [{eco}]",
            "_pgn": movetext,
            "_book_plies": book_plies,
            "_total_plies": len(game.get("moves") or []) or bcg._count_plies(movetext),
            "_session_position": position,
            "_session_id": args.session_id,
            "_game_index": idx + 1,
            "_game_id": gid,
            "_opening": opening,
            "_eco": eco,
            "_result": game.get("result"),
            "_termination": game.get("termination"),
            "lines": lib_lines,
        }

        sessions_summary.append(
            {
                "game_index": idx + 1,
                "game_id": gid,
                "category": category,
                "session_position": position,
                "opening": opening,
                "eco": eco,
                "result": game.get("result"),
                "lines": len(lib_lines),
                "total_seconds": round(
                    sum(d for ln in lib_lines for d in ln["durations"].values()),
                    2,
                ),
            }
        )

        # Persist after each game (resumable).
        lib_path.write_text(json.dumps(lib, indent=2, ensure_ascii=False))

    # Session-level summary file
    summary_path = session_root / "_session.json"
    summary_path.write_text(
        json.dumps(
            {
                "session_id": args.session_id,
                "voice": args.voice,
                "games": sessions_summary,
                "total_seconds": round(
                    sum(g["total_seconds"] for g in sessions_summary), 2
                ),
            },
            indent=2,
        )
    )
    total = sum(g["total_seconds"] for g in sessions_summary)
    print(
        f"\n✓ Session {args.session_id} done: "
        f"{len(sessions_summary)} games, "
        f"{total:.1f}s total ({total / 60:.2f} min) of {args.voice} commentary"
    )
    print(f"  Library: {lib_path.relative_to(REPO_ROOT)}")
    print(f"  Summary: {summary_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
