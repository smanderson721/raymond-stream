"""Generate segmented chess commentary + multi-voice TTS for the narration library.

Pipeline:
  1. Take a single game (PGN) and ask Gemini 3.1 Flash to write SEGMENTED
     commentary in the voice of a chess streamer actively playing the game.
     Output is a list of segments, each tagged with the move range it covers.
  2. For each segment, call Gemini 3.1 Flash TTS once per voice in VOICES with
     the same Director's Notes preamble (only the voice_name changes).
  3. Convert WAV -> MP3, write to narration_library/<category>/L_<id>__<voice>.mp3
     plus a sidecar .txt with the segment text.
  4. Update narration_library/library.json so the editor can render one row
     per segment with one play button per voice.

Usage:
  python production/live/build_commentary_gemini.py            # Evans miniature default
  python production/live/build_commentary_gemini.py --pgn '1.e4 ...'
  python production/live/build_commentary_gemini.py --voices Charon,Iapetus

Requires GEMINI_API_KEY in env.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import uuid
import wave
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from google import genai
from google.genai import types

NARRATION_DIR = REPO_ROOT / "narration_library"

# Default test game: Evans Gambit miniature ending in Be7#
DEFAULT_PGN = (
    "1.e4 e5 2.Nf3 Nc6 3.Bc4 Bc5 4.c3 Nf6 5.d4 exd4 6.cxd4 Bb4+ 7.Nc3 Nxe4 "
    "8.O-O Nxc3 9.bxc3 Bxc3 10.Ba3 d5 11.Bb5 Bxa1 12.Re1+ Be6 13.Qa4 Rb8 "
    "14.Ne5 Qc8 15.Bxc6+ bxc6 16.Qxc6+ Kd8 17.Nxf7+ Bxf7 18.Be7#"
)

# Six male-coded Gemini TTS voices (kept for reference / A/B testing).
ALL_VOICES = ["Charon", "Rasalgethi", "Sadaltager", "Iapetus", "Algenib", "Enceladus"]
# Selected production voice (speaker #6 from the A/B set).
DEFAULT_VOICES = ["Enceladus"]

# Use Gemini 3.1 Pro for scriptwriting. Pro reasons about chess far better
# than the Flash-lite preview, which had no real positional understanding
# and would invent plausible-sounding but incorrect justifications for
# moves (e.g. claiming b4 "provides structural support" when it actually
# saved a trapped knight on a5). Combined with Stockfish per-ply ground
# truth (see stockfish_analyzer.py), Pro now writes chess-accurate prose.
SCRIPT_MODEL = "gemini-3.1-pro-preview"
TTS_MODEL = "gemini-3.1-flash-tts-preview"  # narration

# Director's Notes used for every TTS call. The transcript is appended.
PREAMBLE = """\
You are synthesizing speech for a chess livestreamer.

# AUDIO PROFILE: Reginald T.
## "The Gentleman Streamer"

## THE SCENE: A Wood-Panelled Study
Reginald is sitting in a leather wing-back chair in a wood-panelled study in
his Mayfair flat. A laptop is open in front of him on a small writing desk.
He is in the middle of an ONLINE chess game against an anonymous opponent
he has never met and cannot see, and he is livestreaming his thought
process to a small audience of viewers. A glass of cognac is at his elbow.
He has played thousands of these games. He is good, and he knows he is
good, and he is mildly bored by how predictable his opponents tend to be —
but he keeps playing because the audience seems to enjoy his commentary.

He is talking out loud as he plays: explaining what he is thinking, why he
is choosing each move, occasionally narrating his emotional reactions to
the position, sometimes addressing the chat directly. He treats his
audience like old friends who happen to also enjoy chess. He never breaks
character. He cannot see his opponent's face — only their moves on the
board.

### DIRECTOR'S NOTES
Style: Pensive, intellectual, faintly condescending — but warm enough that
the audience likes him. Reginald speaks as though every observation is a
small effort he is making on the listener's behalf. He never raises his
voice. He occasionally trails off mid-thought as if a more interesting idea
has occurred to him, then returns. He sounds genuinely amused when his
opponent does something foolish.

Pacing: Unhurried. Considered. Pauses between clauses. Roughly 110-130 WPM.
Never rushed, even when describing tactics. Brief silences are fine.

Accent: BRITISH RECEIVED PRONUNCIATION (RP). The English of an Oxford don
or a BBC Radio 4 documentary narrator — non-rhotic, crisp consonants, long
vowels, broad TRAP/BATH split. NOT American. NOT mid-Atlantic. NOT Estuary.
This is the standalone British English voice as heard in Croydon, London,
or on BBC Radio 4 — every clip in this session must sound the same.
Language: en-GB.

### TRANSCRIPT
"""


# ── Gemini text: produce per-ply commentary ──────────────────────────────
# Speaking-rate calibration. Measured ~134 WPM raw; with 10% speedup ~148 WPM.
MAX_CLIP_SECONDS = 12
WPM_AFTER_SPEEDUP = 148
MAX_WORDS = int(MAX_CLIP_SECONDS * WPM_AFTER_SPEEDUP / 60)  # ~29
# Gemini TTS preview models reliably fail on prompts < ~5 words
# (returns text tokens instead of audio). Enforce a floor.
MIN_WORDS = 5

SCRIPT_SYSTEM = f"""\
You write per-ply commentary for a chess livestream. The streamer is
Reginald, a British gentleman in his 50s with Received Pronunciation. He
is actively playing the game (he is playing {{streamer_color_upper}}) and
livestreaming his thinking and reactions to a small audience of viewers
in real time. He is intellectual, pensive, mildly amused, faintly
condescending, but warm with his audience. He talks to his viewers like
old friends.

This game is one of MANY consecutive games being played back-to-back in a
single livestream session. The session position will be given to you and
controls whether Reginald greets the audience or signs off:

  - "first"  : this is the first game of the session. Reginald GREETS the
               audience briefly somewhere in the early plies (a single
               short hello, e.g. "Good evening, friends. Let us begin.").
               No sign-off at the end.
  - "middle" : this is a mid-session game. NO greeting at all. NO sign-off.
               Reginald continues commentating as if the game already had
               momentum. Do NOT begin with "Good evening", "Hello", "Right,
               then", "Welcome back", or any salutation. Open mid-thought.
  - "last"   : this is the FINAL game of the session. NO greeting. Include
               a brief sign-off in the final comment after the game ends
               (e.g. "And that's it for tonight, friends. Thank you for
               watching."). Keep the sign-off inside the last ply's
               comment word budget.
  - "only"   : there is only one game in the session. Greet at the start
               AND sign off at the end.

You will be given:
  - the full game in PGN
  - the number of opening "book" plies (these are pre-played from an
    opening book and should get sparse, brief commentary at most)
  - the session position (first | middle | last | only)

For each ply (half-move) in the game, decide whether Reginald says
something. Each comment is short (≤ {MAX_WORDS} words, which is ≤
{MAX_CLIP_SECONDS} seconds when spoken). One comment refers to ONE ply
only — but the comment may reference earlier moves or future calculations
to explain Reginald's reasoning.

Rules:

1. ONE PLY PER COMMENT. The comment for ply N must be Reginald's reaction
   to / reasoning about the move played at ply N. He may reference the
   plan unfolding from earlier plies or what he expects next, but the
   comment is anchored to the single ply in question.

2. BOOK PLIES ARE SPARSE AND BRIEF. The first {{book_plies}} plies are
   pre-played opening moves. Skip most of them entirely (no comment).
   Cover at most 2-3 of the book plies, and each comment should be
   short ({MIN_WORDS}-10 words, 2-4 seconds): a brief naming of the
   opening, a muttered "standard fare, nothing surprising". Do NOT
   rationalize book moves at length. The greeting (if any) lives inside
   one of the early book-ply comments.

3. POST-BOOK PLIES (ply > {{book_plies}}). HARD QUOTA: you MUST produce
   commentary on AT LEAST 40% of post-book plies, and ideally 45-50%.
   Count carefully. If there are 50 post-book plies, you must produce
   AT LEAST 20 comments on those plies (in addition to any 2-3 book-ply
   comments). Do NOT under-produce — this is the most common failure
   mode. If you find yourself with too few comments on a draft, go back
   and add more on plies you initially skipped. Distribute the comments
   across the whole game (opening transition, middlegame, endgame,
   finish) — do not cluster them all in one phase. Even quiet positional
   plies, prophylactic moves, and routine developing moves deserve a
   thought from Reginald. Skip plies only if they are obvious recaptures
   or fully forced. Always comment on:
     - opponent blunders or surprising moves
     - Reginald's tactical decisions (sacrifices, key tempo moves)
     - the final checkmate
   Before returning, COUNT your post-book comments and verify the count
   is at least 40% of the post-book ply count. If not, add more.

4. STRICT WORD LIMITS: each comment must be between {MIN_WORDS} and
   {MAX_WORDS} words inclusive. NEVER write a comment shorter than
   {MIN_WORDS} words — short utterances cause the TTS engine to fail.
   NEVER write a comment longer than {MAX_WORDS} words.

4b. DELIBERATE LENGTH VARIATION. Comments must vary substantially in
   length across the game. Aim for a roughly even mix:
     - ~30% short reactions (5–9 words): "a quick recapture, knight takes
       e4.", "d4 it is, then.", "that bishop to b4 looks loose."
     - ~40% medium thoughts (10–18 words): one clear idea + a fragment
       of reasoning.
     - ~30% longer reflections (19–{MAX_WORDS} words): the move + WHY,
       what's coming next, what the opponent might be planning.
   NEVER write three consecutive comments of similar length. The pacing
   of the stream depends on this variation — if every comment is the
   same length the moves come at a metronomic clip and it sounds robotic.
   Pick the length to fit the moment: forced recaptures and obvious
   replies get short comments; pivotal tactics, blunders, and quiet
   strategic decisions get longer ones.

5. STRICT FIRST PERSON for Reginald's own moves. When Reginald plays a
   move, he describes it as something HE is doing ("I'll snap that off",
   "let me get the rook over", "I think I'll sacrifice the bishop here").
   NEVER use detached or third-person language about his own moves.
   NEVER admire his own moves with phrases like "a beautiful sacrifice",
   "an elegant finish", "a brilliant tactic", "the trap is sprung",
   "a tactical flurry". Reginald may describe what HE is trying to
   achieve ("I want to clear the file", "this should be the end of it"),
   but he does not pat himself on the back as if he were watching from
   outside. He may freely critique or admire the OPPONENT's moves.

6. ONLINE OPPONENT. The game is played online. Reginald cannot see his
   opponent's face, body, or behaviour. NEVER write "he looks flustered",
   "she is sweating", "he shuffles his pieces", "he glances up". He may
   only INFER the opponent's state from their MOVES ("a quick reply, he
   must have prepared this", "a long think — the position has confused
   him"). The opponent is anonymous: "our opponent", "our friend on the
   other side", "the visitor", etc.

7. ALWAYS NAME THE MOVE — SPELLED OUT FOR SPEECH, INTEGRATED INTO PROSE.
   Every comment must explicitly mention the move it is reacting to,
   but written so the TTS engine speaks it naturally. NEVER write raw
   SAN like "Ne4", "Ke2", "Bxc5", "Qh5+", "Rxf7+", "O-O", "O-O-O",
   "Nf3#". Always expand the piece letter to its spoken word and the
   file/rank to letters and numbers spoken out:
     - K → "king" (e.g. Ke2 → "king to e2")
     - Q → "queen" (Qh5 → "queen h5", Qxd2# → "queen takes d2, mate")
     - R → "rook" (Rxf7+ → "rook takes f7, check")
     - B → "bishop" (Bxc5 → "bishop takes c5")
     - N → "knight" (Ne4 → "knight to e4", NEVER "night" or "N e four")
     - pawn moves: just say the destination square ("d4", "e5")
     - pawn captures: "e takes d5", "f takes e6"
     - O-O → "castles kingside"; O-O-O → "castles queenside"
     - + → "check"; # → "checkmate" or "mate"
     - x in piece moves → "takes"

   CRITICAL — INTEGRATE THE MOVE INTO THE SENTENCE. NEVER open a
   comment by stating the bare move name as a label followed by a
   sentence-stop and then continuing. The move must be woven into the
   prose. The TTS engine treats the period after a bare move name as
   a hard pause and it sounds robotic and lazy.

   FORBIDDEN openings (do NOT do this):
     ✗ "Knight to e5. A strong central post."
     ✗ "d4. Striking in the centre."
     ✗ "Bishop takes c5. Material is even now."
     ✗ "Queen to h5, check. Forcing the king out."

   GOOD openings (do this instead). Vary the form across comments
   — do not lock into a single template:

   For Reginald's OWN moves, use natural first-person framings such as:
     ✓ "Let's go knight to e5 — a strong central post."
     ✓ "Here, I think I'll play d4 and stake the centre."
     ✓ "Knight to e5 looks like the clear choice; nothing else holds."
     ✓ "I'll snap that off with bishop takes c5; material is even now."
     ✓ "Queen to h5, then — with check, dragging his king out."
     ✓ "A simple recapture for me, e takes d5."
     ✓ "This calls for d4. The position demands it."
     ✓ "I rather like rook to e1 here, building pressure on the file."
     ✓ "Let me drop the knight back to f6, out of trouble."
     ✓ "My turn, and castles kingside is overdue."

   For the OPPONENT's moves, use varied reactive framings such as:
     ✓ "He retreats with bishop to c8, surrendering the bishop pair."
     ✓ "Knight to e4? Goodness, that walks straight into the pin."
     ✓ "Our friend tries h4, an aggressive lunge on the flank."
     ✓ "I would not have gone knight to e5 there — it allows me to
        fork his queen and rook."
     ✓ "He chooses d takes e4. Predictable, and entirely good for me."
     ✓ "Queen to b6 is a curious choice from him."
     ✓ "The opponent obliges with rook to d5, missing the threat
        on the back rank."
     ✓ "That's a blunder — bishop takes h6 hangs the piece."
     ✓ "Castles queenside from him; long castles always feels brave
        in this structure."

   The move name should appear ONCE, embedded in the sentence's normal
   syntax — never as a standalone leading label. Vary your sentence
   forms across consecutive comments so the cadence does not become
   repetitive.

8. Plain prose only. NO stage directions, audio tags, or sound effects.

9. GROUND-TRUTH ENGINE ANALYSIS. Below the move list you will be given
   per-ply Stockfish analysis showing the eval before and after each
   move, the engine classification (best/good/inaccuracy/mistake/blunder),
   the engine's preferred alternatives, and which pieces are hanging.
   Use THIS as the basis for Reginald's reasoning. Do NOT invent reasons.
   For example:
     - if a move is classified "blunder" with `opp_hanging:a5` it means
       Reginald's piece on a5 was about to be captured; the move's real
       purpose is to save it (or fail to). Do not say it "strengthens the
       structure" — say what's actually happening.
     - if a move is `best` and the eval is heavily in Reginald's favour,
       he should sound calmly confident, not surprised.
     - if a move is `mistake` or `blunder`, Reginald (if it's his move)
       should sound mildly annoyed, or rationalise quickly; if it's the
       opponent's mistake, he should be quietly delighted.
     - reference the engine's named alternative or the threatened piece
       directly when it makes the comment more truthful.
   Reginald never says "the engine". He just sees the position correctly.

10. NEVER invent tactical justifications that conflict with the engine
    analysis. If a move is just a quiet developing move, say so. Do not
    claim non-existent threats, pins, forks, or piece-trapping.

Output STRICT JSON, sorted by ply ascending. Only include plies that get
a comment; omit plies that are silent.

{{
  "comments": [
    {{"ply": 1, "text": "..."}},
    {{"ply": 7, "text": "..."}},
    ...
  ]
}}
"""


def _count_plies(pgn: str) -> int:
    # Count SAN moves: strip move numbers, split on whitespace.
    text = re.sub(r"\d+\.\.?\.?", " ", pgn)
    text = re.sub(r"\{[^}]*\}", " ", text)
    text = re.sub(r"[\(\)]", " ", text)
    tokens = [t for t in text.split() if t and not re.match(r"^(1-0|0-1|1/2-1/2|\*)$", t)]
    return len(tokens)


def _ply_san_map(pgn: str) -> list[tuple[int, str]]:
    """Return [(ply_number, san_token), ...] in order."""
    text = re.sub(r"\{[^}]*\}", " ", pgn)
    text = re.sub(r"\d+\.\.?\.?", " ", text)
    text = re.sub(r"[\(\)]", " ", text)
    tokens = [
        t for t in text.split()
        if t and not re.match(r"^(1-0|0-1|1/2-1/2|\*)$", t)
    ]
    return [(i + 1, san) for i, san in enumerate(tokens)]


def write_comments(
    client: genai.Client,
    pgn: str,
    book_plies: int,
    session_position: str = "only",
    streamer_color: str = "white",
    engine_analysis: list[dict] | None = None,
) -> list[dict]:
    """Generate per-ply commentary.

    `engine_analysis`, if provided, is the output of
    `stockfish_analyzer.analyze_game(pgn)` and is injected into the prompt
    as ground truth. Strongly recommended — without it the LLM has no real
    chess understanding and will hallucinate justifications.
    """
    streamer_color = (streamer_color or "white").lower()
    if streamer_color not in ("white", "black"):
        streamer_color = "white"
    opp_color = "black" if streamer_color == "white" else "white"
    streamer_is_white = streamer_color == "white"
    ply_map = _ply_san_map(pgn)
    ply_lines = []
    for ply, san in ply_map:
        # ply 1 is always White; map to whichever side is Reginald.
        ply_is_white = (ply % 2 == 1)
        is_streamer = (ply_is_white == streamer_is_white)
        side = (
            f"{'White' if ply_is_white else 'Black'} (Reginald)"
            if is_streamer
            else f"{'White' if ply_is_white else 'Black'} (opponent)"
        )
        ply_lines.append(f"  ply {ply} — {side}: {san}")
    ply_block = "\n".join(ply_lines)

    # Engine ground-truth block. Eval signs are from the side-to-move's
    # POV, so we annotate with which side is moving for clarity.
    engine_block = ""
    if engine_analysis:
        from production.live.stockfish_analyzer import format_for_prompt
        engine_block = (
            "\n\nStockfish ground-truth analysis (depth 18, evals in pawns from the\n"
            "mover's POV; positive = the mover is winning; classification = best |\n"
            "good | inaccuracy | mistake | blunder; own/opp_hanging shows pieces\n"
            "under attack BEFORE the move; alts = engine's preferred alternatives;\n"
            "best=X pv=... shows what the engine wanted instead). Anchor every\n"
            "comment in this analysis. Do not invent threats not shown here.\n\n"
            + format_for_prompt(engine_analysis, book_plies=book_plies)
        )

    prompt = (
        f"Game (Reginald is {streamer_color.capitalize()}, opponent is "
        f"{opp_color.capitalize()}; opening book = first {book_plies} "
        f"plies):\n\n"
        f"{pgn}\n\n"
        f"Per-ply SAN moves (use these exact moves, named naturally and\n"
        f"spelled out for speech — e.g. 'knight to e4', not 'Ne4' — in each\n"
        f"comment):\n"
        f"{ply_block}"
        f"{engine_block}\n\n"
        f"Session position: {session_position}\n\n"
        f"Return JSON only."
    )
    sys_prompt = SCRIPT_SYSTEM.replace("{book_plies}", str(book_plies))
    sys_prompt = sys_prompt.replace(
        "{streamer_color_upper}", streamer_color.capitalize()
    )
    resp = client.models.generate_content(
        model=SCRIPT_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=sys_prompt,
            response_mime_type="application/json",
            temperature=0.9,
        ),
    )
    raw = resp.text.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    data = json.loads(raw)
    items = data.get("comments") or []
    if not items:
        raise RuntimeError("Gemini returned no comments")
    # Enforce word cap and sort by ply.
    items.sort(key=lambda c: c.get("ply", 0))
    cleaned = []
    for c in items:
        text = (c.get("text") or "").strip()
        if not text:
            continue
        words = text.split()
        if len(words) < MIN_WORDS:
            print(
                f"  ⚠ dropping ply {c.get('ply')} comment ({len(words)}w < {MIN_WORDS}w floor): {text!r}"
            )
            continue
        if len(words) > MAX_WORDS:
            text = " ".join(words[:MAX_WORDS])
        cleaned.append({"ply": int(c["ply"]), "text": text})
    return cleaned


# ── Gemini TTS: generate one WAV per (segment, voice) ────────────────────
class NoAudioReturned(Exception):
    """Gemini TTS returned text tokens instead of audio (documented preview limitation)."""


def synth_one(client: genai.Client, voice: str, text: str, out_wav: Path) -> None:
    prompt = PREAMBLE + text.strip() + "\n"
    last_err: Exception | None = None
    for attempt in range(4):
        try:
            resp = client.models.generate_content(
                model=TTS_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        language_code="en-GB",
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name=voice,
                            )
                        )
                    ),
                ),
            )
            try:
                data = resp.candidates[0].content.parts[0].inline_data.data
            except (AttributeError, IndexError, TypeError):
                data = None
            if not data:
                raise NoAudioReturned(
                    "model returned no audio (likely classifier rejection on short prompt)"
                )
            out_wav.parent.mkdir(parents=True, exist_ok=True)
            with wave.open(str(out_wav), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(24000)
                wf.writeframes(data)
            return
        except Exception as exc:
            last_err = exc
            wait = 2 ** attempt
            print(f"  ⚠ {voice} attempt {attempt+1} failed ({exc}); retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"TTS failed for voice {voice}: {last_err}")


async def synth_one_async(
    client: genai.Client, voice: str, text: str, out_wav: Path
) -> None:
    """Async variant of synth_one for concurrent TTS calls.

    Uses google-genai's `client.aio.*` async surface. Each call still
    has 4-attempt exponential backoff; failures are still raised.
    """
    import asyncio  # local import keeps module load light when sync-only
    prompt = PREAMBLE + text.strip() + "\n"
    last_err: Exception | None = None
    for attempt in range(4):
        try:
            resp = await client.aio.models.generate_content(
                model=TTS_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        language_code="en-GB",
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name=voice,
                            )
                        )
                    ),
                ),
            )
            try:
                data = resp.candidates[0].content.parts[0].inline_data.data
            except (AttributeError, IndexError, TypeError):
                data = None
            if not data:
                raise NoAudioReturned(
                    "model returned no audio (likely classifier rejection on short prompt)"
                )
            out_wav.parent.mkdir(parents=True, exist_ok=True)
            with wave.open(str(out_wav), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(24000)
                wf.writeframes(data)
            return
        except Exception as exc:
            last_err = exc
            wait = 2 ** attempt
            await asyncio.sleep(wait)
    raise RuntimeError(f"TTS failed for voice {voice}: {last_err}")


# Speedup factor applied to every TTS clip.
SPEEDUP = 1.10


def wav_to_mp3(wav_path: Path, mp3_path: Path) -> None:
    """Convert mono 24kHz WAV to stereo MP3 with SPEEDUP applied.

    - atempo=SPEEDUP speeds the audio without pitch shifting
    - aformat=channel_layouts=stereo,pan=stereo|c0=c0|c1=c0 duplicates the
      single mono channel onto both L and R, so playback is the same in
      both ears (compatible with macOS "Mono Audio" accessibility setting).
    """
    af = (
        f"atempo={SPEEDUP},"
        "aformat=channel_layouts=mono,"
        "pan=stereo|c0=c0|c1=c0"
    )
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(wav_path),
            "-af", af,
            "-ac", "2",
            "-codec:a", "libmp3lame", "-q:a", "3",
            str(mp3_path),
        ],
        check=True,
    )


def mp3_duration(mp3_path: Path) -> float:
    out = subprocess.check_output(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(mp3_path),
        ]
    )
    return float(out.strip())


# ── Main ────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pgn", default=DEFAULT_PGN, help="PGN movetext")
    p.add_argument(
        "--book-plies",
        type=int,
        default=8,
        help="Number of opening plies pre-played from the book (sparse commentary)",
    )
    p.add_argument(
        "--category",
        default="commentary_evans_miniature",
        help="Library category folder name",
    )
    p.add_argument(
        "--voices",
        default=",".join(DEFAULT_VOICES),
        help="Comma-separated Gemini TTS voice names",
    )
    p.add_argument(
        "--label",
        default="Evans Gambit miniature (Reginald, White, plays Be7#)",
        help="Human label shown in the editor",
    )
    p.add_argument(
        "--session-position",
        default="only",
        choices=["first", "middle", "last", "only"],
        help="Where this game sits in a back-to-back session "
             "(controls greeting/sign-off behavior)",
    )
    p.add_argument(
        "--streamer-color",
        default="white",
        choices=["white", "black"],
        help="Which color Reginald is playing in this game",
    )
    args = p.parse_args()

    if not os.environ.get("GEMINI_API_KEY"):
        sys.exit("GEMINI_API_KEY not set in environment")

    voices = [v.strip() for v in args.voices.split(",") if v.strip()]
    if not voices:
        sys.exit("No voices specified")

    total_plies = _count_plies(args.pgn)
    book_plies = max(0, min(args.book_plies, total_plies))

    client = genai.Client()

    print(
        f"▶ Writing per-ply commentary with {SCRIPT_MODEL} "
        f"({total_plies} plies, first {book_plies} are book, "
        f"session_position={args.session_position})…"
    )
    comments = write_comments(
        client,
        args.pgn,
        book_plies,
        args.session_position,
        streamer_color=args.streamer_color,
    )
    print(f"  got {len(comments)} comments")

    cat_dir = NARRATION_DIR / args.category
    cat_dir.mkdir(parents=True, exist_ok=True)

    lib_lines = []
    for idx, c in enumerate(comments, start=1):
        line_id = f"L_{uuid.uuid4().hex[:8]}"
        text = c["text"].strip()
        ply = int(c["ply"])
        is_book = ply <= book_plies

        (cat_dir / f"{line_id}.txt").write_text(text)

        n_words = len(text.split())
        print(
            f"\n[{idx}/{len(comments)}] ply {ply}{' (book)' if is_book else ''} "
            f"({n_words}w): {text[:80]}…"
        )

        durations: dict[str, float] = {}
        for voice in voices:
            wav = cat_dir / f"{line_id}__{voice}.wav"
            mp3 = cat_dir / f"{line_id}__{voice}.mp3"
            if mp3.exists():
                print(f"  ⏭  {voice} already exists, skipping")
                durations[voice] = round(mp3_duration(mp3), 3)
                continue
            print(f"  ▶ synth {voice}…", end="", flush=True)
            try:
                synth_one(client, voice, text, wav)
                wav_to_mp3(wav, mp3)
                wav.unlink()
                durations[voice] = round(mp3_duration(mp3), 3)
                print(f" done ({durations[voice]:.2f}s)")
            except Exception as exc:
                print(f" FAILED — skipping ({exc})")
                # Clean up any partial wav.
                if wav.exists():
                    try:
                        wav.unlink()
                    except Exception:
                        pass

        lib_lines.append(
            {
                "id": line_id,
                "ply": ply,
                "is_book": is_book,
                "word_count": n_words,
                "durations": durations,
                "text": text,
            }
        )

    lib_path = NARRATION_DIR / "library.json"
    if lib_path.exists():
        lib = json.loads(lib_path.read_text())
    else:
        lib = {}

    lib.setdefault(
        "_meta",
        {
            "voice": "Gemini 3.1 Flash TTS (multi-voice A/B)",
            "personality": (
                "Reginald T. — British gentleman chess streamer in his 50s. "
                "RP accent. Pensive, intellectual, mildly amused, faintly "
                "condescending, warm with audience. Plays White, narrates "
                "his own thinking and reactions to the audience as he plays."
            ),
            "trigger_notes": (
                "Per-ply commentary. Each line is anchored to one ply. "
                "is_book=true means the ply is from the opening book and "
                "the live player can ignore the duration constraint. For "
                "post-book plies, the live player should delay the next "
                "opponent ply until 1-4s after the chosen variant's "
                "`durations[voice]` has elapsed."
            ),
            "audio": {
                "speedup": SPEEDUP,
                "channels": "stereo (mono signal duplicated to L+R)",
                "max_clip_seconds": MAX_CLIP_SECONDS,
                "target_wpm_after_speedup": WPM_AFTER_SPEEDUP,
            },
        },
    )

    lib[args.category] = {
        "_doc": args.label,
        "_pgn": args.pgn,
        "_book_plies": book_plies,
        "_total_plies": total_plies,
        "_session_position": args.session_position,
        "lines": lib_lines,
    }
    lib_path.write_text(json.dumps(lib, indent=2, ensure_ascii=False))

    # Summary
    total_per_voice: dict[str, float] = {v: 0.0 for v in voices}
    for ln in lib_lines:
        for v, d in ln["durations"].items():
            total_per_voice[v] = total_per_voice.get(v, 0.0) + d
    print(f"\n✓ Wrote {lib_path.relative_to(REPO_ROOT)} ({len(lib_lines)} comments)")
    print(f"  Audio in {cat_dir.relative_to(REPO_ROOT)}/ ({len(voices)} voices each)")
    print("\nTotal speaking time per voice:")
    for v in voices:
        t = total_per_voice.get(v, 0.0)
        print(f"  {v:<14} {t:6.1f}s ({t/60:.2f} min)")


if __name__ == "__main__":
    main()
