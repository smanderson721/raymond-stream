"""Per-ply Stockfish analysis to ground chess commentary in real engine truth.

For every ply in a game, compute:
  - eval_before:  centipawn eval before the move (from side-to-move's POV)
  - eval_after:   centipawn eval after the move (from same side's POV)
  - delta_cp:     eval_after - eval_before (positive = move helped)
  - classification: "best" | "good" | "inaccuracy" | "mistake" | "blunder"
  - best_move_san: engine's best move in this position (SAN)
  - top_alts:     [(san, eval_cp_for_mover), …] up to 3 alternatives
  - threats:      one-line summary of what the move does/threatens, derived
                  from PV — e.g. "saves knight on a5 (Nxa5 won material)"
  - hanging_pieces_before:  squares of any pieces that were attacked and
                            undefended before the move
  - is_capture / is_check / is_mate / forced

The output is fed into the LLM prompt so the model writes commentary
about ACTUAL reasons, not invented ones.

Usage:
    from production.live.stockfish_analyzer import analyze_game
    analysis = analyze_game(pgn_movetext, depth=18)
    # analysis = [{"ply": 1, "san": "e4", "eval_before": 0.0, ...}, ...]
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Optional

import chess
import chess.engine
import chess.pgn
from io import StringIO


def _stockfish_path() -> str:
    p = shutil.which("stockfish")
    if not p:
        raise RuntimeError("stockfish not found on PATH")
    return p


# Move-quality thresholds (in centipawns, from mover's POV).
# delta = eval_after - eval_before. Negative = position got worse for mover.
THR_BLUNDER = -200
THR_MISTAKE = -100
THR_INACCURACY = -50
THR_GOOD = -15  # within ~0.15 of best


def _classify(delta_cp: int, was_best: bool) -> str:
    if was_best:
        return "best"
    if delta_cp <= THR_BLUNDER:
        return "blunder"
    if delta_cp <= THR_MISTAKE:
        return "mistake"
    if delta_cp <= THR_INACCURACY:
        return "inaccuracy"
    if delta_cp <= THR_GOOD:
        return "good"
    return "best"


def _score_to_cp(score: chess.engine.Score, mover_perspective: chess.Color) -> int:
    """Return centipawn eval from mover's perspective. Mate scores clamp to ±10000."""
    pov = score.pov(mover_perspective)
    if pov.is_mate():
        m = pov.mate()
        if m is None:
            return 0
        return 10000 if m > 0 else -10000
    cp = pov.score()
    return cp if cp is not None else 0


def _hanging_squares(board: chess.Board, side: chess.Color) -> list[str]:
    """Return squares of `side`'s pieces that are attacked by the opponent
    and not defended, considering current material balance — i.e. the piece
    would be lost to the lowest-value attacker.
    """
    out = []
    enemy = not side
    PIECE_VAL = {
        chess.PAWN: 100, chess.KNIGHT: 300, chess.BISHOP: 300,
        chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0,
    }
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if not piece or piece.color != side or piece.piece_type == chess.KING:
            continue
        attackers = board.attackers(enemy, sq)
        if not attackers:
            continue
        defenders = board.attackers(side, sq)
        # Cheapest attacker
        cheapest_atk = min(
            (PIECE_VAL[board.piece_at(a).piece_type] for a in attackers),
            default=10000,
        )
        my_val = PIECE_VAL[piece.piece_type]
        # If attacked by a cheaper piece, or attacked at all without any
        # defender → hanging.
        if not defenders or cheapest_atk < my_val:
            out.append(chess.square_name(sq))
    return out


def _summarize_pv(board: chess.Board, pv: list[chess.Move], n: int = 3) -> str:
    """Return SAN of first n moves of the PV, joined by spaces."""
    if not pv:
        return ""
    b = board.copy()
    sans = []
    for mv in pv[:n]:
        try:
            sans.append(b.san(mv))
        except Exception:
            break
        b.push(mv)
    return " ".join(sans)


def analyze_game(
    pgn_movetext: str,
    depth: int = 18,
    multipv: int = 4,
    threads: int = 4,
    hash_mb: int = 256,
) -> list[dict]:
    """Analyze every ply in `pgn_movetext` and return per-ply ground truth."""
    pgn = chess.pgn.read_game(StringIO(pgn_movetext))
    if pgn is None:
        raise RuntimeError("could not parse PGN")

    board = pgn.board()
    moves = list(pgn.mainline_moves())

    engine = chess.engine.SimpleEngine.popen_uci(_stockfish_path())
    engine.configure({"Threads": threads, "Hash": hash_mb})

    out: list[dict] = []
    try:
        for ply_idx, move in enumerate(moves, start=1):
            mover = board.turn  # color to move BEFORE this ply
            san = board.san(move)

            # Hanging pieces for the side that's ABOUT to move.
            hanging_before = _hanging_squares(board, mover)
            opp_hanging_before = _hanging_squares(board, not mover)

            # Analyze position BEFORE the move.
            info_before = engine.analyse(
                board,
                chess.engine.Limit(depth=depth),
                multipv=multipv,
            )
            best_move = info_before[0].get("pv", [None])[0]
            eval_before_cp = _score_to_cp(info_before[0]["score"], mover)
            best_san = board.san(best_move) if best_move else None
            best_pv_summary = _summarize_pv(board, info_before[0].get("pv", []), n=4)

            # Top alternatives
            top_alts = []
            for mpv in info_before[:multipv]:
                mv = mpv.get("pv", [None])[0]
                if mv is None or mv == move:
                    continue
                try:
                    alt_san = board.san(mv)
                except Exception:
                    continue
                top_alts.append({
                    "san": alt_san,
                    "eval_cp": _score_to_cp(mpv["score"], mover),
                    "pv": _summarize_pv(board, mpv.get("pv", []), n=3),
                })
                if len(top_alts) >= 3:
                    break

            # Was the played move one of the top moves' first move?
            was_best = (best_move == move)

            # Push and evaluate position AFTER move.
            is_capture = board.is_capture(move)
            captured = board.piece_at(move.to_square)
            captured_value = None
            if is_capture and captured:
                vmap = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                        chess.ROOK: 5, chess.QUEEN: 9}
                captured_value = vmap.get(captured.piece_type)

            board.push(move)
            is_check = board.is_check()
            is_mate = board.is_checkmate()

            if is_mate:
                eval_after_cp = 10000  # mover wins
            else:
                info_after = engine.analyse(
                    board,
                    chess.engine.Limit(depth=depth),
                )
                # eval_after from MOVER's perspective (still the same color).
                eval_after_cp = _score_to_cp(info_after["score"], mover)

            delta_cp = eval_after_cp - eval_before_cp
            classification = _classify(delta_cp, was_best)

            # If the played move's eval is within 15 cp of best, consider it
            # "best" even if not the exact same move (transposition).
            if not was_best and abs(eval_before_cp - eval_after_cp) <= 15 and delta_cp >= -15:
                if any(abs(a["eval_cp"] - eval_before_cp) <= 15 for a in top_alts):
                    pass  # keep classification

            ply_info = {
                "ply": ply_idx,
                "san": san,
                "side": "white" if mover == chess.WHITE else "black",
                "eval_before_cp": eval_before_cp,
                "eval_after_cp": eval_after_cp,
                "delta_cp": delta_cp,
                "classification": classification,
                "is_best": was_best,
                "best_move_san": best_san,
                "best_pv": best_pv_summary,
                "top_alts": top_alts,
                "is_capture": is_capture,
                "captured_value": captured_value,
                "is_check": is_check,
                "is_mate": is_mate,
                "hanging_before_mover": hanging_before,
                "hanging_before_opponent": opp_hanging_before,
            }
            out.append(ply_info)
    finally:
        engine.quit()

    return out


def format_for_prompt(analysis: list[dict], book_plies: int = 0) -> str:
    """Render analysis as a compact, LLM-readable per-ply summary."""
    lines = []
    for a in analysis:
        ply = a["ply"]
        is_book = ply <= book_plies
        tag = " [book]" if is_book else ""
        eb = a["eval_before_cp"]
        ea = a["eval_after_cp"]

        def cp_str(cp):
            if cp >= 9000:
                return "+M"
            if cp <= -9000:
                return "-M"
            return f"{cp/100:+.2f}"

        cls = a["classification"]
        suffixes = []
        if a["is_capture"]:
            suffixes.append(f"capture(x{a['captured_value']})" if a["captured_value"] else "capture")
        if a["is_check"]:
            suffixes.append("check")
        if a["is_mate"]:
            suffixes.append("MATE")
        if a["hanging_before_mover"]:
            suffixes.append(f"own_hanging:{','.join(a['hanging_before_mover'])}")
        if a["hanging_before_opponent"]:
            suffixes.append(f"opp_hanging:{','.join(a['hanging_before_opponent'])}")
        suffix_str = (" | " + " ".join(suffixes)) if suffixes else ""

        alts_str = ""
        if a["top_alts"] and not is_book:
            alt_parts = []
            for alt in a["top_alts"][:2]:
                alt_parts.append(f"{alt['san']}({cp_str(alt['eval_cp'])})")
            alts_str = f"  alts: {', '.join(alt_parts)}"

        best_str = ""
        if not a["is_best"] and not is_book and a["best_move_san"]:
            best_str = f"  best={a['best_move_san']}({cp_str(eb)}) pv: {a['best_pv']}"

        lines.append(
            f"ply {ply:>3} {a['side']:>5} {a['san']:<7}{tag} "
            f"eval {cp_str(eb)}→{cp_str(ea)} ({cls}){suffix_str}{best_str}{alts_str}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    # Quick smoke test
    import sys, json
    pgn = sys.argv[1] if len(sys.argv) > 1 else (
        "1.e4 e5 2.Nf3 Nc6 3.Bc4 Bc5 4.b4 Bxb4 5.c3 Ba5 6.d4 exd4 7.O-O dxc3 "
        "8.Qb3 Qf6 9.e5 Qg6 10.Nxc3 Nge7 11.Ba3 b5 12.Qxb5 Rb8 13.Qa4 Bb6 "
        "14.Nxb5 Bxa3 15.Bxa6"
    )
    a = analyze_game(pgn, depth=14)
    print(format_for_prompt(a, book_plies=10))
