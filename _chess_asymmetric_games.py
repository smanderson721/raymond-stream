"""
Asymmetric Stockfish games: one side plays at full strength ("perfect"),
the other plays at a fixed Elo (default 1600). Games are run in parallel
using a process pool. Each finished game is dumped as JSON to OUTPUT_DIR
including PGN, per-move evals (from the perfect side's viewpoint), and
result.

Usage:
    python3 _chess_asymmetric_games.py --num-games 9 \\
        --strong-as-white 6 --strong-as-black 3 \\
        --weak-elo 1600 --strong-time 8.0 --weak-time 0.3 \\
        --threads-per-game 16 --parallel 3
"""

import argparse
import chess
import chess.engine
import chess.pgn
import io
import json
import math
import os
import sys
import time
import uuid
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone


# ── Opening book ────────────────────────────────────────────────────────
# Cached as: list of (parent_name, weight, [variations])
_OPENING_CACHE: list[tuple[str, float, list[dict]]] | None = None


def _load_opening_book(db_path: str,
                       eco_filter: str | None = None
                       ) -> list[tuple[str, float, list[dict]]]:
    """Load openings_db.json grouped by parent opening.

    Returns a list of (parent_name, weight, variations) tuples, where
    `weight = sqrt(len(variations))` so parents with more known variations
    get a slight boost but every parent has a non-trivial chance of being
    picked. Cached per process.

    Each variation is a dict {name, eco, pgn, num_moves}.
    """
    global _OPENING_CACHE
    if _OPENING_CACHE is not None:
        return _OPENING_CACHE
    with open(db_path, "r") as f:
        db = json.load(f)
    grouped: list[tuple[str, float, list[dict]]] = []
    for cat in db.get("categories", {}).values():
        for parent_name, opening in cat.get("openings", {}).items():
            variations: list[dict] = []
            for var in opening.get("variations", []):
                eco = var.get("eco", "")
                if eco_filter and not eco.startswith(eco_filter):
                    continue
                variations.append({
                    "name": f"{parent_name}: {var['name']}",
                    "eco": eco,
                    "pgn": var["pgn"],
                    "num_moves": var.get("num_moves", 0),
                })
            # Some parents have no named sub-variations — only a bare main
            # line. Fall back to that so every parent is pickable.
            if not variations:
                main_pgn = opening.get("main_line_pgn")
                eco_codes = opening.get("eco_codes") or []
                main_eco = eco_codes[0] if eco_codes else ""
                if not main_pgn:
                    continue
                if eco_filter and not main_eco.startswith(eco_filter):
                    continue
                variations.append({
                    "name": parent_name,
                    "eco": main_eco,
                    "pgn": main_pgn,
                    "num_moves": len(main_pgn.split()),
                })
            weight = math.sqrt(len(variations))
            grouped.append((parent_name, weight, variations))
    _OPENING_CACHE = grouped
    return grouped


def _pick_opening(rng: random.Random,
                  grouped: list[tuple[str, float, list[dict]]]
                  ) -> dict | None:
    """Pick a parent opening (weighted by sqrt of variation count), then a
    uniformly random variation within it."""
    if not grouped:
        return None
    weights = [g[1] for g in grouped]
    parent_name, _, variations = rng.choices(grouped, weights=weights, k=1)[0]
    return rng.choice(variations)


# ── Opening trie (for weak-side-only book mode) ─────────────────────────
# Each node = dict[uci_move, {"count": int, "names": set[str], "children": node}]
# Built once per process.
_OPENING_TRIE: dict | None = None


def _load_opening_trie(db_path: str,
                       eco_filter: str | None = None) -> dict:
    """Parse every variation PGN in the openings DB into a UCI move trie.

    The trie is rooted at the standard starting position. Each edge counts
    how many book lines pass through it, which we use as a frequency weight
    when sampling moves. We also remember the opening names that traverse
    each edge so we can attach an opening label to the game.
    """
    global _OPENING_TRIE
    if _OPENING_TRIE is not None:
        return _OPENING_TRIE
    grouped = _load_opening_book(db_path, eco_filter=eco_filter)
    root: dict = {}
    for parent_name, _, variations in grouped:
        for var in variations:
            uci_seq = _parse_pgn_to_uci(var["pgn"])
            if not uci_seq:
                continue
            node = root
            label = var["name"]
            for uci in uci_seq:
                edge = node.setdefault(uci, {
                    "count": 0,
                    "names": set(),
                    "children": {},
                })
                edge["count"] += 1
                edge["names"].add(label)
                node = edge["children"]
    _OPENING_TRIE = root
    return root


def _trie_node_at(root: dict, uci_history: list[str]) -> dict | None:
    """Walk the trie down `uci_history`. Returns the node (children dict)
    at that position, or None if the history exits the trie."""
    node = root
    for uci in uci_history:
        if uci not in node:
            return None
        node = node[uci]["children"]
    return node


def _pick_weak_book_move(rng: random.Random,
                         children: dict,
                         weirdness: float = 0.0) -> tuple[str, set[str]] | None:
    """Pick a random child UCI move weighted by line frequency.

    `weirdness` ∈ [0.0, 1.0] flattens the distribution by raising counts
    to the power `(1.0 - weirdness)`:
      0.0 = pure frequency weighting (mainstream-heavy)
      0.5 = sqrt-flattening (rare moves much more likely)
      1.0 = uniform across all legal book children (Anderssen as common as Sicilian)
    """
    if not children:
        return None
    items = list(children.items())
    exponent = max(0.0, 1.0 - weirdness)
    weights = [edge["count"] ** exponent for _, edge in items]
    uci, edge = rng.choices(items, weights=weights, k=1)[0]
    return uci, edge["names"]


def _parse_pgn_to_uci(pgn_movetext: str) -> list[str]:
    """Convert a PGN move-text fragment (e.g. '1. e4 c5 2. Nf3 d6') into a
    list of UCI move strings, played from the standard starting position.
    """
    pgn = io.StringIO(pgn_movetext)
    game = chess.pgn.read_game(pgn)
    if game is None:
        return []
    return [mv.uci() for mv in game.mainline_moves()]


def play_one_game(args):
    """Play a single asymmetric game. `args` is a dict so it can be pickled."""
    game_id = args["game_id"]
    strong_color = args["strong_color"]  # "white" or "black"
    stockfish_path = args["stockfish_path"]
    weak_engine_path = args.get("weak_engine_path") or stockfish_path
    weak_engine_kind = args.get("weak_engine_kind", "stockfish")  # "stockfish", "stockfish_capped", or "lc0"
    weak_weights = args.get("weak_weights")  # only used for lc0/maia
    weak_depth = args.get("weak_depth")        # only for stockfish_capped
    weak_skill = args.get("weak_skill", 10)    # only for stockfish_capped (0-20)
    strong_time = args["strong_time"]
    strong_depth = args.get("strong_depth")    # if set, overrides strong_time
    strong_skill = args.get("strong_skill", 20) # 0-20, default full strength
    weak_time = args["weak_time"]
    weak_elo = args["weak_elo"]
    threads = args["threads"]
    hash_mb = args["hash_mb"]
    output_dir = args["output_dir"]
    max_plies = args["max_plies"]
    # Opening-book diversification
    opening_book_path = args.get("opening_book_path")
    opening_eco_filter = args.get("opening_eco_filter")
    weak_book_only = args.get("weak_book_only", True)
    opening_weirdness = float(args.get("opening_weirdness", 0.4))

    # Per-game RNG seeded from game_id so each game is reproducible but
    # different from its siblings.
    rng = random.Random(game_id)

    # Pick a forced opening line (legacy mode: full canned line for both
    # sides). Only used when weak_book_only=False.
    opening_meta: dict | None = None
    book_uci: list[str] = []
    # Trie used for weak-book-only mode (live lookup per ply).
    opening_trie: dict | None = None
    book_names_seen: set[str] = set()
    if opening_book_path and os.path.exists(opening_book_path):
        try:
            if weak_book_only:
                opening_trie = _load_opening_trie(
                    opening_book_path, eco_filter=opening_eco_filter,
                )
            else:
                grouped = _load_opening_book(
                    opening_book_path, eco_filter=opening_eco_filter,
                )
                pick = _pick_opening(rng, grouped)
                if pick:
                    book_uci = _parse_pgn_to_uci(pick["pgn"])
                    opening_meta = {
                        "name": pick["name"],
                        "eco": pick["eco"],
                        "pgn": pick["pgn"],
                        "plies": len(book_uci),
                    }
        except Exception as e:
            print(f"[{game_id}] opening-book load failed: {e}", file=sys.stderr)

    # Strong engine: Stockfish at (configurable) full strength
    strong = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    strong.configure({
        "Threads": threads,
        "Hash": hash_mb,
        "Skill Level": strong_skill,
    })

    # Weak engine: either Stockfish-with-Elo-cap or lc0+Maia weights
    if weak_engine_kind == "lc0":
        weak_cmd = [weak_engine_path]
        if weak_weights:
            weak_cmd += ["--weights=" + weak_weights]
        # Maia is a policy net, no search needed for human-like moves.
        weak_cmd += ["--minibatch-size=1", "--threads=2"]
        weak = chess.engine.SimpleEngine.popen_uci(weak_cmd)
        # Maia plays best at low/zero nodes — nodes=1 = pure policy.
    elif weak_engine_kind == "stockfish_capped":
        weak = chess.engine.SimpleEngine.popen_uci(weak_engine_path)
        # Skill Level + fixed shallow depth = much weaker, more human-like
        # blunders than UCI_LimitStrength (which still searches deeply).
        weak.configure({
            "Threads": max(1, threads // 4),
            "Hash": max(64, hash_mb // 4),
            "Skill Level": weak_skill,
        })
    else:
        weak = chess.engine.SimpleEngine.popen_uci(weak_engine_path)
        weak.configure({
            "Threads": max(1, threads // 4),
            "Hash": max(64, hash_mb // 4),
            "UCI_LimitStrength": True,
            "UCI_Elo": weak_elo,
        })

    board = chess.Board()
    moves_data = []
    t0 = time.time()

    try:
        # ── 1. Replay the forced opening (no engine evaluation) ──────
        for uci in book_uci:
            try:
                mv = chess.Move.from_uci(uci)
            except ValueError:
                break
            if mv not in board.legal_moves:
                break
            white_to_move = board.turn == chess.WHITE
            is_strong_turn = (white_to_move and strong_color == "white") or \
                             (not white_to_move and strong_color == "black")
            san = board.san(mv)
            board.push(mv)
            moves_data.append({
                "ply": board.ply(),
                "san": san,
                "uci": mv.uci(),
                "side": "white" if white_to_move else "black",
                "by_strong": is_strong_turn,
                "eval_cp_white": None,
                "fen_after": board.fen(),
                "book": True,
            })

        # ── 2. Engines play naturally from the book exit position ────
        while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
            white_to_move = board.turn == chess.WHITE
            is_strong_turn = (white_to_move and strong_color == "white") or \
                             (not white_to_move and strong_color == "black")

            # ── Weak-book-only opening: serve the weak side a random book
            # move while the trie still has children at this position.
            # The strong side plays freely, and the trie pointer just
            # follows whatever it picks (until it leaves the book).
            book_move: chess.Move | None = None
            if opening_trie is not None and not is_strong_turn:
                history_uci = [m.uci() for m in board.move_stack]
                node = _trie_node_at(opening_trie, history_uci)
                if node:
                    pick = _pick_weak_book_move(rng, node, weirdness=opening_weirdness)
                    if pick:
                        cand_uci, names = pick
                        try:
                            cand = chess.Move.from_uci(cand_uci)
                        except ValueError:
                            cand = None
                        if cand is not None and cand in board.legal_moves:
                            book_move = cand
                            book_names_seen.update(names)

            if book_move is not None:
                white_cp = None
                san = board.san(book_move)
                board.push(book_move)
                moves_data.append({
                    "ply": board.ply(),
                    "san": san,
                    "uci": book_move.uci(),
                    "side": "white" if white_to_move else "black",
                    "by_strong": is_strong_turn,
                    "eval_cp_white": white_cp,
                    "fen_after": board.fen(),
                    "book": True,
                })
                continue

            if is_strong_turn:
                if strong_depth:
                    limit = chess.engine.Limit(depth=strong_depth)
                else:
                    limit = chess.engine.Limit(time=strong_time)
                # Get analysis with eval BEFORE move so we can record it.
                info = strong.analyse(board, limit, multipv=1)
                if isinstance(info, list):
                    info = info[0]
                pv = info.get("pv", [])
                move = pv[0] if pv else None
                if move is None:
                    # fallback
                    result = strong.play(board, limit)
                    move = result.move
                score = info.get("score")
            else:
                if weak_engine_kind == "lc0":
                    # Maia: 1 node = pure policy net = human-like 1600-style move.
                    limit = chess.engine.Limit(nodes=1)
                elif weak_engine_kind == "stockfish_capped" and weak_depth:
                    limit = chess.engine.Limit(depth=weak_depth)
                else:
                    limit = chess.engine.Limit(time=weak_time)
                info = weak.analyse(board, limit, multipv=1)
                if isinstance(info, list):
                    info = info[0]
                pv = info.get("pv", [])
                move = pv[0] if pv else None
                if move is None:
                    result = weak.play(board, limit)
                    move = result.move
                score = info.get("score")

            if move is None or move not in board.legal_moves:
                break

            # Record the eval from the side-to-move's perspective, then
            # convert to "white-relative cp" for downstream display.
            if score is not None:
                rel = score.relative
                if rel.is_mate():
                    rel_cp = 30000 if rel.mate() > 0 else -30000
                else:
                    rel_cp = rel.score()
                white_cp = rel_cp if white_to_move else -rel_cp
            else:
                white_cp = None

            san = board.san(move)
            board.push(move)
            moves_data.append({
                "ply": board.ply(),
                "san": san,
                "uci": move.uci(),
                "side": "white" if white_to_move else "black",
                "by_strong": is_strong_turn,
                "eval_cp_white": white_cp,
                "fen_after": board.fen(),
                "book": False,
            })

        result = board.result(claim_draw=True)
        termination = "checkmate" if board.is_checkmate() else (
            "stalemate" if board.is_stalemate() else (
                "insufficient" if board.is_insufficient_material() else (
                    "75-move" if board.is_seventyfive_moves() else (
                        "5-fold" if board.is_fivefold_repetition() else (
                            "max-plies" if board.ply() >= max_plies else "other"
                        )
                    )
                )
            )
        )
    finally:
        strong.quit()
        weak.quit()

    pgn_game = chess.pgn.Game()
    pgn_game.headers["Event"] = "Asymmetric: Perfect vs 1600"
    pgn_game.headers["Site"] = "Oracle Cloud"
    pgn_game.headers["Date"] = datetime.now(timezone.utc).strftime("%Y.%m.%d")
    pgn_game.headers["Round"] = game_id
    pgn_game.headers["White"] = "Stockfish (perfect)" if strong_color == "white" else f"Stockfish (1600)"
    pgn_game.headers["Black"] = "Stockfish (perfect)" if strong_color == "black" else f"Stockfish (1600)"
    pgn_game.headers["Result"] = result
    pgn_game.headers["StrongColor"] = strong_color
    pgn_game.headers["WeakElo"] = str(weak_elo)
    if opening_meta:
        pgn_game.headers["Opening"] = opening_meta["name"]
        pgn_game.headers["ECO"] = opening_meta["eco"]
    elif book_names_seen:
        # Weak-book-only mode: pick the most-traversed name as the label.
        # (Names overlap across variations; we just want a representative.)
        # Walk the actual played sequence to find the deepest matching label.
        played_uci = [m["uci"] for m in moves_data if m.get("book")]
        opening_meta = {
            "name": sorted(book_names_seen)[0] if book_names_seen else None,
            "eco": None,
            "pgn": None,
            "plies": len(played_uci),
            "weak_book_only": True,
        }
        if opening_meta["name"]:
            pgn_game.headers["Opening"] = opening_meta["name"]
    node = pgn_game
    tmp_board = chess.Board()
    for md in moves_data:
        mv = chess.Move.from_uci(md["uci"])
        node = node.add_variation(mv)
        tmp_board.push(mv)

    out = {
        "game_id": game_id,
        "strong_color": strong_color,
        "weak_elo": weak_elo,
        "weak_engine_kind": weak_engine_kind,
        "weak_skill": weak_skill if weak_engine_kind == "stockfish_capped" else None,
        "weak_depth": weak_depth if weak_engine_kind == "stockfish_capped" else None,
        "opening": opening_meta,
        "strong_time": strong_time,
        "strong_depth": strong_depth,
        "strong_skill": strong_skill,
        "weak_time": weak_time,
        "threads_strong": threads,
        "result": result,
        "termination": termination,
        "ply_count": len(moves_data),
        "duration_sec": round(time.time() - t0, 2),
        "moves": moves_data,
        "pgn": str(pgn_game),
        "final_fen": board.fen(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"game_{game_id}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    return out_path, result, len(moves_data), out["duration_sec"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-games", type=int, default=9)
    p.add_argument("--strong-as-white", type=int, default=6,
                   help="How many games the strong side plays as white")
    p.add_argument("--strong-as-black", type=int, default=3)
    p.add_argument("--weak-elo", type=int, default=1600)
    p.add_argument("--strong-time", type=float, default=8.0,
                   help="Seconds per move for the strong side")
    p.add_argument("--strong-depth", type=int, default=None,
                   help="If set, strong side uses fixed depth instead of strong-time")
    p.add_argument("--strong-skill", type=int, default=20,
                   help="Skill Level (0-20) for the strong side, default 20")
    p.add_argument("--weak-time", type=float, default=0.3,
                   help="Seconds per move for the weak side")
    p.add_argument("--threads-per-game", type=int, default=16)
    p.add_argument("--hash-mb", type=int, default=2048)
    p.add_argument("--parallel", type=int, default=3,
                   help="Number of games to run concurrently")
    p.add_argument("--max-plies", type=int, default=400)
    p.add_argument("--stockfish-path", default="/usr/games/stockfish")
    p.add_argument("--weak-engine-kind", default="stockfish",
                   choices=["stockfish", "stockfish_capped", "lc0"],
                   help="Engine to use for the weak side")
    p.add_argument("--weak-engine-path", default=None,
                   help="Path to weak engine (defaults to stockfish-path)")
    p.add_argument("--weak-weights", default=None,
                   help="Weights file for weak engine (Maia .pb.gz for lc0)")
    p.add_argument("--weak-depth", type=int, default=None,
                   help="Fixed depth for stockfish_capped mode")
    p.add_argument("--weak-skill", type=int, default=10,
                   help="Skill Level (0-20) for stockfish_capped mode")
    p.add_argument("--opening-book",
                   default="research_output/chess/openings_db.json",
                   help="Path to ECO openings_db.json. Set to '' to disable "
                        "scripted openings.")
    p.add_argument("--opening-eco-filter", default=None,
                   help="Restrict to ECO codes starting with this prefix "
                        "(e.g. 'B' for Sicilians/semi-open). Default: all.")
    p.add_argument("--weak-book-only", action="store_true", default=True,
                   help="(Default) Force ONLY the weak side into book "
                        "moves; the strong side plays freely. Maximises "
                        "opening variability.")
    p.add_argument("--no-weak-book-only", dest="weak_book_only",
                   action="store_false",
                   help="Use legacy mode: replay one full canned line for "
                        "both sides before engines take over.")
    p.add_argument("--opening-weirdness", type=float, default=0.4,
                   help="Bias toward rarer book moves on the weak side. "
                        "0.0 = pure frequency (mainstream); 0.5 = sqrt-flatten "
                        "(rare moves much more likely); 1.0 = uniform across "
                        "all legal book children. Default 0.4.")
    p.add_argument("--output-dir",
                   default="research_output/chess/asymmetric_games")
    args = p.parse_args()

    if args.strong_as_white + args.strong_as_black != args.num_games:
        print(f"Warning: strong_as_white ({args.strong_as_white}) + "
              f"strong_as_black ({args.strong_as_black}) != num_games "
              f"({args.num_games}). Adjusting.", file=sys.stderr)

    os.makedirs(args.output_dir, exist_ok=True)
    session_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    jobs = []
    common = {
        "stockfish_path": args.stockfish_path,
        "weak_engine_kind": args.weak_engine_kind,
        "weak_engine_path": args.weak_engine_path,
        "weak_weights": args.weak_weights,
        "weak_depth": args.weak_depth,
        "weak_skill": args.weak_skill,
        "strong_time": args.strong_time,
        "strong_depth": args.strong_depth,
        "strong_skill": args.strong_skill,
        "weak_time": args.weak_time,
        "weak_elo": args.weak_elo,
        "threads": args.threads_per_game,
        "hash_mb": args.hash_mb,
        "output_dir": args.output_dir,
        "max_plies": args.max_plies,
        "opening_book_path": args.opening_book or None,
        "opening_eco_filter": args.opening_eco_filter,
        "weak_book_only": args.weak_book_only,
        "opening_weirdness": args.opening_weirdness,
    }
    for i in range(args.strong_as_white):
        jobs.append({**common, "game_id": f"{session_id}_w{i+1:02d}", "strong_color": "white"})
    for i in range(args.strong_as_black):
        jobs.append({**common, "game_id": f"{session_id}_b{i+1:02d}", "strong_color": "black"})

    print(f"Running {len(jobs)} games, {args.parallel} in parallel "
          f"({args.strong_as_white} strong-white, {args.strong_as_black} strong-black)")
    print(f"Strong side: {args.strong_time}s/move, {args.threads_per_game} threads")
    print(f"Weak side: Elo {args.weak_elo}, {args.weak_time}s/move")
    print(f"Output: {args.output_dir}")
    print()

    t0 = time.time()
    completed = 0
    with ProcessPoolExecutor(max_workers=args.parallel) as pool:
        futures = [pool.submit(play_one_game, j) for j in jobs]
        for fut in as_completed(futures):
            try:
                path, result, plies, dur = fut.result()
                completed += 1
                print(f"[{completed}/{len(jobs)}] {os.path.basename(path)} "
                      f"result={result} plies={plies} dur={dur}s")
            except Exception as e:
                print(f"[FAIL] {e}", file=sys.stderr)

    total = time.time() - t0
    print(f"\nDone. Total wall time: {total:.1f}s ({total/60:.1f} min)")


if __name__ == "__main__":
    main()
