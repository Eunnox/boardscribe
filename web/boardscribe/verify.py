"""The rules engine is the checker.

A record is only as trustworthy as its weakest ply. Replaying it through
python-chess catches an illegal move, a record that disagrees with itself, and
any ply still waiting for a human decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import chess

from .session import to_pgn


@dataclass
class Verification:
    ok: bool
    first_bad_ply: Optional[int] = None
    issues: list[str] = field(default_factory=list)
    pgn: str = ""
    unresolved: list[int] = field(default_factory=list)  # plies still marked needs_review


def replay_uci(ucis: Sequence[str]) -> Verification:
    """Replay UCI moves from the starting position; stop at the first illegal one."""
    board = chess.Board()
    moves: list[chess.Move] = []
    for n, u in enumerate(ucis, start=1):
        try:
            mv = chess.Move.from_uci(u)
        except ValueError:
            return Verification(False, n, [f"ply {n}: {u!r} is not a UCI move"], to_pgn(moves))
        if mv not in board.legal_moves:
            who = "White" if board.turn == chess.WHITE else "Black"
            return Verification(False, n, [f"ply {n}: {u} is illegal for {who} in {board.fen()}"], to_pgn(moves))
        board.push(mv)
        moves.append(mv)
    return Verification(True, None, [], to_pgn(moves))


def verify_record(record: dict, allow_unresolved: bool = False) -> Verification:
    plies = record.get("plies", [])
    v = replay_uci([p["uci"] for p in plies if p.get("uci")])
    if not v.ok:
        return v
    issues: list[str] = []
    first_bad: Optional[int] = None

    def flag(ply: int, msg: str) -> None:
        nonlocal first_bad
        issues.append(f"ply {ply}: {msg}")
        first_bad = ply if first_bad is None else first_bad

    board = chess.Board()
    for p in plies:
        if not p.get("uci"):
            flag(p["ply"], "no move recorded")
            continue
        mv = chess.Move.from_uci(p["uci"])
        san = board.san(mv)
        board.push(mv)
        if p.get("san") not in (None, san):
            flag(p["ply"], f"san {p['san']!r} does not match {san!r}")
        if p.get("fen_after") not in (None, board.fen()):
            flag(p["ply"], "fen_after does not match the replay")
    unresolved = [p["ply"] for p in plies if p.get("status") == "needs_review"]
    if unresolved and not allow_unresolved:
        issues.append("plies waiting for review: " + ", ".join(map(str, unresolved)))
        first_bad = unresolved[0] if first_bad is None else first_bad
    return Verification(not issues, first_bad, issues, v.pgn, unresolved)


def correct_ply(ucis: Sequence[str], ply: int, new_uci: str) -> Verification:
    """What happens to the rest of the game if one ply is corrected?

    Returns the first later ply that is no longer legal, or ok if the whole
    corrected game replays. This is the chain-of-trust check the review UI uses.
    """
    edited = list(ucis)
    if not 1 <= ply <= len(edited):
        raise IndexError(f"ply {ply} out of range")
    edited[ply - 1] = new_uci
    return replay_uci(edited)
