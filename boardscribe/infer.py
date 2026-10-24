"""Which legal move produced what the camera sees?

The previous position is known, so the next move is one of its legal moves. For
each legal move we know exactly which squares should be empty / white / black
afterwards, and we score that against the per-square observations. Perception
only has to separate a few dozen hypotheses, not recognise a board from scratch.

When the evidence does not settle it, the user is asked, at most once per move.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import chess
import numpy as np

from .squares import occupancy_classes

LOG_FLOOR = float(np.log(0.02))  # one square can cost a hypothesis at most this much per frame


@dataclass(frozen=True)
class MoveCandidate:
    move: Optional[chess.Move]  # None = "nothing changed"
    san: str
    score: float
    posterior: float


@dataclass
class Inference:
    candidates: list[MoveCandidate]  # best first
    fit: float  # mean log-probability per square under the best hypothesis (closer to 0 is better)
    tied_with_best: list[MoveCandidate] = field(default_factory=list)  # same occupancy as the best, e.g. promotions

    @property
    def best(self) -> MoveCandidate:
        return self.candidates[0]


def score_moves(board: chess.Board, frame_log_posts: Sequence[np.ndarray]) -> Inference:
    """Rank the legal moves (plus "no change") by how well they explain the observed squares."""
    if not frame_log_posts:
        raise ValueError("need at least one observation")
    clipped = np.maximum(np.stack(frame_log_posts), LOG_FLOOR)  # (frames, 64, 3)
    avg = clipped.mean(axis=0)  # average over frames: correlated frames must not inflate confidence
    idx = np.arange(64)

    hyps: list[tuple[Optional[chess.Move], str, np.ndarray]] = [(None, "(no change)", occupancy_classes(board))]
    for mv in board.legal_moves:
        san = board.san(mv)
        board.push(mv)
        hyps.append((mv, san, occupancy_classes(board)))
        board.pop()

    scores = np.array([avg[idx, occ].sum() for _, _, occ in hyps])
    post = np.exp(scores - scores.max())
    post /= post.sum()

    order = np.argsort(-scores, kind="stable")
    cands = [MoveCandidate(hyps[i][0], hyps[i][1], float(scores[i]), float(post[i])) for i in order]
    best_occ = hyps[order[0]][2]
    tied = [
        MoveCandidate(hyps[i][0], hyps[i][1], float(scores[i]), float(post[i]))
        for i in order[1:]
        if np.array_equal(hyps[i][2], best_occ)
    ]
    fit = float(avg[idx, best_occ].mean())
    return Inference(cands, fit, tied)


@dataclass(frozen=True)
class Question:
    ply: int
    text: str
    options: tuple[str, ...]  # SAN strings, most likely first


Asker = Callable[[Question], Optional[str]]  # returns the chosen SAN, or None if not answered


@dataclass(frozen=True)
class DecisionConfig:
    accept_posterior: float = 0.9
    min_fit: float = -1.2  # worse than this: the board does not look like the record
    ask_floor: float = 0.03  # only offer moves with at least this posterior
    max_options: int = 4
    questions_per_ply: int = 1  # the interruption budget


@dataclass
class Decision:
    move: Optional[chess.Move]
    san: str
    status: str  # "auto" | "asked" | "needs_review"
    confidence: float
    note: str = ""
    question: Optional[Question] = None
    answer: Optional[str] = None


def decide(
    board: chess.Board,
    inference: Inference,
    ply: int,
    asker: Optional[Asker],
    config: DecisionConfig = DecisionConfig(),
) -> Decision:
    best = inference.best
    group = [best] + inference.tied_with_best
    group_p = sum(c.posterior for c in group)
    needs_choice = len(group) > 1  # e.g. promotion: occupancy cannot tell Q from N
    no_change = best.move is None
    off_fit = inference.fit < config.min_fit

    confident = group_p >= config.accept_posterior and not needs_choice and not no_change and not off_fit
    if confident:
        return Decision(best.move, best.san, "auto", best.posterior)

    reason = (
        "promotion piece cannot be seen from occupancy"
        if needs_choice
        else "no change was detected"
        if no_change
        else "the board does not match the record so far"
        if off_fit
        else "two moves look alike"
    )

    options = [c for c in inference.candidates if c.posterior >= config.ask_floor or c in group][: config.max_options]
    if needs_choice:
        options = sorted(group, key=lambda c: (not (c.move and c.move.promotion == chess.QUEEN), c.san))[: config.max_options]
    options = [c for c in options if c.move is not None] or [c for c in inference.candidates if c.move is not None][:2]
    texts = tuple(c.san for c in options)
    q = Question(ply, f"Which move was that? ({reason})", texts)

    if asker is not None and config.questions_per_ply > 0:
        ans = asker(q)
        if ans is not None:
            chosen = next((c for c in options if c.san == ans), None)
            if chosen is not None:
                return Decision(chosen.move, chosen.san, "asked", chosen.posterior, note=reason, question=q, answer=ans)

    # No answer available: keep the best legal move and flag it for the review screen.
    fallback = options[0]
    return Decision(fallback.move, fallback.san, "needs_review", fallback.posterior, note=reason, question=q)
