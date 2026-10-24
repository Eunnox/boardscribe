import chess
import numpy as np

from boardscribe.infer import DecisionConfig, Question, decide, score_moves
from boardscribe.squares import occupancy_classes


def observe(board: chess.Board, p_correct: float = 0.95, blind: tuple[int, ...] = ()) -> np.ndarray:
    """Log-posteriors as a classifier would give for `board`. `blind` squares are flat (hand over them)."""
    occ = occupancy_classes(board)
    post = np.full((64, 3), (1 - p_correct) / 2)
    post[np.arange(64), occ] = p_correct
    for sq in blind:
        post[sq] = 1 / 3
    return np.log(post)


def after(board: chess.Board, san: str) -> chess.Board:
    b = board.copy()
    b.push_san(san)
    return b


def test_clear_evidence_gives_auto_accept_with_the_right_move():
    b = chess.Board()
    inf = score_moves(b, [observe(after(b, "e4"))])
    d = decide(b, inf, 1, asker=None)
    assert d.status == "auto" and d.san == "e4" and d.confidence > 0.99


def test_castling_is_not_confused_with_a_king_move():
    b = chess.Board("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1")
    inf = score_moves(b, [observe(after(b, "O-O"))])
    assert inf.best.san == "O-O"
    inf = score_moves(b, [observe(after(b, "Kf1"))])
    assert inf.best.san == "Kf1"


def test_en_passant_is_found_from_the_vanished_pawn():
    b = chess.Board("rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3")
    inf = score_moves(b, [observe(after(b, "exd6"))])
    assert inf.best.san == "exd6"


def test_two_knights_to_one_square_are_told_apart():
    b = chess.Board("rnbqkbnr/pppppppp/8/8/8/5N2/PPP1PPPP/RNBQKB1R w KQkq - 0 1")
    inf = score_moves(b, [observe(after(b, "Nbd2"))])
    assert inf.best.san == "Nbd2"


def test_hidden_origin_square_leaves_two_moves_close_and_triggers_a_question():
    b = chess.Board("rnbqkbnr/pppppppp/8/8/8/5N2/PPP1PPPP/RNBQKB1R w KQkq - 0 1")
    truth = after(b, "Nbd2")
    # hand covers b1 and f3: the only squares that separate Nbd2 from Nfd2
    inf = score_moves(b, [observe(truth, blind=(chess.B1, chess.F3))])
    sans = {c.san for c in inf.candidates[:2]}
    assert sans == {"Nbd2", "Nfd2"}
    asked = []

    def asker(q: Question):
        asked.append(q)
        return "Nbd2"

    d = decide(b, inf, 7, asker)
    assert d.status == "asked" and d.san == "Nbd2" and len(asked) == 1
    assert set(asked[0].options) >= {"Nbd2", "Nfd2"}


def test_unanswered_question_becomes_needs_review_with_best_guess():
    b = chess.Board("rnbqkbnr/pppppppp/8/8/8/5N2/PPP1PPPP/RNBQKB1R w KQkq - 0 1")
    inf = score_moves(b, [observe(after(b, "Nbd2"), blind=(chess.B1, chess.F3))])
    d = decide(b, inf, 7, asker=lambda q: None)
    assert d.status == "needs_review" and d.question is not None
    d0 = decide(b, inf, 7, asker=lambda q: "Nbd2", config=DecisionConfig(questions_per_ply=0))
    assert d0.status == "needs_review"  # zero budget: never interrupts


def test_promotion_piece_cannot_be_seen_so_it_is_asked():
    b = chess.Board("8/P6k/8/8/8/8/8/4K3 w - - 0 1")
    inf = score_moves(b, [observe(after(b, "a8=Q"))])
    assert len(inf.tied_with_best) == 3
    d = decide(b, inf, 1, asker=lambda q: "a8=N")
    assert d.status == "asked" and d.san == "a8=N"
    d2 = decide(b, inf, 1, asker=None)
    assert d2.status == "needs_review" and d2.san == "a8=Q"  # queen first, flagged


def test_no_change_is_flagged_not_guessed():
    b = chess.Board()
    inf = score_moves(b, [observe(b)])
    assert inf.best.move is None
    d = decide(b, inf, 1, asker=None)
    assert d.status == "needs_review" and d.move is not None  # a legal move is offered for review, never auto


def test_board_that_does_not_match_the_record_is_flagged():
    b = chess.Board()
    other = chess.Board("4k3/8/8/8/8/8/8/4K3 w - - 0 1")  # almost empty board
    inf = score_moves(b, [observe(other)])
    assert inf.fit < DecisionConfig().min_fit
    assert decide(b, inf, 1, asker=None).status == "needs_review"


def test_one_garbage_frame_cannot_outvote_two_good_ones():
    b = chess.Board()
    good = observe(after(b, "e4"))
    rng = np.random.default_rng(0)
    junk = np.log(rng.dirichlet(np.ones(3), size=64))
    inf = score_moves(b, [good, junk, good])
    assert inf.best.san == "e4"


def test_requires_an_observation():
    import pytest

    with pytest.raises(ValueError):
        score_moves(chess.Board(), [])
