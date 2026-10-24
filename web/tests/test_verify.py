import chess

from boardscribe.verify import correct_ply, replay_uci, verify_record

OPENING = ["e2e4", "e7e5", "d1h5", "b8c6", "f1c4", "g8f6", "h5f7"]  # scholar's mate


def make_record(ucis, statuses=None):
    b = chess.Board()
    plies = []
    for i, u in enumerate(ucis):
        mv = chess.Move.from_uci(u)
        san = b.san(mv)
        b.push(mv)
        plies.append({"ply": i + 1, "uci": u, "san": san, "fen_after": b.fen(), "status": (statuses or {}).get(i + 1, "auto")})
    return {"plies": plies}


def test_legal_game_replays_and_produces_pgn():
    v = replay_uci(OPENING)
    assert v.ok and "Qxf7#" in v.pgn


def test_illegal_move_is_named_with_its_ply():
    v = replay_uci(["e2e4", "e7e5", "d1h6"])  # queen path is blocked
    assert not v.ok and v.first_bad_ply == 3 and "illegal" in v.issues[0]


def test_garbage_move_string():
    assert replay_uci(["e2e4", "zz"]).first_bad_ply == 2


def test_clean_record_verifies():
    assert verify_record(make_record(OPENING)).ok


def test_tampered_san_or_fen_is_caught():
    rec = make_record(OPENING)
    rec["plies"][3]["san"] = "Nc6??"
    v = verify_record(rec)
    assert not v.ok and v.first_bad_ply == 4
    rec = make_record(OPENING)
    rec["plies"][2]["fen_after"] = chess.STARTING_FEN
    assert verify_record(rec).first_bad_ply == 3


def test_record_with_plies_waiting_for_review_is_not_final():
    rec = make_record(OPENING, {5: "needs_review"})
    v = verify_record(rec)
    assert not v.ok and v.unresolved == [5]
    assert verify_record(rec, allow_unresolved=True).ok


def test_correcting_an_early_ply_names_the_first_ply_it_breaks():
    # 1.e4 -> 1.d4 makes the queen's trip d1-h5 illegal at ply 3
    v = correct_ply(OPENING, 1, "d2d4")
    assert not v.ok and v.first_bad_ply == 3


def test_correcting_a_late_ply_to_a_legal_alternative_keeps_the_game_valid():
    v = correct_ply(OPENING, 6, "g8h6")
    assert v.ok and "Nh6" in v.pgn and "Nf6" not in v.pgn
