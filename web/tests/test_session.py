import os

import chess
import pytest

from boardscribe.session import CalibrationError, SessionConfig, run_session
from boardscribe.synth import SceneConfig, SyntheticGame, random_game
from boardscribe.verify import verify_record


def oracle(truth):
    """A user who answers every question correctly."""

    def ask(q):
        b = chess.Board()
        for m in truth[: q.ply - 1]:
            b.push(m)
        san = b.san(truth[q.ply - 1])
        return san if san in q.options else None

    return ask


def test_short_game_is_recovered_and_passes_the_rules_check(tmp_path):
    g = SyntheticGame(random_game(3, 10), seed=3)
    r = run_session(g, g.imu_t, g.imu_gyro, g.event_times, asker=oracle(g.moves), out_dir=str(tmp_path), meta={"seed": 3})
    assert r.complete
    assert r.moves == g.moves
    rec = r.record
    assert rec["schema"] == "boardscribe/record@1" and rec["synthetic"] is True
    assert verify_record(rec).ok
    assert all(p["status"] in {"auto", "asked"} for p in rec["plies"])
    # every ply points at evidence frames that exist
    for p in rec["plies"]:
        assert p["evidence"]
        for e in p["evidence"]:
            assert os.path.exists(tmp_path / e["frame"])


def test_few_frames_are_used_out_of_many():
    g = SyntheticGame(random_game(5, 10), seed=5)
    r = run_session(g, g.imu_t, g.imu_gyro, g.event_times, asker=oracle(g.moves))
    s = r.stats
    assert s.total_frames > 500
    assert s.selected_frames <= 11 * 3  # calibration + 10 moves, 3 frames each
    assert s.decoded_frames < 0.35 * s.total_frames


def test_selected_frames_respect_the_head_speed_gate():
    cfg = SceneConfig(p_shake=1.0)  # every move has a shake episode near it
    g = SyntheticGame(random_game(2, 8), seed=2, config=cfg)
    r = run_session(g, g.imu_t, g.imu_gyro, g.event_times, asker=oracle(g.moves))
    gate = SessionConfig().selection.max_head_speed
    for p in r.record["plies"]:
        if not p["degraded"]:
            assert all(e["head_speed"] <= gate for e in p["evidence"])


def test_no_asker_means_questions_become_review_flags_not_silent_guesses():
    g = SyntheticGame(random_game(1, 12), seed=1)
    r = run_session(g, g.imu_t, g.imu_gyro, g.event_times, asker=None)
    for p in r.record["plies"]:
        assert p["status"] in {"auto", "needs_review"}
        if p["status"] == "needs_review":
            assert p["question"] is not None and p["question"]["answer"] is None
    flagged = [p["ply"] for p in r.record["plies"] if p["status"] == "needs_review"]
    v = verify_record(r.record)
    assert v.ok == (not flagged)


def test_at_most_one_question_per_ply():
    g = SyntheticGame(random_game(4, 12), seed=4)
    seen = []

    def counting(q):
        seen.append(q.ply)
        return oracle(g.moves)(q)

    run_session(g, g.imu_t, g.imu_gyro, g.event_times, asker=counting)
    assert len(seen) == len(set(seen))


def test_blank_video_fails_loudly_at_calibration():
    import numpy as np

    class Blank:
        timestamps = [i / 10 for i in range(60)]

        def get(self, i):
            return np.full((120, 160, 3), 50, np.uint8)

    t = np.arange(0, 6, 0.01)
    with pytest.raises(CalibrationError):
        run_session(Blank(), t, np.zeros((len(t), 3)), [3.0])
