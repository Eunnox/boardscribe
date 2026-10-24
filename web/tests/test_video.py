import chess
import numpy as np
import pytest

from boardscribe.session import run_session
from boardscribe.synth import SyntheticGame, random_game
from boardscribe.verify import verify_record
from boardscribe.video import VideoFileSource, read_imu_csv, write_imu_csv


def test_mp4_round_trip_gives_the_same_game(tmp_path):
    g = SyntheticGame(random_game(6, 6), seed=6)
    mp4 = str(tmp_path / "game.mp4")
    g.write_mp4(mp4)
    write_imu_csv(str(tmp_path / "imu.csv"), g.imu_t, g.imu_gyro)

    src = VideoFileSource(mp4)
    assert abs(len(src.timestamps) - len(g.timestamps)) <= 1
    assert src.get(10).shape == (g.config.height, g.config.width, 3)
    t, gyro = read_imu_csv(str(tmp_path / "imu.csv"))
    assert t.shape[0] == g.imu_t.shape[0] and gyro.shape == (len(t), 3)

    def ask(q):
        b = chess.Board()
        for m in g.moves[: q.ply - 1]:
            b.push(m)
        san = b.san(g.moves[q.ply - 1])
        return san if san in q.options else None

    r = run_session(src, t, gyro, g.event_times, asker=ask)
    src.close()
    assert r.moves == g.moves
    assert verify_record(r.record).ok


def test_missing_video_raises():
    with pytest.raises(FileNotFoundError):
        VideoFileSource("/nonexistent/file.mp4")
