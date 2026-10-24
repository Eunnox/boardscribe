"""Video + IMU + "move done" times  ->  evidence-linked record of the game.

    frames.select_frames   sharp, steady frames around each move
    grid                   board corners -> top-down squares
    squares                calibrate on the starting position, score squares
    infer                  which legal move explains the squares; ask if unsure
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass, field, replace
from typing import Optional, Sequence

import chess
import chess.pgn
import cv2
import numpy as np

from .frames import ActionEvent, FrameSource, SelectedFrame, SelectionConfig, SelectionStats, select_frames
from .grid import BoardHomography, BoardLocator, ClassicalBoardLocator, square_patches
from .infer import Asker, Decision, DecisionConfig, decide, score_moves
from .squares import SquareModel, square_features

SCHEMA = "boardscribe/record@1"


class CalibrationError(RuntimeError):
    pass


@dataclass(frozen=True)
class SessionConfig:
    selection: SelectionConfig = SelectionConfig(window_before_s=0.5, window_after_s=1.5)
    decision: DecisionConfig = DecisionConfig()
    px_per_square: int = 32
    calibration_t: float = 0.5
    calibration_half_window_s: float = 0.5
    naive_frames: bool = False  # ablation: the 3 frames nearest the event, no IMU gate, no sharpness
    max_corner_shift_px: float = 20.0  # a board quad that jumps further than this is a locator error (e.g. a hand)
    jpeg_quality: int = 85


@dataclass
class Observation:
    log_post: np.ndarray
    corners: np.ndarray


@dataclass
class SessionResult:
    record: dict
    moves: list[chess.Move]
    stats: SelectionStats
    complete: bool


def _observe(
    frame: np.ndarray,
    locator: BoardLocator,
    s: int,
    ref_corners: Optional[np.ndarray] = None,
    max_shift: float = 20.0,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    corners = locator.locate(frame)
    if corners is None:
        return None, None
    if ref_corners is not None and float(np.abs(corners - ref_corners).max()) > max_shift:
        return None, None
    hom = BoardHomography(corners)
    feats = square_features(square_patches(hom.warp(frame, s), s))
    return feats, corners


def _annotate(frame: np.ndarray, corners: np.ndarray, move: chess.Move) -> np.ndarray:
    out = frame.copy() if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    hom = BoardHomography(corners)
    for sq, colour in ((move.from_square, (80, 200, 255)), (move.to_square, (90, 220, 90))):
        file, row = chess.square_file(sq), 7 - chess.square_rank(sq)
        quad = np.array([hom.units_to_px(p) for p in ((file, row), (file + 1, row), (file + 1, row + 1), (file, row + 1))], np.int32)
        cv2.polylines(out, [quad], True, colour, 2, cv2.LINE_AA)
    return out


def _nearest_frames(source: FrameSource, ev: ActionEvent, k: int = 3) -> list[SelectedFrame]:
    times = np.asarray(source.timestamps)
    idx = np.argsort(np.abs(times - ev.t), kind="stable")[:k]
    return [SelectedFrame(ev.id, int(i), float(times[i]), 0.0, 0.0, source.get(int(i))) for i in sorted(idx)]


def run_session(
    source: FrameSource,
    imu_t: Sequence[float],
    imu_gyro: np.ndarray,
    move_times: Sequence[float],
    asker: Optional[Asker] = None,
    locator: Optional[BoardLocator] = None,
    config: SessionConfig = SessionConfig(),
    out_dir: Optional[str] = None,
    meta: Optional[dict] = None,
) -> SessionResult:
    locator = locator or ClassicalBoardLocator()
    s = config.px_per_square
    cal_ev = ActionEvent("calibration", config.calibration_t, "calibrate")
    events = [ActionEvent(f"ply{i + 1}", t, "move") for i, t in enumerate(move_times)]

    if config.naive_frames:
        per_event = {cal_ev.id: _nearest_frames(source, cal_ev), **{e.id: _nearest_frames(source, e) for e in events}}
        total = len(source.timestamps)
        stats = SelectionStats(total, sum(len(v) for v in per_event.values()), sum(len(v) for v in per_event.values()), [])
    else:
        sel_cfg = replace(config.selection)
        cal_cfg = replace(sel_cfg, window_before_s=config.calibration_half_window_s, window_after_s=config.calibration_half_window_s)
        cal_sel = select_frames(source, imu_t, imu_gyro, [cal_ev], cal_cfg)
        sel = select_frames(source, imu_t, imu_gyro, events, sel_cfg)
        per_event = {**cal_sel.per_event, **sel.per_event}
        stats = SelectionStats(
            sel.stats.total_frames,
            sel.stats.decoded_frames + cal_sel.stats.decoded_frames,
            sel.stats.selected_frames + cal_sel.stats.selected_frames,
            sel.stats.degraded_events,
        )

    # ---- calibrate on the starting position
    cal = [_observe(fr.image, locator, s) for fr in per_event[cal_ev.id]]
    cal = [(f, c) for f, c in cal if f is not None]
    if not cal:
        raise CalibrationError("could not find the board in any calibration frame")
    model = SquareModel.fit_from_start(np.mean([f for f, _ in cal], axis=0))
    ref_corners = np.median(np.stack([c for _, c in cal]), axis=0)

    # ---- one move at a time
    board = chess.Board()
    moves: list[chess.Move] = []
    plies: list[dict] = []
    complete = True
    if out_dir:
        os.makedirs(os.path.join(out_dir, "frames"), exist_ok=True)

    for n, ev in enumerate(events, start=1):
        obs: list[tuple[SelectedFrame, Observation]] = []
        for fr in per_event[ev.id]:
            feats, corners = _observe(fr.image, locator, s, ref_corners, config.max_corner_shift_px)
            if feats is not None:
                obs.append((fr, Observation(model.log_posteriors(feats), corners)))
        if obs:  # follow slow head drift between moves
            ref_corners = np.median(np.stack([o.corners for _, o in obs]), axis=0)
        if not obs:
            complete = False
            plies.append({"ply": n, "t": ev.t, "status": "needs_review", "san": None, "uci": None, "confidence": 0.0, "first_guess": None,
                          "alternatives": [], "question": None, "evidence": [], "degraded": True,
                          "note": "board not found in any selected frame"})
            break

        inf = score_moves(board, [o.log_post for _, o in obs])
        d: Decision = decide(board, inf, n, asker, config.decision)
        if d.move is None:  # only possible if there was nothing legal to choose
            complete = False
            break

        evidence = []
        for k, (fr, o) in enumerate(obs):
            name = f"frames/ply{n:02d}_{k}.jpg"
            if out_dir:
                img = _annotate(fr.image, o.corners, d.move)
                cv2.imwrite(os.path.join(out_dir, name), img, [cv2.IMWRITE_JPEG_QUALITY, config.jpeg_quality])
            evidence.append({"frame": name, "t": round(fr.t, 3), "sharpness": round(fr.sharpness, 1), "head_speed": round(fr.head_speed, 3)})

        san = board.san(d.move)
        board.push(d.move)
        moves.append(d.move)
        plies.append(
            {
                "ply": n,
                "t": round(ev.t, 3),
                "san": san,
                "uci": d.move.uci(),
                "fen_after": board.fen(),
                "status": d.status,
                "confidence": round(d.confidence, 4),
                "note": d.note,
                "first_guess": None
                if inf.best.move is None
                else {"san": inf.best.san, "uci": inf.best.move.uci(), "p": round(inf.best.posterior, 4)},
                "alternatives": [
                    {"san": c.san, "uci": c.move.uci() if c.move else None, "p": round(c.posterior, 4)}
                    for c in inf.candidates[:4]
                    if c.move is not None and c.move != d.move and c.posterior >= 0.005
                ],
                "question": None
                if d.question is None
                else {"text": d.question.text, "options": list(d.question.options), "answer": d.answer},
                "evidence": evidence,
                "degraded": ev.id in stats.degraded_events,
            }
        )

    record = {
        "schema": SCHEMA,
        "synthetic": True,
        "meta": meta or {},
        "stats": {
            "total_frames": stats.total_frames,
            "decoded_frames": stats.decoded_frames,
            "selected_frames": stats.selected_frames,
            "degraded_events": stats.degraded_events,
        },
        "plies": plies,
        "pgn": to_pgn(moves, [p["status"] for p in plies if p.get("san")]),
    }
    return SessionResult(record, moves, stats, complete and len(moves) == len(events))


def to_pgn(moves: Sequence[chess.Move], statuses: Sequence[str] = ()) -> str:
    game = chess.pgn.Game()
    game.headers["Event"] = "BoardScribe record"
    game.headers["Site"] = "?"
    game.headers["Date"] = "????.??.??"
    game.headers["Result"] = "*"
    node = game
    for i, mv in enumerate(moves):
        node = node.add_variation(mv)
        if i < len(statuses) and statuses[i] == "needs_review":
            node.comment = "needs review"
    buf = io.StringIO()
    exporter = chess.pgn.FileExporter(buf)
    game.accept(exporter)
    return buf.getvalue().strip()
