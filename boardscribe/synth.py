"""Synthetic first-person chess video. Everything here is rendered, not recorded.

It exists so the pipeline can be tested and measured without real footage, with
exact ground truth. It models the things that make the real problem hard:

* a tilted board with a few palettes (the classifier must calibrate per board),
* a head that sometimes shakes: the IMU shows it and the frame is motion-blurred,
* a hand that carries each piece across the board and occludes squares,
* lighting gradient and sensor noise.

It does not model real pieces, real hands, reflections or real cameras. Results
on this data say the pipeline logic works; they do not say it works on a table.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import chess
import cv2
import numpy as np

from .grid import BoardHomography

TOP_PX = 40  # pixels per square in the flat board texture


@dataclass(frozen=True)
class SceneConfig:
    width: int = 480
    height: int = 360
    fps: int = 10
    seconds_per_ply: float = 6.0
    carry_seconds: float = 2.2
    retreat_seconds: float = 0.7
    p_shake: float = 0.45
    imu_rate: int = 100
    noise_sigma: float = 2.5
    blur_px_per_rad_s: float = 14.0  # motion blur length per rad/s of head speed
    occlusion: bool = True  # draw the hand


# ---------------------------------------------------------------- games


def random_game(seed: int, plies: int, prefer_special: bool = True) -> list[chess.Move]:
    """A random legal game. Castling, en passant and promotion are favoured so they get exercised."""
    rng = np.random.default_rng(seed)
    board = chess.Board()
    moves: list[chess.Move] = []
    while len(moves) < plies and not board.is_game_over():
        legal = list(board.legal_moves)
        w = np.ones(len(legal))
        if prefer_special:
            for i, m in enumerate(legal):
                if board.is_castling(m) or board.is_en_passant(m) or m.promotion:
                    w[i] = 12.0
                elif board.is_capture(m):
                    w[i] = 3.0
        m = legal[int(rng.choice(len(legal), p=w / w.sum()))]
        board.push(m)
        moves.append(m)
    return moves


# ---------------------------------------------------------------- rendering


def _palette(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    hue = float(rng.choice([12, 22, 60, 105, 140])) + float(rng.uniform(-4, 4))
    light = cv2.cvtColor(np.uint8([[[hue, rng.uniform(95, 125), rng.uniform(225, 245)]]]), cv2.COLOR_HSV2BGR)[0, 0]
    dark = cv2.cvtColor(np.uint8([[[hue, rng.uniform(150, 190), rng.uniform(135, 165)]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return light, dark


_PIECE_LETTER = {chess.PAWN: "P", chess.KNIGHT: "N", chess.BISHOP: "B", chess.ROOK: "R", chess.QUEEN: "Q", chess.KING: "K"}


def _draw_piece(img: np.ndarray, cx: float, cy: float, piece: chess.Piece, radius: float) -> None:
    white = piece.color == chess.WHITE
    fill = (232, 234, 238) if white else (38, 36, 40)
    edge = (70, 70, 70) if white else (150, 150, 150)
    ink = (60, 60, 60) if white else (210, 210, 210)
    c = (int(round(cx)), int(round(cy)))
    cv2.circle(img, c, int(radius), fill, -1, cv2.LINE_AA)
    cv2.circle(img, c, int(radius), edge, 2, cv2.LINE_AA)
    letter = _PIECE_LETTER[piece.piece_type]
    scale = radius / 22.0
    (tw, th), _ = cv2.getTextSize(letter, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    cv2.putText(img, letter, (c[0] - tw // 2, c[1] + th // 2), cv2.FONT_HERSHEY_SIMPLEX, scale, ink, 1, cv2.LINE_AA)


def _board_texture(pieces: dict[int, chess.Piece], light: np.ndarray, dark: np.ndarray) -> np.ndarray:
    s = TOP_PX
    img = np.zeros((8 * s, 8 * s, 3), np.uint8)
    for sq in range(64):
        file, rank = chess.square_file(sq), chess.square_rank(sq)
        r, c = 7 - rank, file
        is_light = (file + rank) % 2 == 1
        img[r * s : (r + 1) * s, c * s : (c + 1) * s] = light if is_light else dark
    for sq, piece in pieces.items():
        file, rank = chess.square_file(sq), chess.square_rank(sq)
        _draw_piece(img, (file + 0.5) * s, (7 - rank + 0.5) * s, piece, 0.34 * s)
    return img


def _smooth(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


@dataclass
class SyntheticGame:
    moves: list[chess.Move]
    seed: int
    config: SceneConfig = field(default_factory=SceneConfig)

    def __post_init__(self) -> None:
        cfg = self.config
        rng = np.random.default_rng(self.seed)
        self._light, self._dark = _palette(rng)
        base = np.array([[92, 52], [402, 62], [424, 322], [66, 312]], dtype=np.float64)
        self._corners = (base + rng.uniform(-14, 14, base.shape)).astype(np.float32)
        g0 = rng.uniform(42, 62)
        self._bg = g0 + rng.uniform(-1.5, 1.5, 3)
        self._light_dir = rng.uniform(-1, 1, 2)
        self._phase = rng.uniform(0, 6.28, 4)
        yy, xx = np.mgrid[0 : cfg.height, 0 : cfg.width].astype(np.float32)
        self._grad = 1.0 + 0.06 * (
            (xx / cfg.width - 0.5) * self._light_dir[0] + (yy / cfg.height - 0.5) * self._light_dir[1]
        )

        # positions after k moves
        self.boards: list[chess.Board] = [chess.Board()]
        for m in self.moves:
            b = self.boards[-1].copy()
            b.push(m)
            self.boards.append(b)
        self.captures = [self.boards[i].is_capture(m) for i, m in enumerate(self.moves)]
        self.event_times = [3.4 + cfg.seconds_per_ply * i + float(rng.uniform(-0.2, 0.3)) for i in range(len(self.moves))]
        self.duration = cfg.seconds_per_ply * len(self.moves) + 1.0

        n = int(self.duration * cfg.fps)
        self.timestamps = [i / cfg.fps for i in range(n)]

        # IMU: still baseline plus shake episodes that often overlap the action window
        t = np.arange(0, self.duration, 1.0 / cfg.imu_rate)
        mag = np.abs(rng.normal(0.03, 0.01, len(t)))
        for te in self.event_times:
            if rng.random() < cfg.p_shake:
                start = te + rng.uniform(-1.2, 0.8)
                dur = rng.uniform(0.4, 1.0)
                amp = rng.uniform(0.5, 1.5)
                env = np.clip(1 - np.abs((t - (start + dur / 2)) / (dur / 2)), 0, 1)
                mag += amp * np.sqrt(env)
        direction = rng.normal(size=(len(t), 3))
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)
        self.imu_t = t
        self.imu_gyro = direction * mag[:, None]
        self._mag = mag
        self._tex_cache: dict[tuple, np.ndarray] = {}

    # ---- FrameSource protocol
    def get(self, index: int) -> np.ndarray:
        return self.render(self.timestamps[index], index)

    # ---- scene state
    def _gyro_at(self, t: float) -> float:
        i = int(np.clip(round(t * self.config.imu_rate), 0, len(self._mag) - 1))
        lo, hi = max(0, i - 5), min(len(self._mag), i + 6)
        return float(self._mag[lo:hi].mean())

    def _state(self, t: float):
        """(pieces dict, carried piece or None, hand position in board units or None)."""
        cfg = self.config
        k = sum(1 for te in self.event_times if te <= t)
        pieces = dict(self.boards[k].piece_map())
        carried = None
        hand = None
        if k < len(self.moves):
            m = self.moves[k]
            te = self.event_times[k]
            t_lift = te - cfg.carry_seconds
            if t >= t_lift:
                carried = self.boards[k].piece_at(m.from_square)
                pieces.pop(m.from_square, None)
                if m.promotion and carried is not None:
                    pass  # the pawn is carried; the promoted piece appears at the event
                cap_sq = None
                if self.captures[k]:
                    cap_sq = (
                        chess.square(chess.square_file(m.to_square), chess.square_rank(m.from_square))
                        if self.boards[k].is_en_passant(m)
                        else m.to_square
                    )
                if cap_sq is not None and t >= te - 0.8:
                    pieces.pop(cap_sq, None)
                if self.boards[k].is_castling(m):
                    # king is carried; the rook jumps at the event (handled by boards[k+1])
                    pieces.pop(m.to_square, None)
                s = _smooth((t - t_lift) / cfg.carry_seconds)
                f = (chess.square_file(m.from_square) + 0.5, 7 - chess.square_rank(m.from_square) + 0.5)
                to = (chess.square_file(m.to_square) + 0.5, 7 - chess.square_rank(m.to_square) + 0.5)
                hand = (f[0] + (to[0] - f[0]) * s, f[1] + (to[1] - f[1]) * s)
        if hand is None and k > 0 and cfg.retreat_seconds > 0:
            te = self.event_times[k - 1]
            if t - te < cfg.retreat_seconds:
                m = self.moves[k - 1]
                to = (chess.square_file(m.to_square) + 0.5, 7 - chess.square_rank(m.to_square) + 0.5)
                s = _smooth((t - te) / cfg.retreat_seconds)
                hand = (to[0] + (11.0 - to[0]) * s, to[1] + (11.0 - to[1]) * s)
        return pieces, carried, hand

    def _texture(self, pieces: dict[int, chess.Piece]) -> np.ndarray:
        key = tuple(sorted((sq, p.symbol()) for sq, p in pieces.items()))
        tex = self._tex_cache.get(key)
        if tex is None:
            if len(self._tex_cache) > 64:
                self._tex_cache.clear()
            tex = _board_texture(pieces, self._light, self._dark)
            self._tex_cache[key] = tex
        return tex

    def pose_corners(self, t: float) -> np.ndarray:
        """Board corners in the frame at time t: base pose + small drift + shake displacement."""
        p = self._phase
        dx = 4.0 * np.sin(0.7 * t + p[0]) + 2.0 * np.sin(1.9 * t + p[1])
        dy = 3.0 * np.sin(0.5 * t + p[2]) + 2.0 * np.sin(1.3 * t + p[3])
        w = self._gyro_at(t)
        if w > 0.15:
            dx += 9.0 * w * np.sin(41.0 * t)
            dy += 6.0 * w * np.sin(37.0 * t + 1.0)
        return (self._corners + np.array([dx, dy], dtype=np.float32)).astype(np.float32)

    def render(self, t: float, index: int = 0) -> np.ndarray:
        cfg = self.config
        h, w = cfg.height, cfg.width
        rng = np.random.default_rng((self.seed * 1_000_003 + index) & 0xFFFFFFFF)

        grad = self._grad
        frame = np.empty((h, w, 3), np.float32)
        frame[:] = self._bg
        pieces, carried, hand = self._state(t)
        tex = self._texture(pieces)

        corners = self.pose_corners(t)
        src = np.array([[0, 0], [8 * TOP_PX, 0], [8 * TOP_PX, 8 * TOP_PX], [0, 8 * TOP_PX]], np.float32)
        m = cv2.getPerspectiveTransform(src, corners)
        warped = cv2.warpPerspective(tex, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        mask = cv2.warpPerspective(np.full(tex.shape[:2], 255, np.uint8), m, (w, h), flags=cv2.INTER_NEAREST)
        frame = np.where(mask[..., None] > 0, warped.astype(np.float32), frame)
        frame *= grad[..., None]
        frame = np.clip(frame, 0, 255).astype(np.uint8)

        if hand is not None and cfg.occlusion:
            hom = BoardHomography(corners)
            hx, hy = hom.units_to_px(hand)
            scale = float(np.linalg.norm(np.array(hom.units_to_px((1, 0))) - np.array(hom.units_to_px((0, 0)))))
            if carried is not None:
                _draw_piece(frame, hx, hy - 0.15 * scale, carried, 0.34 * scale)
            cv2.ellipse(frame, (int(hx + 0.35 * scale), int(hy + 0.55 * scale)), (int(0.45 * scale), int(0.8 * scale)), 25, 0, 360, (120, 160, 215), -1, cv2.LINE_AA)

        wmag = self._gyro_at(t)
        length = int(round(wmag * cfg.blur_px_per_rad_s))
        if length >= 2:
            k = np.zeros((length, length), np.float32)
            k[length // 2, :] = 1.0 / length
            frame = cv2.filter2D(frame, -1, k)
        noise = rng.normal(0, cfg.noise_sigma, frame.shape)
        return np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    def write_mp4(self, path: str) -> None:
        cfg = self.config
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), cfg.fps, (cfg.width, cfg.height))
        if not vw.isOpened():
            raise RuntimeError("cannot open VideoWriter (mp4v)")
        for i in range(len(self.timestamps)):
            vw.write(self.get(i))
        vw.release()
