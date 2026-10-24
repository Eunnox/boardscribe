"""Per-square occupancy model: empty / white piece / black piece.

No pretrained weights. The starting position is known, so the first stable
frame labels all 64 squares for free (32 empty, 16 white, 16 black, on both
light and dark squares). A diagonal-Gaussian model is fitted on those labels for
this board, this lighting and this camera. Later frames are scored against it.

The likelihood is a narrow Gaussian core, a wide tail and a uniform "anything at
all" component, so a square covered by a hand does not produce a confident wrong
answer; it produces a flat posterior and carries little weight.
"""

from __future__ import annotations

from dataclasses import dataclass

import chess
import cv2
import numpy as np

EMPTY, WHITE, BLACK = 0, 1, 2
N_CLASSES = 3

# Feature order: L, a, b of the square's centre; texture (std of L); then the
# centre minus the square's own corners (where no piece stands) for L, a, b.
FEATURE_FLOOR = np.array([5.0, 3.0, 3.0, 4.0, 5.0, 3.0, 3.0])
TAIL_SCALE = 6.0
TAIL_WEIGHT = 0.03
OUTLIER_WEIGHT = 0.01
OUTLIER_RANGE = 100.0  # an "anything at all" component: uniform over this range per feature


def is_light_square(square: int) -> bool:
    return (chess.square_file(square) + chess.square_rank(square)) % 2 == 1


def occupancy_classes(board: chess.Board) -> np.ndarray:
    """Class (EMPTY/WHITE/BLACK) of every square, indexed like python-chess squares."""
    out = np.zeros(64, dtype=np.int64)
    for sq, piece in board.piece_map().items():
        out[sq] = WHITE if piece.color == chess.WHITE else BLACK
    return out


def square_features(patches_bgr: np.ndarray) -> np.ndarray:
    """(64, S, S, 3) uint8 BGR patches -> (64, 7) float features."""
    n, s = patches_bgr.shape[0], patches_bgr.shape[1]
    lab = np.stack([cv2.cvtColor(p, cv2.COLOR_BGR2LAB) for p in patches_bgr]).astype(np.float64)
    c0, c1 = int(0.3 * s), int(0.7 * s)
    centre = lab[:, c0:c1, c0:c1, :]
    inner = lab[:, int(0.15 * s) : int(0.85 * s), int(0.15 * s) : int(0.85 * s), 0]
    k = max(2, int(0.18 * s))
    corners = np.concatenate(
        [
            lab[:, :k, :k, :].reshape(n, -1, 3),
            lab[:, :k, -k:, :].reshape(n, -1, 3),
            lab[:, -k:, :k, :].reshape(n, -1, 3),
            lab[:, -k:, -k:, :].reshape(n, -1, 3),
        ],
        axis=1,
    ).mean(axis=1)
    cm = centre.reshape(n, -1, 3).mean(axis=1)
    tex = inner.reshape(n, -1).std(axis=1)
    return np.concatenate([cm, tex[:, None], cm - corners], axis=1)


@dataclass
class SquareModel:
    mean: np.ndarray  # (2 colours, 3 classes, D)
    std: np.ndarray  # (2, 3, D)

    @classmethod
    def fit_from_start(cls, features: np.ndarray) -> "SquareModel":
        """Fit from the features of the 64 squares of the starting position."""
        if features.shape[0] != 64:
            raise ValueError("expected features for 64 squares")
        labels = occupancy_classes(chess.Board())
        d = features.shape[1]
        mean = np.zeros((2, N_CLASSES, d))
        var = np.zeros((2, N_CLASSES, d))
        pooled = np.zeros((N_CLASSES, d))
        for k in range(N_CLASSES):
            sel = labels == k
            pooled[k] = features[sel].var(axis=0)
        for colour in (0, 1):
            for k in range(N_CLASSES):
                sel = np.array([labels[s] == k and int(is_light_square(s)) == colour for s in range(64)])
                f = features[sel]
                mean[colour, k] = f.mean(axis=0)
                var[colour, k] = 0.5 * f.var(axis=0) + 0.5 * pooled[k]
        std = np.sqrt(var)
        std = np.maximum(std, FEATURE_FLOOR)
        return cls(mean, std)

    def log_posteriors(self, features: np.ndarray) -> np.ndarray:
        """(64, 3) log P(class | features), uniform prior, robust to occlusion."""
        out = np.empty((64, N_CLASSES))
        for sq in range(64):
            colour = int(is_light_square(sq))
            mu, sd = self.mean[colour], self.std[colour]  # (3, D)
            z = (features[sq][None, :] - mu) / sd
            core = -0.5 * (z**2).sum(axis=1) - np.log(sd).sum(axis=1)
            tail = -0.5 * ((z / TAIL_SCALE) ** 2).sum(axis=1) - np.log(sd * TAIL_SCALE).sum(axis=1)
            uniform = np.log(OUTLIER_WEIGHT) - mu.shape[1] * np.log(OUTLIER_RANGE)
            ll = np.logaddexp.reduce(
                np.stack([np.log(1 - TAIL_WEIGHT - OUTLIER_WEIGHT) + core, np.log(TAIL_WEIGHT) + tail, np.full(3, uniform)]),
                axis=0,
            )
            ll -= np.logaddexp.reduce(ll)
            out[sq] = ll
        return out


def classify(log_post: np.ndarray) -> np.ndarray:
    return log_post.argmax(axis=1)
