"""Chessboard geometry: four corners -> homography -> 64 top-down square patches.

The board is a known object, so once the four corners of the 8x8 playing area
are known, every square can be cut out of the image as a flat patch no matter
how the camera is tilted.

Corners come from a ``BoardLocator``. In a full system that would be a
segmentation model; ``ClassicalBoardLocator`` below is a plain OpenCV stand-in
(a brighter board on a darker table) that works on the synthetic frames here.

Square indices follow python-chess: a1 = 0, b1 = 1, ... h8 = 63.
Image convention: the camera sits on White's side, so a8 is top-left.
"""

from __future__ import annotations

from typing import Protocol, Sequence

import chess
import cv2
import numpy as np

BOARD_CORNERS_UNITS = np.array([[0, 0], [8, 0], [8, 8], [0, 8]], dtype=np.float32)  # TL, TR, BR, BL


def square_row_col(square: int) -> tuple[int, int]:
    """(row, col) of a square in the top-down image: row 0 is rank 8, col 0 is file a."""
    return 7 - chess.square_rank(square), chess.square_file(square)


class BoardHomography:
    """Pixel <-> board-unit mapping (1 unit = 1 square) from corners TL, TR, BR, BL."""

    def __init__(self, corners_px: Sequence[Sequence[float]]):
        c = np.asarray(corners_px, dtype=np.float32)
        if c.shape != (4, 2):
            raise ValueError("expected four (x, y) corners")
        self.corners_px = c
        self._px_to_units = cv2.getPerspectiveTransform(c, BOARD_CORNERS_UNITS)
        self._units_to_px = cv2.getPerspectiveTransform(BOARD_CORNERS_UNITS, c)

    def units_to_px(self, pt: Sequence[float]) -> tuple[float, float]:
        out = cv2.perspectiveTransform(np.array([[pt]], dtype=np.float32), self._units_to_px)[0, 0]
        return float(out[0]), float(out[1])

    def px_to_units(self, pt: Sequence[float]) -> tuple[float, float]:
        out = cv2.perspectiveTransform(np.array([[pt]], dtype=np.float32), self._px_to_units)[0, 0]
        return float(out[0]), float(out[1])

    def square_center_px(self, square: int) -> tuple[float, float]:
        r, c = square_row_col(square)
        return self.units_to_px((c + 0.5, r + 0.5))

    def warp(self, image: np.ndarray, px_per_square: int = 32) -> np.ndarray:
        """Top-down view of the board, ``8 * px_per_square`` pixels on a side."""
        s = float(px_per_square)
        scale = np.diag([s, s, 1.0])
        m = scale @ self._px_to_units
        size = 8 * px_per_square
        return cv2.warpPerspective(image, m, (size, size), flags=cv2.INTER_AREA)


def square_patches(top_down: np.ndarray, px_per_square: int) -> np.ndarray:
    """Cut a top-down board image into 64 patches indexed like python-chess squares."""
    s = px_per_square
    if top_down.shape[0] != 8 * s or top_down.shape[1] != 8 * s:
        raise ValueError("top-down image size does not match px_per_square")
    out = np.empty((64, s, s) + top_down.shape[2:], dtype=top_down.dtype)
    for sq in range(64):
        r, c = square_row_col(sq)
        out[sq] = top_down[r * s : (r + 1) * s, c * s : (c + 1) * s]
    return out


class BoardLocator(Protocol):
    def locate(self, image: np.ndarray) -> np.ndarray | None:
        """Return four playing-area corners (TL, TR, BR, BL) in pixels, or None."""


def order_corners(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = pts[:, 0] - pts[:, 1]
    return np.array([pts[np.argmin(s)], pts[np.argmax(d)], pts[np.argmax(s)], pts[np.argmin(d)]], dtype=np.float32)


class ClassicalBoardLocator:
    """A coloured board on a neutral table -> its largest quadrilateral.

    Thresholds saturation (Otsu), fills the holes where pieces stand, and fits a
    quadrilateral to the convex hull. A stand-in for a segmentation model.
    """

    def __init__(self, min_area_frac: float = 0.15):
        self.min_area_frac = min_area_frac

    def locate(self, image: np.ndarray) -> np.ndarray | None:
        if image.ndim != 3:
            return None
        smooth = cv2.GaussianBlur(image, (0, 0), 3.0)  # sensor noise would otherwise look like colour
        sat = cv2.cvtColor(smooth, cv2.COLOR_BGR2HSV)[..., 1]
        _, mask = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cnt) < self.min_area_frac * image.shape[0] * image.shape[1]:
            return None
        hull = cv2.convexHull(cnt)
        peri = cv2.arcLength(hull, True)
        for eps in (0.01, 0.02, 0.03, 0.05, 0.08):
            approx = cv2.approxPolyDP(hull, eps * peri, True)
            if len(approx) == 4:
                return order_corners(approx.reshape(4, 2))
        return None
