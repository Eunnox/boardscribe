import chess
import cv2
import numpy as np
import pytest

from boardscribe.grid import BoardHomography, ClassicalBoardLocator, square_patches, square_row_col
from boardscribe.synth import SyntheticGame, random_game

CORNERS = [(100, 60), (420, 80), (440, 330), (80, 310)]


def test_square_row_col_follows_image_convention():
    assert square_row_col(chess.A1) == (7, 0)
    assert square_row_col(chess.H8) == (0, 7)
    assert square_row_col(chess.E4) == (4, 4)


def test_homography_round_trip_and_corners():
    h = BoardHomography(CORNERS)
    for pt in [(0, 0), (8, 8), (3.5, 4.5), (7.9, 0.1)]:
        back = h.px_to_units(h.units_to_px(pt))
        assert back == pytest.approx(pt, abs=1e-3)
    assert h.units_to_px((0, 0)) == pytest.approx(CORNERS[0], abs=1e-3)
    assert h.units_to_px((8, 8)) == pytest.approx(CORNERS[2], abs=1e-3)


def test_square_centre_lands_inside_the_right_cell():
    h = BoardHomography(CORNERS)
    a8 = h.square_center_px(chess.A8)
    h1 = h.square_center_px(chess.H1)
    assert a8[0] < h1[0] and a8[1] < h1[1]
    assert h.px_to_units(a8) == pytest.approx((0.5, 0.5), abs=1e-3)
    assert h.px_to_units(h1) == pytest.approx((7.5, 7.5), abs=1e-3)


def test_warp_recovers_each_square_from_a_tilted_view():
    s = 20
    flat = np.zeros((8 * s, 8 * s, 3), np.uint8)
    for sq in range(64):
        r, c = square_row_col(sq)
        flat[r * s : (r + 1) * s, c * s : (c + 1) * s] = (sq * 3 % 256, (sq * 7 + 40) % 256, (sq * 11 + 90) % 256)
    frame = np.zeros((360, 480, 3), np.uint8)
    h = BoardHomography(CORNERS)
    src = np.float32([[0, 0], [8 * s, 0], [8 * s, 8 * s], [0, 8 * s]])
    m = cv2.getPerspectiveTransform(src, np.float32(CORNERS))
    frame = cv2.warpPerspective(flat, m, (480, 360), flags=cv2.INTER_NEAREST)
    patches = square_patches(h.warp(frame, s), s)
    for sq in (chess.A1, chess.H8, chess.E4, chess.B7, chess.G2):
        centre = patches[sq][s // 2 - 2 : s // 2 + 2, s // 2 - 2 : s // 2 + 2].reshape(-1, 3).mean(axis=0)
        r, c = square_row_col(sq)
        assert centre == pytest.approx(flat[r * s + s // 2, c * s + s // 2], abs=2)


def test_square_patches_validates_size():
    with pytest.raises(ValueError):
        square_patches(np.zeros((100, 100, 3), np.uint8), 20)


def test_locator_finds_the_synthetic_board_within_a_few_pixels():
    loc = ClassicalBoardLocator()
    errs = []
    for seed in range(6):
        g = SyntheticGame(random_game(seed, 4), seed=seed)
        frame = g.get(3)
        corners = loc.locate(frame)
        assert corners is not None
        errs.append(np.abs(corners - g.pose_corners(g.timestamps[3])).max())
    assert max(errs) < 6.0


def test_locator_returns_none_without_a_board():
    loc = ClassicalBoardLocator()
    assert loc.locate(np.full((240, 320, 3), 50, np.uint8)) is None
    assert loc.locate(np.zeros((240, 320), np.uint8)) is None
