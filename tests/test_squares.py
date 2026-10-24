import chess
import numpy as np

from boardscribe.grid import BoardHomography, ClassicalBoardLocator, square_patches
from boardscribe.squares import (
    BLACK,
    EMPTY,
    WHITE,
    SquareModel,
    classify,
    is_light_square,
    occupancy_classes,
    square_features,
)
from boardscribe.synth import SyntheticGame, random_game

S = 32


def features_at(game, index, locator=ClassicalBoardLocator()):
    frame = game.get(index)
    corners = locator.locate(frame)
    return square_features(square_patches(BoardHomography(corners).warp(frame, S), S))


def test_occupancy_of_start_position():
    occ = occupancy_classes(chess.Board())
    assert (occ[:16] == WHITE).all() and (occ[48:] == BLACK).all() and (occ[16:48] == EMPTY).all()


def test_light_and_dark_squares():
    assert not is_light_square(chess.A1) and is_light_square(chess.H1) and is_light_square(chess.A8)


def test_calibrated_on_start_position_it_reads_later_boards():
    correct = total = 0
    for seed in range(4):
        g = SyntheticGame(random_game(seed, 10), seed=seed)
        model = SquareModel.fit_from_start(np.mean([features_at(g, i) for i in (3, 5, 7)], axis=0))
        for k in range(1, 8):
            i = int((g.event_times[k - 1] + 1.3) * g.config.fps)  # hand has left
            pred = classify(model.log_posteriors(features_at(g, i)))
            truth = occupancy_classes(g.boards[k])
            correct += int((pred == truth).sum())
            total += 64
    assert correct / total > 0.98


def test_garbage_square_gets_a_flat_posterior():
    g = SyntheticGame(random_game(0, 4), seed=0)
    f = np.mean([features_at(g, i) for i in (3, 5, 7)], axis=0)
    model = SquareModel.fit_from_start(f)
    noisy = f.copy()
    noisy[27] = noisy[27] + 150.0  # a hand, a glare, anything far from all three classes
    post = np.exp(model.log_posteriors(noisy))
    assert post[27].max() < 0.5
    assert np.allclose(post.sum(axis=1), 1.0)


def test_fit_needs_64_squares():
    import pytest

    with pytest.raises(ValueError):
        SquareModel.fit_from_start(np.zeros((10, 7)))
