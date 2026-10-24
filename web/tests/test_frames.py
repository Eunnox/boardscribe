import cv2
import numpy as np
import pytest

from boardscribe.frames import (
    ActionEvent,
    SelectionConfig,
    head_speed_series,
    laplacian_variance,
    select_frames,
)


def textured(seed: int, size=(120, 160)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, size, dtype=np.uint8)
    return cv2.GaussianBlur(img, (3, 3), 0.8)


class FakeVideo:
    """Frames at `fps`. Frames whose time is in a `moving` interval are blurred."""

    def __init__(self, n, fps, moving=()):
        self.timestamps = [i / fps for i in range(n)]
        self.moving = moving
        self.decoded = []
        self.base = textured(1)

    def get(self, i):
        self.decoded.append(i)
        t = self.timestamps[i]
        img = self.base.copy()
        # tiny per-frame change so near-duplicates still differ a bit
        img = np.clip(img.astype(int) + (i % 3), 0, 255).astype(np.uint8)
        if any(a <= t <= b for a, b in self.moving):
            img = cv2.GaussianBlur(img, (0, 0), 4.0)
        return img


def imu(duration, rate=100, moving=(), still=0.02, shake=1.0):
    t = np.arange(0, duration, 1 / rate)
    mag = np.full(len(t), still)
    for a, b in moving:
        mag[(t >= a) & (t <= b)] = shake
    gyro = np.stack([mag, np.zeros_like(mag), np.zeros_like(mag)], axis=1)
    return t, gyro


def test_laplacian_variance_matches_opencv_and_drops_with_blur():
    img = textured(3).astype(np.float64)
    cv_lap = cv2.Laplacian(img, cv2.CV_64F, ksize=1)[1:-1, 1:-1]
    assert laplacian_variance(img) == pytest.approx(cv_lap.var(), rel=1e-9)
    blurred = cv2.GaussianBlur(img, (0, 0), 3.0)
    assert laplacian_variance(blurred) < 0.05 * laplacian_variance(img)


def test_laplacian_variance_accepts_color_and_tiny_images():
    assert laplacian_variance(np.zeros((2, 2), np.uint8)) == 0.0
    bgr = np.dstack([textured(4)] * 3)
    assert laplacian_variance(bgr) > 0


def test_head_speed_series_averages_imu_window():
    t = np.arange(0, 2, 0.01)
    gyro = np.zeros((len(t), 3))
    gyro[t >= 1.0, 0] = 2.0
    speed = head_speed_series([0.5, 1.5, 1.0], t, gyro, window_s=0.1)
    assert speed[0] == pytest.approx(0.0)
    assert speed[1] == pytest.approx(2.0)
    assert 0.8 < speed[2] < 1.2  # straddles the step


def test_head_speed_series_validates_shapes():
    with pytest.raises(ValueError):
        head_speed_series([0.0], [0.0, 0.1], np.zeros((1, 3)), 0.1)
    with pytest.raises(ValueError):
        head_speed_series([0.0], [], np.zeros((0, 3)), 0.1)


def test_frames_while_head_moves_are_never_selected():
    moving = [(4.5, 5.2)]
    video = FakeVideo(100, 10, moving)
    t, g = imu(10, moving=moving)
    res = select_frames(video, t, g, [ActionEvent("a1", 5.0, "dispense")])
    chosen = res.per_event["a1"]
    assert chosen, "expected frames outside the shaking interval"
    for f in chosen:
        assert not (4.5 <= f.t <= 5.2)
        assert f.head_speed <= SelectionConfig().max_head_speed
    assert res.stats.degraded_events == []


def test_selected_frames_are_sharp_and_spaced():
    video = FakeVideo(100, 10)
    t, g = imu(10)
    cfg = SelectionConfig(keep_per_event=3, min_gap_s=0.3)
    res = select_frames(video, t, g, [ActionEvent("a1", 5.0, "aspirate")], cfg)
    chosen = res.per_event["a1"]
    assert 1 <= len(chosen) <= 3
    times = sorted(f.t for f in chosen)
    assert all(b - a >= 0.3 - 1e-9 for a, b in zip(times, times[1:]))
    assert all(f.sharpness >= cfg.min_abs_sharpness for f in chosen)


def test_whole_window_shaking_falls_back_and_flags_event():
    moving = [(3.0, 7.0)]
    video = FakeVideo(100, 10, moving)
    t, g = imu(10, moving=moving)
    res = select_frames(video, t, g, [ActionEvent("a1", 5.0, "dispense")])
    assert res.stats.degraded_events == ["a1"]
    assert len(res.per_event["a1"]) == 1


def test_event_outside_video_is_degraded_not_crash():
    video = FakeVideo(20, 10)
    t, g = imu(2)
    res = select_frames(video, t, g, [ActionEvent("late", 50.0, "dispense")])
    assert res.per_event["late"] == []
    assert "late" in res.stats.degraded_events


def test_long_video_decodes_only_a_few_frames():
    fps, minutes = 15, 10
    n = fps * 60 * minutes  # 9000 frames
    video = FakeVideo(n, fps)
    t, g = imu(n / fps)
    events = [ActionEvent(f"e{i}", 30.0 + 90.0 * i, "dispense") for i in range(6)]
    res = select_frames(video, t, g, events)
    assert res.stats.total_frames == n
    assert res.stats.selected_frames <= 6 * 3
    assert res.stats.decoded_frames <= 6 * SelectionConfig().max_decode_per_event
    assert len(video.decoded) == res.stats.decoded_frames  # nothing decoded twice
    assert res.stats.reduction > 400
