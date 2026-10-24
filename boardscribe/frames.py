"""Pick a few sharp frames around each pipetting action.

A first-person video of a bench session is tens of thousands of frames. Only the
moments around a pipetting action matter, and only if the head was still enough
for the frame to be readable.

Selection runs in two phases so that almost nothing is decoded:

1. Timestamps + IMU only. For every action, list the frames inside its time
   window whose head speed is below a limit. No pixels are touched.
2. Decode only those candidates, rank them by sharpness (variance of the
   Laplacian), and keep the best few that are not near-duplicates in time.

If no frame in a window passes the head-speed gate, the least-moving frames are
used and the event is flagged ``degraded`` so the UI can say so.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence

import numpy as np


class FrameSource(Protocol):
    """Random-access video. Decoding frame ``i`` is the expensive operation."""

    timestamps: Sequence[float]

    def get(self, index: int) -> np.ndarray:  # grayscale or BGR, uint8
        ...


@dataclass(frozen=True)
class ActionEvent:
    id: str
    t: float  # seconds, same clock as the video and the IMU
    kind: str  # "aspirate" | "dispense"


@dataclass(frozen=True)
class SelectionConfig:
    window_before_s: float = 1.0
    window_after_s: float = 1.0
    imu_window_s: float = 0.1  # average gyro over +-this around each frame
    max_head_speed: float = 0.15  # rad/s
    max_decode_per_event: int = 12
    keep_per_event: int = 3
    min_gap_s: float = 0.2
    min_rel_sharpness: float = 0.5  # drop frames much blurrier than the best
    min_abs_sharpness: float = 20.0
    fallback_frames: int = 1


@dataclass
class SelectedFrame:
    event_id: str
    index: int
    t: float
    sharpness: float
    head_speed: float
    image: np.ndarray = field(repr=False)


@dataclass
class SelectionStats:
    total_frames: int
    decoded_frames: int
    selected_frames: int
    degraded_events: list[str]

    @property
    def reduction(self) -> float:
        """How many video frames there are per kept frame."""
        return self.total_frames / max(self.selected_frames, 1)


@dataclass
class SelectionResult:
    per_event: dict[str, list[SelectedFrame]]
    stats: SelectionStats


def laplacian_variance(image: np.ndarray) -> float:
    """Sharpness score: variance of the 4-neighbour Laplacian (higher = sharper)."""
    img = image
    if img.ndim == 3:
        img = img.mean(axis=2)
    img = img.astype(np.float64)
    if img.shape[0] < 3 or img.shape[1] < 3:
        return 0.0
    lap = (
        -4.0 * img[1:-1, 1:-1]
        + img[:-2, 1:-1]
        + img[2:, 1:-1]
        + img[1:-1, :-2]
        + img[1:-1, 2:]
    )
    return float(lap.var())


def head_speed_series(
    frame_times: Sequence[float],
    imu_t: Sequence[float],
    imu_gyro: np.ndarray,
    window_s: float,
) -> np.ndarray:
    """Mean gyro magnitude (rad/s) around each frame time. Uses IMU data only."""
    imu_t = np.asarray(imu_t, dtype=np.float64)
    gyro = np.asarray(imu_gyro, dtype=np.float64)
    if gyro.ndim != 2 or gyro.shape[1] != 3 or len(gyro) != len(imu_t):
        raise ValueError("imu_gyro must have shape (n, 3) and match imu_t")
    if len(imu_t) == 0:
        raise ValueError("no IMU samples")
    mag = np.linalg.norm(gyro, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(mag)])
    ft = np.asarray(frame_times, dtype=np.float64)
    lo = np.searchsorted(imu_t, ft - window_s, side="left")
    hi = np.searchsorted(imu_t, ft + window_s, side="right")
    count = hi - lo
    out = np.empty(len(ft))
    nonempty = count > 0
    out[nonempty] = (cum[hi[nonempty]] - cum[lo[nonempty]]) / count[nonempty]
    if not nonempty.all():
        # no IMU sample inside the window: use the nearest one
        nearest = np.clip(np.searchsorted(imu_t, ft[~nonempty]), 0, len(imu_t) - 1)
        out[~nonempty] = mag[nearest]
    return out


def select_frames(
    source: FrameSource,
    imu_t: Sequence[float],
    imu_gyro: np.ndarray,
    events: Sequence[ActionEvent],
    config: SelectionConfig = SelectionConfig(),
) -> SelectionResult:
    times = np.asarray(source.timestamps, dtype=np.float64)
    speed = head_speed_series(times, imu_t, imu_gyro, config.imu_window_s)

    decoded = 0
    per_event: dict[str, list[SelectedFrame]] = {}
    degraded: list[str] = []
    cache: dict[int, np.ndarray] = {}

    def decode(i: int) -> np.ndarray:
        nonlocal decoded
        if i not in cache:
            cache[i] = source.get(i)
            decoded += 1
        return cache[i]

    for ev in events:
        lo = np.searchsorted(times, ev.t - config.window_before_s, side="left")
        hi = np.searchsorted(times, ev.t + config.window_after_s, side="right")
        idx = np.arange(lo, hi)
        if len(idx) == 0:
            per_event[ev.id] = []
            degraded.append(ev.id)
            continue

        steady = idx[speed[idx] <= config.max_head_speed]
        if len(steady) == 0:
            # Nothing was still. Use the calmest frames and say so.
            order = idx[np.argsort(speed[idx], kind="stable")][: config.fallback_frames]
            per_event[ev.id] = [
                SelectedFrame(ev.id, int(i), float(times[i]), laplacian_variance(decode(int(i))), float(speed[i]), decode(int(i)))
                for i in order
            ]
            degraded.append(ev.id)
            continue

        # Phase 1 result: steady candidates. Decode the ones closest to the action.
        steady = steady[np.argsort(np.abs(times[steady] - ev.t), kind="stable")]
        steady = steady[: config.max_decode_per_event]
        scored = [(laplacian_variance(decode(int(i))), int(i)) for i in steady]
        scored.sort(key=lambda s: -s[0])
        best = scored[0][0]

        kept: list[SelectedFrame] = []
        for sharp, i in scored:
            if sharp < config.min_abs_sharpness or sharp < config.min_rel_sharpness * best:
                continue
            if any(abs(times[i] - k.t) < config.min_gap_s for k in kept):
                continue
            kept.append(SelectedFrame(ev.id, i, float(times[i]), sharp, float(speed[i]), decode(i)))
            if len(kept) == config.keep_per_event:
                break

        if not kept:  # steady but too featureless to score
            sharp, i = scored[0]
            kept = [SelectedFrame(ev.id, i, float(times[i]), sharp, float(speed[i]), decode(i))]
            degraded.append(ev.id)
        kept.sort(key=lambda f: f.t)
        per_event[ev.id] = kept

    selected = sum(len(v) for v in per_event.values())
    stats = SelectionStats(
        total_frames=len(times),
        decoded_frames=decoded,
        selected_frames=selected,
        degraded_events=degraded,
    )
    return SelectionResult(per_event, stats)
