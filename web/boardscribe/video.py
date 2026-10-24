"""Random-access video file as a FrameSource (OpenCV)."""

from __future__ import annotations

import csv
from typing import Sequence

import cv2
import numpy as np


class VideoFileSource:
    def __init__(self, path: str):
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise FileNotFoundError(f"cannot open video: {path}")
        fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        n = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = fps
        self.timestamps: Sequence[float] = [i / fps for i in range(n)]

    def get(self, index: int) -> np.ndarray:
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self._cap.read()
        if not ok:
            raise IndexError(f"cannot decode frame {index}")
        return frame

    def close(self) -> None:
        self._cap.release()


def read_imu_csv(path: str) -> tuple[np.ndarray, np.ndarray]:
    """CSV with header t,gx,gy,gz (seconds, rad/s) -> (t, gyro[n, 3])."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    t = np.array([float(r["t"]) for r in rows])
    g = np.array([[float(r["gx"]), float(r["gy"]), float(r["gz"])] for r in rows])
    return t, g


def write_imu_csv(path: str, t: np.ndarray, gyro: np.ndarray) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "gx", "gy", "gz"])
        for ti, (x, y, z) in zip(t, gyro):
            w.writerow([f"{ti:.4f}", f"{x:.5f}", f"{y:.5f}", f"{z:.5f}"])
