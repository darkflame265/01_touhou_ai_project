# env/screen_util/metrics.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


def downsample_gray(gray: np.ndarray, max_side: int = 160) -> np.ndarray:
    """
    ROI가 너무 크면 max_side 기준으로 축소 (INTER_AREA).
    - Canny/통계(mean/std) 계산 비용을 줄이기 위함.
    """
    if gray is None or gray.size == 0:
        return gray

    h, w = gray.shape[:2]
    if max_side <= 0:
        return gray

    m = max(h, w)
    if m <= max_side:
        return gray

    scale = max_side / float(m)
    nh = max(1, int(round(h * scale)))
    nw = max(1, int(round(w * scale)))
    return cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)


@dataclass
class CannyCacheConfig:
    every_n_frames: int = 3     # N프레임마다만 Canny 갱신
    max_side: int = 160         # ROI 다운샘플 최대 변 길이
    thr1: int = 80
    thr2: int = 160


class CannyEdgeRatioCache:
    """
    edge_ratio = (Canny 결과에서 edge 픽셀 비율)
    - 매 프레임 Canny를 돌리지 않고 every_n_frames마다만 갱신
    - ROI는 downsample_gray로 축소해서 계산
    """
    def __init__(self, cfg: CannyCacheConfig):
        self.cfg = cfg
        self._last_update_frame: int = -1
        self._cached_edge_ratio: float = 0.0
        self._cached_shape: Optional[Tuple[int, int]] = None

    def reset(self):
        self._last_update_frame = -1
        self._cached_edge_ratio = 0.0
        self._cached_shape = None

    def edge_ratio(self, gray_roi: np.ndarray, frame_idx: int) -> float:
        if gray_roi is None or gray_roi.size == 0:
            self._cached_edge_ratio = 0.0
            self._cached_shape = None
            return 0.0

        every = max(1, int(self.cfg.every_n_frames))

        # ROI 사이즈가 크게 바뀌면 바로 갱신(캐시 무의미해짐)
        h, w = gray_roi.shape[:2]
        shape = (h, w)

        need_update = (
            self._last_update_frame < 0
            or (frame_idx - self._last_update_frame) >= every
            or (self._cached_shape != shape)
        )

        if not need_update:
            return float(self._cached_edge_ratio)

        roi_small = downsample_gray(gray_roi, max_side=int(self.cfg.max_side))
        if roi_small is None or roi_small.size == 0:
            self._cached_edge_ratio = 0.0
            self._cached_shape = shape
            self._last_update_frame = frame_idx
            return 0.0

        edges = cv2.Canny(roi_small, int(self.cfg.thr1), int(self.cfg.thr2))
        self._cached_edge_ratio = float((edges > 0).mean())

        self._cached_shape = shape
        self._last_update_frame = frame_idx
        return float(self._cached_edge_ratio)


def mean_std(gray_roi: np.ndarray) -> Tuple[float, float]:
    """
    gray ROI의 mean/std
    """
    if gray_roi is None or gray_roi.size == 0:
        return 0.0, 0.0
    return float(gray_roi.mean()), float(gray_roi.std())


def bright_ratio(gray_roi: np.ndarray, thr: int) -> float:
    """
    gray ROI에서 (pixel > thr) 비율
    """
    if gray_roi is None or gray_roi.size == 0:
        return 0.0
    return float((gray_roi > int(thr)).mean())
