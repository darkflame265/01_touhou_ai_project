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
    if gray_roi is None or gray_roi.size == 0:
        return 0.0, 0.0
    return float(gray_roi.mean()), float(gray_roi.std())


def bright_ratio(gray_roi: np.ndarray, thr: int) -> float:
    if gray_roi is None or gray_roi.size == 0:
        return 0.0
    return float((gray_roi > int(thr)).mean())


# =========================
# 여기부터 "screen.py에서 빼낼" 계산 로직
# =========================

@dataclass
class UiPanelHeuristics:
    edge_ratio_thr: float = 0.040
    std_min: float = 15.0
    std_max: float = 80.0
    mean_min: float = 20.0
    mean_max: float = 200.0


def ui_panel_present_cached(
    panel_gray: np.ndarray,
    *,
    frame_idx: int,
    edge_cache: CannyEdgeRatioCache,
    heur: UiPanelHeuristics,
) -> bool:
    """
    panel ROI만 받아서 UI 패널 존재 여부 판정.
    - edge_ratio는 캐시 + 다운샘플 ROI 적용
    """
    if panel_gray is None or panel_gray.size == 0:
        return False

    mean, std = mean_std(panel_gray)
    edge_ratio = edge_cache.edge_ratio(panel_gray, frame_idx)

    ok = (
        (edge_ratio >= float(heur.edge_ratio_thr)) and
        (float(heur.std_min) <= std <= float(heur.std_max)) and
        (float(heur.mean_min) <= mean <= float(heur.mean_max))
    )
    return bool(ok)


@dataclass
class DangerWeights:
    w_edge: float = 4.0
    w_bright: float = 2.0
    w_std: float = 1.2
    bright_thr: int = 160


def danger_from_playfield_cached(
    danger_roi_gray: np.ndarray,
    *,
    frame_idx: int,
    edge_cache: CannyEdgeRatioCache,
    weights: DangerWeights,
    return_parts: bool = False,
):
    """
    danger ROI만 받아서 danger 계산.
    - edge_ratio는 캐시 + 다운샘플 ROI 적용
    """
    if danger_roi_gray is None or danger_roi_gray.size == 0:
        if return_parts:
            return 0.0, 0.0, 0.0, 0.0
        return 0.0

    edge_ratio = edge_cache.edge_ratio(danger_roi_gray, frame_idx)
    b_ratio = bright_ratio(danger_roi_gray, int(weights.bright_thr))
    std_norm = float(danger_roi_gray.std()) / 255.0

    danger = 0.0
    danger += float(weights.w_edge) * float(edge_ratio)
    danger += float(weights.w_bright) * float(b_ratio)
    danger += float(weights.w_std) * float(std_norm)
    danger = max(0.0, min(1.0, danger))

    if return_parts:
        return float(danger), float(edge_ratio), float(b_ratio), float(std_norm)
    return float(danger)
