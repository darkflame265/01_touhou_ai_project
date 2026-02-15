from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

import cv2
import numpy as np

Pt = Tuple[float, float]  # (x,y) in ROI pixels


def ensure_uint8_bgr(frame: np.ndarray) -> np.ndarray:
    """DXCam/기타 캡처가 BGRA/float인 경우를 포함해 uint8 BGR로 정규화."""
    if frame is None or frame.size == 0:
        return frame

    if frame.dtype != np.uint8:
        f = frame.astype(np.float32)
        mx = float(np.nanmax(f)) if f.size else 0.0
        if mx <= 1.5:
            f *= 255.0
        frame = np.clip(f, 0, 255).astype(np.uint8)

    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.ndim == 3 and frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    if not frame.flags["C_CONTIGUOUS"]:
        frame = np.ascontiguousarray(frame)
    return frame


@dataclass
class BulletTrackerConfig:
    # 검출: 밝은 픽셀(탄막) 기반 1차 후보 생성
    use_hsv: bool = True
    hsv_v_min: int = 190
    hsv_s_min: int = 15
    hsv_s_max: int = 255
    hsv_h_min: int = 0
    hsv_h_max: int = 179

    # 흰색 계열 보강
    use_white: bool = True
    white_min: int = 210

    # morphology
    open_ks: int = 3
    open_iter: int = 1
    dilate_iter: int = 1

    # 후보 필터 (탄막은 보통 작다)
    area_min: int = 3
    area_max: int = 300
    w_min: int = 1
    w_max: int = 40
    h_min: int = 1
    h_max: int = 40

    # 출력
    max_candidates: int = 256
    topk: int = 16  # player 기준 가까운 순으로 최대 K개 반환

    # 디버그
    debug_max_draw: int = 120


class BulletTrackerCV:
    """
    입력: playfield ROI(BGR), player_center_roi(optional)
    출력: topk 탄막 중심점 리스트(ROI 좌표계)

    초기 버전은 "밝기 기반 후보 + size filter"만.
    정확도는 부족할 수 있지만, MLP 벡터화/파이프라인 구축용으로는 충분함.
    """

    def __init__(self, cfg: Optional[BulletTrackerConfig] = None):
        self.cfg = cfg or BulletTrackerConfig()
        self._open_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (int(self.cfg.open_ks), int(self.cfg.open_ks))
        )
        self.last_mask_u8: Optional[np.ndarray] = None
        self.last_points_roi: List[Pt] = []
        self.last_points_topk_roi: List[Pt] = []
        self._dbg: Dict[str, Any] = {}

    def reset(self) -> None:
        self.last_mask_u8 = None
        self.last_points_roi = []
        self.last_points_topk_roi = []
        self._dbg = {}

    def _build_mask(self, roi_bgr: np.ndarray) -> np.ndarray:
        roi_bgr = ensure_uint8_bgr(roi_bgr)
        h, w = roi_bgr.shape[:2]
        if h <= 0 or w <= 0:
            return np.zeros((1, 1), np.uint8)

        masks = []

        if self.cfg.use_hsv:
            hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
            hh, ss, vv = cv2.split(hsv)

            m_v = (vv >= int(self.cfg.hsv_v_min))
            m_s = (ss >= int(self.cfg.hsv_s_min)) & (ss <= int(self.cfg.hsv_s_max))
            m_h = (hh >= int(self.cfg.hsv_h_min)) & (hh <= int(self.cfg.hsv_h_max))
            m = (m_v & m_s & m_h).astype(np.uint8) * 255
            masks.append(m)

        if self.cfg.use_white:
            g = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
            mw = (g >= int(self.cfg.white_min)).astype(np.uint8) * 255
            masks.append(mw)

        if not masks:
            return np.zeros((h, w), np.uint8)

        mask = masks[0]
        for mm in masks[1:]:
            mask = cv2.bitwise_or(mask, mm)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._open_kernel, iterations=int(self.cfg.open_iter))
        if int(self.cfg.dilate_iter) > 0:
            mask = cv2.dilate(mask, None, iterations=int(self.cfg.dilate_iter))
        return mask

    def _extract_points(self, mask_u8: np.ndarray) -> List[Pt]:
        if mask_u8 is None or mask_u8.size == 0:
            return []

        cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        pts: List[Pt] = []
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)

            if not (self.cfg.w_min <= w <= self.cfg.w_max and self.cfg.h_min <= h <= self.cfg.h_max):
                continue

            area = float(w * h)
            if not (float(self.cfg.area_min) <= area <= float(self.cfg.area_max)):
                continue

            cx = float(x + 0.5 * w)
            cy = float(y + 0.5 * h)
            pts.append((cx, cy))

            if len(pts) >= int(self.cfg.max_candidates):
                break

        return pts

    def step(self, roi_bgr: np.ndarray, player_center_roi: Optional[Tuple[int, int]] = None) -> List[Pt]:
        roi_bgr = ensure_uint8_bgr(roi_bgr)

        mask = self._build_mask(roi_bgr)
        pts = self._extract_points(mask)

        if player_center_roi is not None and pts:
            px, py = map(float, player_center_roi)
            d2 = [((x - px) ** 2 + (y - py) ** 2, (x, y)) for (x, y) in pts]
            d2.sort(key=lambda t: t[0])
            topk = [p for _, p in d2[: int(self.cfg.topk)]]
        else:
            topk = pts[: int(self.cfg.topk)]

        self.last_mask_u8 = mask
        self.last_points_roi = pts
        self.last_points_topk_roi = topk

        self._dbg = {
            "n": int(len(pts)),
            "topk": int(len(topk)),
            "points": pts[: int(self.cfg.debug_max_draw)],
            "points_topk": topk,
            "player_center_roi": player_center_roi,
            "roi_shape": tuple(map(int, roi_bgr.shape[:2])),
        }
        return topk

    def get_debug(self) -> Dict[str, Any]:
        return self._dbg or {}
