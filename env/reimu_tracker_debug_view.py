# env/reimu_tracker_debug_view.py
"""
ReimuTrackerCV 디버그 뷰 (touhou_02 스타일 그대로)

touhou_02(reimu_track_test.py)와 동일하게:
- ROI: 파랑 사각형 + 파랑 텍스트(scale=0.6, thickness=2)
- candidates: 노랑 박스(thickness=2)  [UNLOCK에서만]
- lock_cand: 주황 박스(thickness=3) + 주황 텍스트(scale=0.55, thickness=2)  [UNLOCK에서만]
- locked: 초록 박스(thickness=2) + 초록 텍스트(scale=0.6, thickness=2)

요청사항 반영:
- LOCK 확정 텍스트는 "w x h"만 표시
- ✅ LOCK 상태에서는 candidates/lock_cand를 그리지 않음(touhou_02 느낌 그대로)
- R: tracker.reset()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any

import cv2
import numpy as np

from env.reimu_tracker_cv import ReimuTrackerCV, BBox


@dataclass
class DebugViewConfig:
    window_name: str = "debug"  # touhou_02는 "debug" 창 이름

    # thickness (touhou_02 동일)
    roi_thickness: int = 2
    cand_thickness: int = 2
    lock_cand_thickness: int = 3
    locked_thickness: int = 2

    # font (touhou_02 동일)
    font = cv2.FONT_HERSHEY_SIMPLEX
    roi_text_scale: float = 0.6
    roi_text_thickness: int = 2

    locked_text_scale: float = 0.6
    locked_text_thickness: int = 2

    lock_cand_text_scale: float = 0.55
    lock_cand_text_thickness: int = 2

    # colors (BGR) touhou_02 동일
    color_roi: Tuple[int, int, int] = (255, 0, 0)         # Blue
    color_candidates: Tuple[int, int, int] = (0, 255, 255) # Yellow
    color_lock_cand: Tuple[int, int, int] = (0, 128, 255)  # Orange
    color_locked: Tuple[int, int, int] = (0, 255, 0)       # Green

    # key handling
    enable_keys: bool = True
    wait_ms: int = 1


def _clamp_bbox(b: BBox, W: int, H: int) -> BBox:
    x, y, w, h = map(int, b)
    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))
    return (x, y, w, h)


def _draw_bbox(img: np.ndarray, bbox: BBox, color: Tuple[int, int, int], thickness: int):
    x, y, w, h = bbox
    cv2.rectangle(img, (x, y), (x + w, y + h), color, int(thickness))


class ReimuTrackerDebugView:
    def __init__(self, tracker: ReimuTrackerCV, cfg: Optional[DebugViewConfig] = None):
        self.tracker = tracker
        self.cfg = cfg or DebugViewConfig()
        self._window_inited = False

    def _ensure_window(self):
        if self._window_inited:
            return
        try:
            cv2.namedWindow(self.cfg.window_name, cv2.WINDOW_NORMAL)
        except Exception:
            pass
        self._window_inited = True

    def close(self):
        if self._window_inited:
            try:
                cv2.destroyWindow(self.cfg.window_name)
            except Exception:
                pass
        self._window_inited = False

    def handle_key(self, key: int):
        if key in (ord("r"), ord("R")):
            self.tracker.reset()

    def render(self, frame_bgr: np.ndarray) -> int:
        """
        tracker.step(frame) 이후 호출.
        내부에서 tracker.get_debug() 읽어서 touhou_02 스타일로 그린다.
        """
        self._ensure_window()

        if frame_bgr is None or frame_bgr.size == 0:
            return -1

        vis = frame_bgr.copy()
        H, W = vis.shape[:2]

        try:
            dbg: Dict[str, Any] = self.tracker.get_debug() or {}
        except Exception:
            dbg = {}

        locked = bool(dbg.get("locked", False))

        # ROI 표시 + 안내 텍스트 (touhou_02 동일)
        roi = dbg.get("roi_xyxy", None)
        if roi is not None:
            x0, y0, x1, y1 = roi
            x0 = int(np.clip(x0, 0, W - 1))
            y0 = int(np.clip(y0, 0, H - 1))
            x1 = int(np.clip(x1, x0 + 1, W))
            y1 = int(np.clip(y1, y0 + 1, H))
            cv2.rectangle(vis, (x0, y0), (x1, y1), self.cfg.color_roi, int(self.cfg.roi_thickness))
            cv2.putText(
                vis,
                "ROI (press R to re-detect, ESC to quit)",
                (x0 + 6, y0 + 22),
                self.cfg.font,
                float(self.cfg.roi_text_scale),
                self.cfg.color_roi,
                int(self.cfg.roi_text_thickness),
            )

        # ✅ LOCK 상태에서는 candidates/lock_cand를 그리지 않음 (touhou_02 느낌)
        if not locked:
            # candidates (yellow)
            cands: List[BBox] = dbg.get("candidates", []) or []
            for b in cands:
                b2 = _clamp_bbox(tuple(b), W, H)
                _draw_bbox(vis, b2, self.cfg.color_candidates, self.cfg.cand_thickness)

            # lock candidate (orange) + 텍스트
            lc = dbg.get("lock_cand", None)
            if lc is not None:
                b2 = _clamp_bbox(tuple(lc), W, H)
                _draw_bbox(vis, b2, self.cfg.color_lock_cand, self.cfg.lock_cand_thickness)

                x, y, w, h = b2
                area = int(w * h)
                cv2.putText(
                    vis,
                    f"LOCK CAND size={w}x{h} area={area}",
                    (x, max(0, y - 6)),
                    self.cfg.font,
                    float(self.cfg.lock_cand_text_scale),
                    self.cfg.color_lock_cand,
                    int(self.cfg.lock_cand_text_thickness),
                )

        # locked (green) + "w x h"만 표시 (요청사항)
        lb = dbg.get("locked_bbox", None)
        if lb is not None:
            b2 = _clamp_bbox(tuple(lb), W, H)
            _draw_bbox(vis, b2, self.cfg.color_locked, self.cfg.locked_thickness)

            x, y, w, h = b2
            cv2.putText(
                vis,
                f"{w}x{h}",
                (x, max(0, y - 6)),
                self.cfg.font,
                float(self.cfg.locked_text_scale),
                self.cfg.color_locked,
                int(self.cfg.locked_text_thickness),
            )

        cv2.imshow(self.cfg.window_name, vis)

        if not self.cfg.enable_keys:
            return -1

        key = cv2.waitKey(int(self.cfg.wait_ms)) & 0xFF
        if key != 255:
            self.handle_key(key)
            return key
        return -1
