from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional, List

import cv2
import numpy as np

from env.bullet_tracker_cv import BulletTrackerCV, ensure_uint8_bgr


@dataclass
class BulletDebugViewConfig:
    window_name: str = "debug_bullets"
    wait_ms: int = 1
    enable_keys: bool = False

    # BGR colors
    color_points: Tuple[int, int, int] = (0, 255, 255)  # yellow
    color_topk: Tuple[int, int, int] = (0, 0, 255)      # red
    color_player: Tuple[int, int, int] = (0, 255, 0)    # green

    r_points: int = 2
    r_topk: int = 3


class BulletTrackerDebugView:
    def __init__(self, tracker: BulletTrackerCV, cfg: Optional[BulletDebugViewConfig] = None):
        self.tracker = tracker
        self.cfg = cfg or BulletDebugViewConfig()
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

    def render(self, roi_bgr: np.ndarray) -> int:
        self._ensure_window()
        roi_bgr = ensure_uint8_bgr(roi_bgr)
        if roi_bgr is None or roi_bgr.size == 0:
            return -1

        vis = roi_bgr.copy()
        dbg: Dict[str, Any] = self.tracker.get_debug() or {}

        pts: List[Tuple[float, float]] = dbg.get("points", []) or []
        topk: List[Tuple[float, float]] = dbg.get("points_topk", []) or []
        pc = dbg.get("player_center_roi", None)

        for (x, y) in pts:
            cv2.circle(vis, (int(x), int(y)), int(self.cfg.r_points), self.cfg.color_points, -1)

        for (x, y) in topk:
            cv2.circle(vis, (int(x), int(y)), int(self.cfg.r_topk), self.cfg.color_topk, 2)

        if pc is not None:
            px, py = pc
            cv2.circle(vis, (int(px), int(py)), 4, self.cfg.color_player, 2)

        cv2.putText(
            vis,
            f"bullets n={dbg.get('n',0)} topk={dbg.get('topk',0)}",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.imshow(self.cfg.window_name, vis)

        # ✅ 리페인트/이벤트 펌프는 항상
        key = cv2.waitKey(int(self.cfg.wait_ms)) & 0xFF
        return int(key) if key != 255 else -1
