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
    window_size: Optional[Tuple[int, int]] = None

    # BGR colors
    color_points: Tuple[int, int, int] = (0, 255, 255)  # yellow
    color_topk: Tuple[int, int, int] = (0, 0, 255)      # red
    color_player: Tuple[int, int, int] = (0, 255, 0)    # green
    color_player_bbox: Tuple[int, int, int] = (255, 200, 0)  # cyan-ish
    color_player_ring: Tuple[int, int, int] = (120, 255, 120)
    color_reimu_boxes: Tuple[int, int, int] = (255, 120, 255)

    r_points: int = 2
    r_topk: int = 3
    show_player_bbox: bool = True
    player_bbox_thickness: int = 2
    show_reimu_boxes: bool = True
    reimu_box_thickness: int = 1
    max_draw_points: int = 72
    use_antialias: bool = False


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
            if self.cfg.window_size is not None:
                ww, hh = self.cfg.window_size
                cv2.resizeWindow(self.cfg.window_name, int(max(64, ww)), int(max(64, hh)))
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
        linetype = cv2.LINE_AA if bool(self.cfg.use_antialias) else cv2.LINE_8

        pts: List[Tuple[float, float]] = dbg.get("points", []) or []
        topk: List[Tuple[float, float]] = dbg.get("points_topk", []) or []
        pc = dbg.get("player_center_roi", None)
        pb = dbg.get("player_bbox_roi", None)
        ps: Dict[str, Any] = dbg.get("player_suppress", {}) or {}
        reimu_boxes: List[Tuple[int, int, int, int]] = dbg.get("reimu_boxes", []) or []

        max_pts = int(max(0, self.cfg.max_draw_points))
        if max_pts > 0 and len(pts) > max_pts:
            pts = pts[:max_pts]
        for (x, y) in pts:
            cv2.circle(vis, (int(x), int(y)), int(self.cfg.r_points), self.cfg.color_points, -1, linetype)

        for (x, y) in topk:
            cv2.circle(vis, (int(x), int(y)), int(self.cfg.r_topk), self.cfg.color_topk, 2, linetype)

        if pc is not None:
            px, py = pc
            cv2.circle(vis, (int(px), int(py)), 4, self.cfg.color_player, 2, linetype)
        if self.cfg.show_player_bbox:
            mode = str(ps.get("mode", ""))
            ec = ps.get("ellipse_center", None)
            ea = ps.get("ellipse_axes", None)
            eb = ps.get("expanded_bbox", None)
            if ec is not None and ea is not None and ("ellipse" in mode):
                cx, cy = map(int, ec)
                rx, ry = map(int, ea)
                cv2.ellipse(
                    vis,
                    (cx, cy),
                    (max(1, rx), max(1, ry)),
                    0.0,
                    0.0,
                    360.0,
                    self.cfg.color_player_bbox,
                    int(max(1, self.cfg.player_bbox_thickness)),
                    linetype,
                )
            elif eb is not None:
                bx, by, bw, bh = map(int, eb)
                cv2.rectangle(
                    vis,
                    (bx, by),
                    (bx + max(1, bw) - 1, by + max(1, bh) - 1),
                    self.cfg.color_player_bbox,
                    int(max(1, self.cfg.player_bbox_thickness)),
                    linetype,
                )
            elif pb is not None:
                bx, by, bw, bh = map(int, pb)
                cv2.rectangle(
                    vis,
                    (bx, by),
                    (bx + max(1, bw) - 1, by + max(1, bh) - 1),
                    self.cfg.color_player_bbox,
                    int(max(1, self.cfg.player_bbox_thickness)),
                    linetype,
                )

            # Optional visual for keep-ring if enabled
            if pc is not None:
                rin = int(ps.get("ring_in", 0))
                rout = int(ps.get("ring_out", 0))
                if rout > 0:
                    px, py = map(int, pc)
                    cv2.circle(vis, (px, py), rout, self.cfg.color_player_ring, 1, linetype)
                    if rin > 0:
                        cv2.circle(vis, (px, py), rin, self.cfg.color_player_ring, 1, linetype)

        if self.cfg.show_reimu_boxes:
            th = int(max(1, self.cfg.reimu_box_thickness))
            for b in reimu_boxes:
                bx, by, bw, bh = map(int, b)
                cv2.rectangle(
                    vis,
                    (bx, by),
                    (bx + max(1, bw) - 1, by + max(1, bh) - 1),
                    self.cfg.color_reimu_boxes,
                    th,
                    linetype,
                )

        cv2.putText(
            vis,
            f"bullets n={dbg.get('n',0)} topk={dbg.get('topk',0)}",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            linetype,
        )

        cv2.imshow(self.cfg.window_name, vis)

        # ✅ 리페인트/이벤트 펌프는 항상
        key = cv2.waitKey(int(self.cfg.wait_ms)) & 0xFF
        return int(key) if key != 255 else -1
