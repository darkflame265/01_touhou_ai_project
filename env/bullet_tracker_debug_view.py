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
    use_antialias: bool = False
    grid_alpha: float = 0.70
    grid_draw_lines: bool = True
    grid_line_color: Tuple[int, int, int] = (70, 70, 70)
    grid_line_thickness: int = 1
    show_player_hitbox: bool = True
    player_hitbox_radius: int = 4
    color_player_hitbox: Tuple[int, int, int] = (0, 255, 0)  # green


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

    def _overlay_occ_grid(self, vis: np.ndarray, occ_grid: np.ndarray) -> np.ndarray:
        g = np.asarray(occ_grid, dtype=np.float32)
        if g.ndim != 2:
            return vis
        gh, gw = g.shape[:2]
        if gh <= 0 or gw <= 0:
            return vis

        h, w = vis.shape[:2]
        out = vis.copy()
        alpha = float(np.clip(self.cfg.grid_alpha, 0.0, 1.0))
        if alpha <= 0.0:
            return vis

        cell_w = float(w) / float(gw)
        cell_h = float(h) / float(gh)

        for iy in range(gh):
            for ix in range(gw):
                v = float(np.clip(g[iy, ix], 0.0, 1.0))
                if v <= 1e-4:
                    continue
                x1 = int(round(ix * cell_w))
                y1 = int(round(iy * cell_h))
                x2 = int(round((ix + 1) * cell_w))
                y2 = int(round((iy + 1) * cell_h))
                x1 = int(np.clip(x1, 0, w - 1))
                y1 = int(np.clip(y1, 0, h - 1))
                x2 = int(np.clip(x2, x1 + 1, w))
                y2 = int(np.clip(y2, y1 + 1, h))

                # BGR heat color: low=blue, high=red
                color = (int(255 * (1.0 - v)), 0, int(255 * v))
                patch = out[y1:y2, x1:x2]
                if patch.size == 0:
                    continue
                blended = cv2.addWeighted(
                    patch,
                    1.0 - alpha,
                    np.full_like(patch, color, dtype=np.uint8),
                    alpha,
                    0.0,
                )
                out[y1:y2, x1:x2] = blended

        if bool(self.cfg.grid_draw_lines):
            c = tuple(map(int, self.cfg.grid_line_color))
            th = int(max(1, self.cfg.grid_line_thickness))
            for ix in range(1, gw):
                x = int(round(ix * cell_w))
                cv2.line(out, (x, 0), (x, h - 1), c, th, cv2.LINE_8)
            for iy in range(1, gh):
                y = int(round(iy * cell_h))
                cv2.line(out, (0, y), (w - 1, y), c, th, cv2.LINE_8)

        return out

    def render(self, roi_bgr: np.ndarray) -> int:
        self._ensure_window()
        roi_bgr = ensure_uint8_bgr(roi_bgr)
        if roi_bgr is None or roi_bgr.size == 0:
            return -1

        vis = roi_bgr.copy()
        dbg: Dict[str, Any] = self.tracker.get_debug() or {}

        occ = dbg.get("grid_occ", None)
        if occ is not None:
            vis = self._overlay_occ_grid(vis, np.asarray(occ))

        if bool(self.cfg.show_player_hitbox):
            pc = dbg.get("player_center_roi", None)
            if pc is not None:
                px, py = map(int, pc)
                cv2.circle(
                    vis,
                    (px, py),
                    int(max(1, self.cfg.player_hitbox_radius)),
                    tuple(map(int, self.cfg.color_player_hitbox)),
                    2,
                    cv2.LINE_AA if bool(self.cfg.use_antialias) else cv2.LINE_8,
                )

        cv2.imshow(self.cfg.window_name, vis)
        key = cv2.waitKey(int(self.cfg.wait_ms)) & 0xFF
        return int(key) if key != 255 else -1
