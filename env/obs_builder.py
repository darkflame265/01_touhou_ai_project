# env/obs_builder.py
from __future__ import annotations

import time
from typing import Optional, Tuple, List

import cv2
import numpy as np

from env.reimu_tracker_cv import ReimuTrackerCV
from env.reimu_tracker_debug_view import ReimuTrackerDebugView, DebugViewConfig

# ✅ 추가: bullet tracker imports
from env.bullet_tracker_cv import BulletTrackerCV
from env.bullet_tracker_debug_view import BulletTrackerDebugView, BulletDebugViewConfig
from env.actions import ACTIONS


class ObsBuilder:
    """
    4ch obs (float32):
      ch0: current gray (0..1) + player marker + meta(x,y,conf)
      ch1: prev gray (0..1)
      ch2: absdiff(current, prev) (0..1)
      ch3: player position hint (0..1) gaussian coord-map
    """

    def __init__(
        self,
        screen,
        obs_out_size: int = 128,
        crop_size: int = 256,
        use_fallback_full_preprocess: bool = True,
    ):
        self.screen = screen
        self.s = int(obs_out_size)
        self.obs_channels = 4

        img0 = self.screen.capture()
        self.H, self.W = map(int, img0.shape[:2])

        # playfield crop (UI 제거)
        ratio = float(getattr(self.screen, "PLAYFIELD_RIGHT_RATIO", 0.70))
        self._x0, self._x1 = 0, max(1, min(self.W, int(self.W * ratio)))
        top = float(getattr(self.screen, "PLAYFIELD_TOP_CROP", 0.0))
        bot = float(getattr(self.screen, "PLAYFIELD_BOTTOM_CROP", 1.0))
        self._y0 = int(np.clip(round(self.H * top), 0, self.H - 1))
        self._y1 = int(np.clip(round(self.H * bot), self._y0 + 1, self.H))
        self._pw = max(1, self._x1 - self._x0)
        self._ph = max(1, self._y1 - self._y0)

        # tracker (full-frame)
        self.tracker = ReimuTrackerCV()
        self.last_xy_norm: Tuple[float, float] = (0.5, 0.78)
        self.last_conf: float = 0.0
        self.player_center: Tuple[int, int] = (self.W // 2, int(self.H * 0.78))
        self._uv = (self.s // 2, self.s // 2)

        # meta/marker (ch0 only)
        self.meta_patch = 4
        self.mark_player_on_ch0 = True
        self.marker_half = 2
        self.marker_value = 1.0
        self.marker_use_conf = True
        self.marker_min_scale = 0.35

        # ch3: player position hint (gaussian)
        self.player_hint_enable = True
        self.player_hint_sigma = 2.0
        self.player_hint_peak = 1.0
        self._grid_xy: Optional[Tuple[np.ndarray, np.ndarray]] = None

        # prev gray (s,s) uint8
        self._prev_gray: Optional[np.ndarray] = None

        # debug windows
        self.show_reimu_debug = False
        self.reimu_dbg_view: Optional[ReimuTrackerDebugView] = None

        self.show_obs_debug = False
        self.win_crop = "OBS_CROP"
        self._obs_win_inited = False
        self.obs_debug_channel = 2
        self.debug_upscale = 4
        self.debug_font_scale = 0.70
        self.debug_thickness = 2

        # debug cross
        self.debug_draw_cross = True
        self.debug_cross_half = 1
        self.debug_cross_thickness = 2
        self.debug_cross_outer_thickness = 3
        self.debug_cross_color = (255, 255, 255)
        self.debug_cross_outer_color = (0, 0, 0)

        # tracker pause (bomb 등)
        self._pause_until = 0.0
        self._pause_active = False
        self._pause_reset_pending = False

        # buffers
        self._z_u8 = np.zeros((self.s, self.s), np.uint8)
        self._z_f32 = np.zeros((self.s, self.s), np.float32)
        self._obs = np.empty((4, self.s, self.s), np.float32)

        # ✅ bullet tracker (playfield crop)
        self.bullet_tracker = BulletTrackerCV()
        self.last_bullets_xy_norm: List[Tuple[float, float]] = []  # playfield normalized (0..1)
        self.show_bullet_debug = True
        self.bullet_dbg_view: Optional[BulletTrackerDebugView] = None

        # ✅ MLP용: bullet TOP-K relative + velocity (slot-based)
        self.bullet_topk_k = int(getattr(self.bullet_tracker.cfg, "topk", 16))
        self.last_bullets_dxdy: List[Tuple[float, float]] = []     # len<=K
        self.last_bullets_vdxdy: List[Tuple[float, float]] = []    # len<=K
        self._prev_bullets_dxdy_pad: List[Tuple[float, float]] = [(0.0, 0.0)] * self.bullet_topk_k
        # Spatial bullet features for MLP (ID-free): meta4 + occ(64x64) + delta(64x64) = 8196 dims
        self.bullet_grid_size = 64
        self.last_bullet_spatial_vec = np.zeros((4 + 2 * self.bullet_grid_size * self.bullet_grid_size,), dtype=np.float32)
        self._prev_bullet_occ_grid = np.zeros((self.bullet_grid_size, self.bullet_grid_size), dtype=np.float32)


    # -------------------------
    # public hooks
    # -------------------------
    def reset(self):
        try:
            self.tracker.reset()
        except Exception:
            pass

        self.player_center = (self.W // 2, int(self.H * 0.78))
        self.last_xy_norm = (0.5, 0.78)
        self.last_conf = 0.0
        self._uv = (self.s // 2, self.s // 2)

        self._prev_gray = None

        self._pause_until = 0.0
        self._pause_active = False
        self._pause_reset_pending = False

        # debug window
        self._obs_win_inited = False
        try:
            cv2.destroyWindow(self.win_crop)
        except Exception:
            pass

        # ✅ bullet reset
        try:
            self.bullet_tracker.reset()
        except Exception:
            pass
        self.last_bullets_xy_norm = []

        if self.bullet_dbg_view is not None:
            try:
                self.bullet_dbg_view.close()
            except Exception:
                pass
            self.bullet_dbg_view = None

        # ✅ reset bullet rel/vel buffers
        self.bullet_topk_k = int(getattr(self.bullet_tracker.cfg, "topk", 16))
        self.last_bullets_dxdy = []
        self.last_bullets_vdxdy = []
        self._prev_bullets_dxdy_pad = [(0.0, 0.0)] * self.bullet_topk_k
        self.last_bullet_spatial_vec = np.zeros((4 + 2 * self.bullet_grid_size * self.bullet_grid_size,), dtype=np.float32)
        self._prev_bullet_occ_grid = np.zeros((self.bullet_grid_size, self.bullet_grid_size), dtype=np.float32)

    def _build_bullet_spatial_vec(
        self,
        bullets_xy_norm: List[Tuple[float, float]],
        px_n: float,
        py_n: float,
        conf: float,
        bullet_mask_u8: Optional[np.ndarray] = None,
        player_center_roi: Optional[Tuple[int, int]] = None,
        player_bbox_roi: Optional[Tuple[int, int, int, int]] = None,
    ) -> np.ndarray:
        g = int(max(2, self.bullet_grid_size))
        if bullet_mask_u8 is not None and getattr(bullet_mask_u8, "size", 0) > 0:
            m = bullet_mask_u8.astype(np.uint8)
            # Keep MLP spatial input consistent with bullet debug grid by removing player core.
            try:
                if bool(getattr(self.bullet_tracker.cfg, "debug_grid_suppress_player", False)) and player_center_roi is not None:
                    h, w = m.shape[:2]
                    pcore = np.zeros((h, w), dtype=np.uint8)
                    if player_bbox_roi is not None:
                        bx, by, bw, bh = map(int, player_bbox_roi)
                        cx = int(bx + 0.5 * bw)
                        cy = int(by + 0.5 * bh)
                        rx = int(max(2, round(float(bw) * float(getattr(self.bullet_tracker.cfg, "debug_grid_player_rx_scale", 0.18)))))
                        ry = int(max(2, round(float(bh) * float(getattr(self.bullet_tracker.cfg, "debug_grid_player_ry_scale", 0.24)))))
                        cv2.ellipse(pcore, (cx, cy), (rx, ry), 0.0, 0.0, 360.0, 255, thickness=-1)
                    else:
                        px, py = map(int, player_center_roi)
                        rx = int(max(1, int(getattr(self.bullet_tracker.cfg, "debug_grid_player_fallback_rx", 6))))
                        ry = int(max(1, int(getattr(self.bullet_tracker.cfg, "debug_grid_player_fallback_ry", 8))))
                        cv2.ellipse(pcore, (px, py), (rx, ry), 0.0, 0.0, 360.0, 255, thickness=-1)
                    m = cv2.bitwise_and(m, cv2.bitwise_not(pcore))
            except Exception:
                pass
            occ_raw = cv2.resize(
                m,
                (g, g),
                interpolation=cv2.INTER_AREA,
            ).astype(np.float32) / 255.0
        else:
            occ_raw = np.zeros((g, g), dtype=np.float32)
            for (bx, by) in bullets_xy_norm:
                ix = int(np.clip(int(float(bx) * g), 0, g - 1))
                iy = int(np.clip(int(float(by) * g), 0, g - 1))
                occ_raw[iy, ix] += 1.0

        # Apply the same player-centered donut mask used in debug grid.
        try:
            cfg = self.bullet_tracker.cfg
            if bool(getattr(cfg, "debug_grid_player_donut_enable", False)):
                cx = float(np.clip(float(px_n) * (g - 1), 0.0, float(g - 1)))
                cy = float(np.clip(float(py_n) * (g - 1), 0.0, float(g - 1)))
                yy, xx = np.indices((g, g), dtype=np.float32)
                rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
                rin = float(max(0.0, getattr(cfg, "debug_grid_player_donut_r_inner", 1.2)))
                rout = float(max(rin + 1e-6, getattr(cfg, "debug_grid_player_donut_r_outer", 3.2)))
                donut = (rr >= rin) & (rr <= rout)
                mode = str(getattr(cfg, "debug_grid_player_donut_mode", "clear")).strip().lower()
                if mode == "gaussian":
                    sigma = float(max(1e-6, getattr(cfg, "debug_grid_player_donut_sigma", 0.9)))
                    strength = float(np.clip(getattr(cfg, "debug_grid_player_donut_strength", 1.0), 0.0, 1.0))
                    att = np.exp(-0.5 * ((rr - rin) / sigma) ** 2).astype(np.float32)
                    factor = np.clip(1.0 - strength * att, 0.0, 1.0)
                    occ_raw[donut] *= factor[donut]
                else:
                    occ_raw[donut] = 0.0
        except Exception:
            pass

        # Soft normalize to [0,1] while preserving local density.
        occ = np.tanh(occ_raw / 0.8).astype(np.float32)
        delta = np.clip(occ - self._prev_bullet_occ_grid, -1.0, 1.0).astype(np.float32)
        self._prev_bullet_occ_grid = occ.copy()

        n = int(len(bullets_xy_norm))
        n_norm = float(np.clip(n / float(max(1, self.bullet_topk_k)), 0.0, 1.0))
        meta = np.asarray([float(px_n), float(py_n), float(np.clip(conf, 0.0, 1.0)), float(n_norm)], dtype=np.float32)
        return np.concatenate([meta, occ.reshape(-1), delta.reshape(-1)], axis=0).astype(np.float32)

    def on_player_death(self):
        try:
            self.tracker.reset()
        except Exception:
            pass

    def on_bomb_used(self, pause_sec: float = 2.0):
        now = time.time()
        self._pause_until = float(now + float(pause_sec))
        self._pause_active = True
        self._pause_reset_pending = True

    def pump_key(self, key: int):
        if key is None or key < 0:
            return
        if self.show_reimu_debug and self.reimu_dbg_view is not None:
            self.reimu_dbg_view.handle_key(int(key))

    # -------------------------
    # helpers
    # -------------------------
    def _crop_pf(self, img):
        h, w = img.shape[:2]
        x0 = int(np.clip(self._x0, 0, w - 1))
        x1 = int(np.clip(self._x1, x0 + 1, w))
        y0 = int(np.clip(self._y0, 0, h - 1))
        y1 = int(np.clip(self._y1, y0 + 1, h))
        return img[y0:y1, x0:x1]

    def _xy_norm(self, cx, cy):
        x_pf = float(cx - self._x0)
        y_pf = float(cy - self._y0)
        x_n = float(np.clip(x_pf / max(1, self._pw - 1), 0.0, 1.0))
        y_n = float(np.clip(y_pf / max(1, self._ph - 1), 0.0, 1.0))
        return x_n, y_n

    def _update_uv(self):
        px, py = map(int, self.player_center)
        x_pf = float(px - self._x0)
        y_pf = float(py - self._y0)
        u = int(round(x_pf * (self.s - 1) / max(1, (self._pw - 1))))
        v = int(round(y_pf * (self.s - 1) / max(1, (self._ph - 1))))
        self._uv = (int(np.clip(u, 0, self.s - 1)), int(np.clip(v, 0, self.s - 1)))

    def _inject_meta(self, ch0):
        p = int(self.meta_patch)
        if p <= 0 or ch0.shape[0] < p or ch0.shape[1] < p * 3:
            return
        x_n, y_n = self.last_xy_norm
        c = float(np.clip(self.last_conf, 0.0, 1.0))
        ch0[0:p, 0:p] = float(np.clip(x_n, 0.0, 1.0))
        ch0[0:p, p:2 * p] = float(np.clip(y_n, 0.0, 1.0))
        ch0[0:p, 2 * p:3 * p] = c

    def _stamp_marker(self, ch0):
        if not self.mark_player_on_ch0:
            return
        u, v = self._uv
        r = int(self.marker_half)
        if r <= 0:
            return
        val = float(self.marker_value)
        if self.marker_use_conf:
            val *= max(float(self.marker_min_scale), float(np.clip(self.last_conf, 0.0, 1.0)))
        x1, x2 = max(0, u - r), min(self.s, u + r + 1)
        y1, y2 = max(0, v - r), min(self.s, v + r + 1)
        ch0[v, x1:x2] = val
        ch0[y1:y2, u] = val

    def _get_grid(self):
        if self._grid_xy is not None:
            return self._grid_xy
        xs = np.arange(self.s, dtype=np.float32)
        ys = np.arange(self.s, dtype=np.float32)
        xx, yy = np.meshgrid(xs, ys)
        self._grid_xy = (xx, yy)
        return self._grid_xy

    def _player_hint_map(self) -> np.ndarray:
        if not self.player_hint_enable:
            return self._z_f32

        sigma = float(max(1e-6, self.player_hint_sigma))
        peak = float(np.clip(self.player_hint_peak, 0.0, 1.0))

        xx, yy = self._get_grid()
        u, v = self._uv
        du = xx - float(u)
        dv = yy - float(v)
        d2 = du * du + dv * dv
        hint = np.exp(-d2 / (2.0 * sigma * sigma)).astype(np.float32)
        if peak != 1.0:
            hint *= peak
        return hint

    # -------------------------
    # debug
    # -------------------------
    def _ensure_obs_window(self):
        if self._obs_win_inited:
            return
        try:
            cv2.namedWindow(self.win_crop, cv2.WINDOW_NORMAL)
        except Exception:
            pass
        self._obs_win_inited = True

    def _put_text(self, img, text: str, x=10, y=28):
        fs = float(self.debug_font_scale)
        th = int(self.debug_thickness)
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), th + 2, cv2.LINE_AA)
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), th, cv2.LINE_AA)

    def _draw_debug_cross(self, vis_bgr: np.ndarray, up: int):
        if not self.debug_draw_cross:
            return

        u, v = self._uv
        up = int(max(1, up))
        x = int(u * up)
        y = int(v * up)

        half = int(max(1, self.debug_cross_half)) * up
        th_in = int(max(1, self.debug_cross_thickness))
        th_out = int(max(th_in + 1, self.debug_cross_outer_thickness))
        c_in = tuple(map(int, self.debug_cross_color))
        c_out = tuple(map(int, self.debug_cross_outer_color))

        h, w = vis_bgr.shape[:2]
        x1, x2 = max(0, x - half), min(w - 1, x + half)
        y1, y2 = max(0, y - half), min(h - 1, y + half)

        cv2.line(vis_bgr, (x1, y), (x2, y), c_out, th_out, cv2.LINE_AA)
        cv2.line(vis_bgr, (x, y1), (x, y2), c_out, th_out, cv2.LINE_AA)
        cv2.line(vis_bgr, (x1, y), (x2, y), c_in, th_in, cv2.LINE_AA)
        cv2.line(vis_bgr, (x, y1), (x, y2), c_in, th_in, cv2.LINE_AA)
        cv2.circle(vis_bgr, (x, y), max(1, up), c_out, thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(vis_bgr, (x, y), max(1, up // 2), c_in, thickness=-1, lineType=cv2.LINE_AA)

    def _action_idx_to_move_vec(self, action_idx: Optional[int]) -> Optional[Tuple[float, float]]:
        if action_idx is None:
            return None
        try:
            i = int(action_idx)
        except Exception:
            return None
        if not (0 <= i < len(ACTIONS)):
            return None
        try:
            keys = set(getattr(ACTIONS[i], "value", []) or [])
        except Exception:
            keys = set()
        if "BOMB" in keys:
            return None
        dx = 0.0
        dy = 0.0
        if "LEFT" in keys:
            dx -= 1.0
        if "RIGHT" in keys:
            dx += 1.0
        if "UP" in keys:
            dy -= 1.0
        if "DOWN" in keys:
            dy += 1.0
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return None
        return (dx, dy)

    # -------------------------
    # main
    # -------------------------
    def make_state(self, img_bgr: np.ndarray, action_idx: Optional[int] = None) -> np.ndarray:
        # 1) reimu tracker
        now = time.time()
        expected_move_vec = self._action_idx_to_move_vec(action_idx)
        if self._pause_active and now < float(self._pause_until):
            bbox, conf = None, 0.0
        else:
            if self._pause_active and self._pause_reset_pending:
                try:
                    self.tracker.reset()
                except Exception:
                    pass
                self._pause_reset_pending = False
                self._pause_active = False
            bbox, conf = self.tracker.step(img_bgr, expected_move_vec=expected_move_vec)

        if bbox is not None:
            x, y, w, h = map(int, bbox)
            cx, cy = int(round(x + 0.5 * w)), int(round(y + 0.5 * h))
            self.player_center = (cx, cy)
            self.last_conf = float(np.clip(conf, 0.0, 1.0))
            self.last_xy_norm = self._xy_norm(cx, cy)
            self._update_uv()
        else:
            self.last_conf = float(np.clip(conf, 0.0, 1.0))

        # 2) crop -> small
        pf = self._crop_pf(img_bgr)
        interp = cv2.INTER_AREA if max(pf.shape[:2]) >= self.s else cv2.INTER_LINEAR
        bgr = cv2.resize(pf, (self.s, self.s), interpolation=interp)

        # ✅ bullet tracking on playfield (ROI coords)
        px, py = self.player_center
        px_pf = int(np.clip(px - self._x0, 0, self._pw - 1))
        py_pf = int(np.clip(py - self._y0, 0, self._ph - 1))
        player_center_pf = None

        bbox_pf = None
        if bbox is not None:
            bx, by, bw, bh = map(int, bbox)
            bx_pf = int(np.clip(bx - self._x0, 0, self._pw - 1))
            by_pf = int(np.clip(by - self._y0, 0, self._ph - 1))
            bw_pf = int(max(1, min(self._pw - bx_pf, bw)))
            bh_pf = int(max(1, min(self._ph - by_pf, bh)))
            if bw_pf > 0 and bh_pf > 0:
                bbox_pf = (bx_pf, by_pf, bw_pf, bh_pf)
                player_center_pf = (px_pf, py_pf)

        pts_pf = self.bullet_tracker.step(
            pf,
            player_center_roi=player_center_pf,
            player_bbox_roi=bbox_pf,
        )

        bul_norm: List[Tuple[float, float]] = []
        den_x = max(1, self._pw - 1)
        den_y = max(1, self._ph - 1)
        for (bx, by) in pts_pf:
            x_n = float(np.clip(bx / den_x, 0.0, 1.0))
            y_n = float(np.clip(by / den_y, 0.0, 1.0))
            bul_norm.append((x_n, y_n))
        self.last_bullets_xy_norm = bul_norm

        # ✅ build relative dxdy + velocity vdxdy (slot-based, padded)
        px_n, py_n = self.last_xy_norm
        K = int(self.bullet_topk_k)

        dxdy: List[Tuple[float, float]] = []
        for (bx, by) in self.last_bullets_xy_norm[:K]:
            dxdy.append((float(bx) - float(px_n), float(by) - float(py_n)))

        # pad to K for stable velocity computation
        cur_pad: List[Tuple[float, float]] = dxdy + [(0.0, 0.0)] * max(0, K - len(dxdy))

        prev_pad = self._prev_bullets_dxdy_pad
        vdxdy_pad: List[Tuple[float, float]] = []
        for i in range(K):
            cdx, cdy = cur_pad[i]
            pdx, pdy = prev_pad[i]
            vdxdy_pad.append((float(cdx) - float(pdx), float(cdy) - float(pdy)))

        # store (un-padded lists are fine, but we keep padded length for downstream simplicity)
        self.last_bullets_dxdy = cur_pad
        self.last_bullets_vdxdy = vdxdy_pad
        self._prev_bullets_dxdy_pad = cur_pad
        self.last_bullet_spatial_vec = self._build_bullet_spatial_vec(
            self.last_bullets_xy_norm,
            float(px_n),
            float(py_n),
            float(self.last_conf),
            bullet_mask_u8=getattr(self.bullet_tracker, "last_mask_u8", None),
            player_center_roi=player_center_pf,
            player_bbox_roi=bbox_pf,
        )


        if self.show_bullet_debug:
            if self.bullet_dbg_view is None:
                self.bullet_dbg_view = BulletTrackerDebugView(
                    self.bullet_tracker,
                    cfg=BulletDebugViewConfig(
                        window_name="debug_bullets",
                        enable_keys=False,
                        wait_ms=1,
                        window_size=(int(self._pw), int(self._ph)),
                    ),
                )
            try:
                self.bullet_dbg_view.render(pf)
            except Exception as e:
                print("[bullet_dbg_view.render ERROR]", repr(e))

        # 3) gray
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)  # uint8

        # 4) prev/diff
        if self._prev_gray is None or self._prev_gray.shape != gray.shape:
            prev = self._z_u8
            diff = self._z_u8
        else:
            prev = self._prev_gray
            diff = cv2.absdiff(gray, prev)

        # 5) ch3 player hint
        hint = self._player_hint_map()

        # 6) assemble obs
        self._obs[0] = gray.astype(np.float32) * (1.0 / 255.0)
        self._stamp_marker(self._obs[0])
        self._inject_meta(self._obs[0])

        self._obs[1] = prev.astype(np.float32) * (1.0 / 255.0)
        self._obs[2] = diff.astype(np.float32) * (1.0 / 255.0)
        self._obs[3] = hint

        # 7) obs debug
        if self.show_obs_debug:
            try:
                self._ensure_obs_window()
                ch = int(np.clip(self.obs_debug_channel, 0, 3))
                vis = (np.clip(self._obs[ch], 0.0, 1.0) * 255.0).astype(np.uint8)
                vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

                up = int(max(1, self.debug_upscale))
                if up != 1:
                    vis = cv2.resize(
                        vis, (vis.shape[1] * up, vis.shape[0] * up),
                        interpolation=cv2.INTER_NEAREST
                    )

                self._draw_debug_cross(vis, up)

                u, v = self._uv
                self._put_text(
                    vis,
                    f"ch{ch} conf={self.last_conf:.2f} uv=({u},{v}) sigma={self.player_hint_sigma:.2f}",
                    x=10,
                    y=28,
                )
                cv2.imshow(self.win_crop, vis)
                cv2.waitKey(1)
            except Exception:
                pass

        # reimu debug (lazy init)
        if self.show_reimu_debug:
            if self.reimu_dbg_view is None:
                self.reimu_dbg_view = ReimuTrackerDebugView(
                    self.tracker,
                    cfg=DebugViewConfig(window_name="debug_hell", enable_keys=False, wait_ms=1),
                )
            try:
                self.reimu_dbg_view.render(img_bgr)
            except Exception as e:
                print("[reimu_dbg_view.render ERROR]", repr(e))

        # 8) update prev
        self._prev_gray = gray
        return self._obs.copy()
