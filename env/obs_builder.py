from __future__ import annotations

import time
from typing import Optional, Tuple

import cv2
import numpy as np

from env.reimu_tracker_cv import ReimuTrackerCV
from env.reimu_tracker_debug_view import ReimuTrackerDebugView, DebugViewConfig


class ObsBuilder:
    """
    4ch obs (float32):
      ch0: current gray (0..1) + player marker + meta(x,y,conf)
      ch1: prev gray (0..1)                # 원본 그대로(이전 프레임)
      ch2: absdiff(current, prev) (0..1)   # 원본 그대로(차이). threshold/morph/EMA/잔상 없음
      ch3: player position hint (0..1)     # gaussian coord-map (플레이어 위치를 "채널"로 명확히)

    설계 의도:
      - ch0: 공간정보(현재 화면)
      - ch1/ch2: 시간정보(이전 프레임/움직임)
      - ch3: "내 위치"를 항상 명확히(레이무를 장애물로 오해하는 혼란 감소)
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
        # - sigma가 작을수록 "점"에 가까움 (정확 좌표 강조)
        # - sigma가 크면 완만한 언덕(학습은 편한데, 미세회피엔 과할 수 있음)
        self.player_hint_enable = True
        self.player_hint_sigma = 2.0     # s좌표 기준. 1.5~3.5 추천
        self.player_hint_peak = 1.0      # 최대값(0..1)
        self._grid_xy: Optional[Tuple[np.ndarray, np.ndarray]] = None

        # prev gray (s,s) uint8
        self._prev_gray: Optional[np.ndarray] = None

        # debug windows
        self.show_reimu_debug = True
        self.reimu_dbg_view: Optional[ReimuTrackerDebugView] = None  # lazy init

        self.show_obs_debug = True
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

        # ✅ 디버그 윈도우 상태 리셋(새 게임에서 갱신 꼬임 방지)
        self._obs_win_inited = False
        try:
            cv2.destroyWindow(self.win_crop)
        except Exception:
            pass

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
        # gaussian: exp(-d^2 / (2*sigma^2))
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

    # -------------------------
    # main
    # -------------------------
    def make_state(self, img_bgr: np.ndarray) -> np.ndarray:
        # 1) tracker
        now = time.time()
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
            bbox, conf = self.tracker.step(img_bgr)

        if bbox is not None:
            x, y, w, h = map(int, bbox)
            cx, cy = int(round(x + 0.5 * w)), int(round(y + 0.5 * h))
            self.player_center = (cx, cy)
            self.last_conf = float(np.clip(conf, 0.0, 1.0))
            self.last_xy_norm = self._xy_norm(cx, cy)
            self._update_uv()
        else:
            self.last_conf = float(np.clip(conf, 0.0, 1.0))

        # 2) crop -> small (형태 변화의 유일한 원인: 리사이즈)
        pf = self._crop_pf(img_bgr)
        interp = cv2.INTER_AREA if max(pf.shape[:2]) >= self.s else cv2.INTER_LINEAR
        bgr = cv2.resize(pf, (self.s, self.s), interpolation=interp)

        # 3) gray (원본 그대로)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)  # uint8

        # 4) prev/diff (원본 그대로)
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

        # 7) debug
        if self.show_obs_debug:
            try:
                self._ensure_obs_window()
                ch = int(np.clip(self.obs_debug_channel, 0, 3))
                vis = (np.clip(self._obs[ch], 0.0, 1.0) * 255.0).astype(np.uint8)
                vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

                up = int(max(1, self.debug_upscale))
                if up != 1:
                    vis = cv2.resize(vis, (vis.shape[1] * up, vis.shape[0] * up), interpolation=cv2.INTER_NEAREST)

                self._draw_debug_cross(vis, up)

                u, v = self._uv
                self._put_text(
                    vis,
                    f"ch{ch} conf={self.last_conf:.2f} uv=({u},{v}) sigma={self.player_hint_sigma:.2f}",
                    x=10,
                    y=28,
                )
                cv2.imshow(self.win_crop, vis)
                cv2.waitKey(1)  # ✅ 창 이벤트/리페인트 강제 처리

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
