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
      ch0: gray(0..1) + player marker + meta(x,y,conf)
      ch1: signed diff (gray_t - gray_{t-1}), global-shift removed (0..1)
      ch2: motion_hazard (0..1)  # ✅ 잔상 없음: 현재 프레임 기반 hazard만
           + (선택) 플레이어 근접 강조(지우지 않고 '조금 더 위험'으로)
      ch3: risk_heat (0..1) from distanceTransform on hazard_bin
           + (선택) hazard fill에 따라 tau 동적 조절

    핵심 변경(요청 반영):
      - persistence(잔상) 완전 제거
      - EMA도 기본 off 유지 (원하면 켤 수는 있음)
      - ch2는 "현재 프레임의 움직임 위험"만 보여서 디버그 시각화도 깔끔
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

        # prev
        self._prev_gray: Optional[np.ndarray] = None

        # gray invert (선택)
        self.auto_invert_gray = True
        self.invert_mean_thr = 0.58
        self._last_inverted = False

        # (옵션) ch0/ch1 안정화: CLAHE
        self.gray_clahe_enable = False
        self.gray_clahe_clip = 2.0
        self.gray_clahe_tile = 8
        self._clahe = None
        self._clahe_key = None  # (clip,tile) 변경 감지

        # hazard on/off
        self.enable_hazard = True

        # diff -> motion mask params (MAD threshold)
        self.diff_min = 10
        self.diff_k_mad = 3.0

        # diff 포화(플래시/톤 변화) 대응
        self.diff_saturation_ratio = 0.30
        self.diff_saturation_boost = 2.0

        # morph (기본 off: 뭉개짐 방지)
        self.motion_open = 0
        self.motion_close = 0
        self._ker_open = None
        self._ker_close = None
        self._ker_open_k = -1
        self._ker_close_k = -1

        # ch1: signed diff 설정
        self.signed_diff_enable = True
        self.signed_diff_remove_global_shift = True  # d - median(d)
        self.signed_diff_clip = 48.0                 # [-clip, +clip] 후 0..1 매핑

        # ✅ 잔상/EMA 제거(기본)
        self.hazard_ema_enable = False
        self.hazard_ema_alpha = 0.65
        self._hazard_ema: Optional[np.ndarray] = None

        # 플레이어 근접 강조 (지우지 않고 "조금 더 위험"으로)
        self.hazard_proximity_boost_enable = True
        self.hazard_proximity_w = 0.35     # 0.0~0.8
        self.hazard_proximity_tau = 10.0   # s좌표 기준 거리 감쇠(6~16)
        self._grid_xy = None               # (xx,yy) 캐시

        # hazard bin threshold (risk 입력용)
        self.hazard_bin_thr_for_risk = 0.35

        # risk map
        self.risk_tau_px = 8.0
        self.risk_clip_max = 1.0
        self.risk_use_max_normalize = False

        # tau 동적 조절(선택): hazard가 많을수록 tau 줄여 과포화 방지
        self.risk_dynamic_tau_enable = True
        self.risk_tau_min = 4.0
        self.risk_tau_max = 10.0
        self.risk_tau_fill_lo = 0.02
        self.risk_tau_fill_hi = 0.20

        # exported maps
        self.hazard_bin = None          # (s,s) uint8 {0,1}
        self.risk_heatmap = None        # (s,s) float32 0..1
        self.hazard_global_fill = 0.0

        # debug windows (최소)
        self.show_reimu_debug = False
        self.reimu_dbg_view = ReimuTrackerDebugView(
            self.tracker,
            cfg=DebugViewConfig(window_name="debug_hell", enable_keys=False, wait_ms=1),
        )

        self.show_obs_debug = True
        self.win_crop = "OBS_CROP"
        self._obs_win_inited = False
        self.obs_debug_channel = 3
        self.debug_upscale = 4
        self.debug_font_scale = 0.70
        self.debug_thickness = 2

        # debug cross (시각화 전용)
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
        self._hazard_ema = None
        self._last_inverted = False

        self.hazard_bin = None
        self.risk_heatmap = None
        self.hazard_global_fill = 0.0

        self._pause_until = 0.0
        self._pause_active = False
        self._pause_reset_pending = False

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

    def _maybe_invert(self, gray_u8):
        if not self.auto_invert_gray:
            self._last_inverted = False
            return gray_u8
        inv = (float(gray_u8.mean(dtype=np.float32) / 255.0) >= float(self.invert_mean_thr))
        self._last_inverted = bool(inv)
        return (255 - gray_u8) if inv else gray_u8

    def _maybe_clahe(self, gray_u8: np.ndarray) -> np.ndarray:
        if not self.gray_clahe_enable:
            return gray_u8
        tile = int(max(2, self.gray_clahe_tile))
        clip = float(max(0.5, self.gray_clahe_clip))
        key = (clip, tile)
        if self._clahe is None or self._clahe_key != key:
            self._clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
            self._clahe_key = key
        return self._clahe.apply(gray_u8)

    @staticmethod
    def _mad_u8(x):
        med = float(np.median(x))
        mad = float(np.median(np.abs(x.astype(np.float32) - med)))
        return med, mad

    def _ker(self, kind: str, k: int):
        k = int(k)
        if k <= 0:
            return None
        if kind == "open":
            if self._ker_open is not None and self._ker_open_k == k:
                return self._ker_open
            self._ker_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
            self._ker_open_k = k
            return self._ker_open
        if kind == "close":
            if self._ker_close is not None and self._ker_close_k == k:
                return self._ker_close
            self._ker_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
            self._ker_close_k = k
            return self._ker_close
        return None

    def _get_grid(self):
        if self._grid_xy is not None:
            return self._grid_xy
        xs = np.arange(self.s, dtype=np.float32)
        ys = np.arange(self.s, dtype=np.float32)
        xx, yy = np.meshgrid(xs, ys)
        self._grid_xy = (xx, yy)
        return self._grid_xy

    # -------------------------
    # motion hazard
    # -------------------------
    def _motion_mask(self, diff_u8: np.ndarray) -> np.ndarray:
        med, mad = self._mad_u8(diff_u8)
        thr = max(float(self.diff_min), med + float(self.diff_k_mad) * mad)

        base = (diff_u8.astype(np.float32) >= float(thr)).astype(np.uint8) * 255
        fill = float(np.mean(base > 0)) if base.size else 0.0

        # 포화(플래시/톤변화)면 threshold 강화
        if fill >= float(self.diff_saturation_ratio):
            thr2 = max(float(self.diff_min), med + float(self.diff_k_mad) * float(self.diff_saturation_boost) * mad)
            base = (diff_u8.astype(np.float32) >= float(thr2)).astype(np.uint8) * 255

        k = int(self.motion_open)
        if k > 0:
            base = cv2.morphologyEx(base, cv2.MORPH_OPEN, self._ker("open", k), iterations=1)
        k2 = int(self.motion_close)
        if k2 > 0:
            base = cv2.morphologyEx(base, cv2.MORPH_CLOSE, self._ker("close", k2), iterations=1)

        return base

    def _signed_diff_map(self, gray_u8: np.ndarray, prev_u8: np.ndarray) -> np.ndarray:
        if prev_u8 is None or prev_u8.shape != gray_u8.shape:
            return self._z_f32

        d = gray_u8.astype(np.int16) - prev_u8.astype(np.int16)

        if self.signed_diff_remove_global_shift:
            d = d - int(np.median(d))

        clip = float(max(1.0, self.signed_diff_clip))
        d = np.clip(d.astype(np.float32), -clip, clip)
        return (d + clip) * (0.5 / clip)

    # -------------------------
    # risk
    # -------------------------
    def _dynamic_tau(self, fill: float) -> float:
        if not self.risk_dynamic_tau_enable:
            return float(self.risk_tau_px)

        lo = float(np.clip(self.risk_tau_fill_lo, 0.0, 1.0))
        hi = float(np.clip(self.risk_tau_fill_hi, lo + 1e-6, 1.0))
        tmin = float(max(1e-6, self.risk_tau_min))
        tmax = float(max(tmin, self.risk_tau_max))

        if fill <= lo:
            return tmax
        if fill >= hi:
            return tmin

        a = (fill - lo) / (hi - lo)
        return (1.0 - a) * tmax + a * tmin

    def _risk(self, hazard_u8_bin: np.ndarray, tau_override: Optional[float] = None) -> np.ndarray:
        if hazard_u8_bin is None or hazard_u8_bin.size == 0 or int(np.count_nonzero(hazard_u8_bin)) == 0:
            return self._z_f32

        inv = cv2.bitwise_not(hazard_u8_bin)
        dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3)

        tau = float(self.risk_tau_px) if tau_override is None else float(tau_override)
        tau = max(1e-6, tau)
        risk = np.exp(-dist / tau).astype(np.float32)

        if self.risk_use_max_normalize:
            mx = float(risk.max())
            if mx > 1e-6:
                risk *= (1.0 / mx)

        if self.risk_clip_max is not None:
            np.clip(risk, 0.0, float(self.risk_clip_max), out=risk)

        return risk

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

        # 2) crop -> small
        pf = self._crop_pf(img_bgr)
        interp = cv2.INTER_AREA if max(pf.shape[:2]) >= self.s else cv2.INTER_LINEAR
        bgr = cv2.resize(pf, (self.s, self.s), interpolation=interp)

        # 3) gray
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = self._maybe_invert(gray)
        gray = self._maybe_clahe(gray)

        # 4) ch1 signed diff
        ch1_map = self._signed_diff_map(gray, self._prev_gray) if self.signed_diff_enable else self._z_f32

        # hazard용 absdiff
        if self._prev_gray is None or self._prev_gray.shape != gray.shape:
            diff_u8 = self._z_u8
        else:
            diff_u8 = cv2.absdiff(gray, self._prev_gray)

        self._prev_gray = gray.copy()

        # 5) hazard (현재 프레임만) + (선택)근접 강조 + risk
        if self.enable_hazard:
            hazard_u8 = self._motion_mask(diff_u8)  # 0/255
            ch2_map = (hazard_u8 > 0).astype(np.float32)  # 0/1
        else:
            ch2_map = self._z_f32

        # (선택) EMA - 잔상과 달리 "평균화"라 더 짧게 남지만, 기본 off
        if self.hazard_ema_enable:
            if self._hazard_ema is None or self._hazard_ema.shape != ch2_map.shape:
                self._hazard_ema = ch2_map.copy()
            else:
                a = float(np.clip(self.hazard_ema_alpha, 0.0, 0.999))
                self._hazard_ema = a * self._hazard_ema + (1.0 - a) * ch2_map
            ch2_map = np.clip(self._hazard_ema, 0.0, 1.0)

        # proximity boost (multiply, then clip)
        if self.hazard_proximity_boost_enable and float(self.hazard_proximity_w) > 0.0:
            xx, yy = self._get_grid()
            u, v = self._uv
            du = xx - float(u)
            dv = yy - float(v)
            dist = np.sqrt(du * du + dv * dv)
            tau = float(max(1e-6, self.hazard_proximity_tau))
            w = float(max(0.0, self.hazard_proximity_w))
            boost = 1.0 + w * np.exp(-dist / tau)
            ch2_map = np.clip(ch2_map * boost, 0.0, 1.0)

        # hazard bin for risk
        hazard_for_risk_u8 = (ch2_map >= float(self.hazard_bin_thr_for_risk)).astype(np.uint8) * 255
        self.hazard_bin = (hazard_for_risk_u8 > 0).astype(np.uint8)
        self.hazard_global_fill = float(np.mean(self.hazard_bin > 0)) if self.hazard_bin.size else 0.0

        tau_eff = self._dynamic_tau(self.hazard_global_fill)
        risk = self._risk(hazard_for_risk_u8, tau_override=tau_eff) if self.enable_hazard else self._z_f32
        self.risk_heatmap = risk.astype(np.float32, copy=False)

        # 6) assemble obs
        self._obs[0] = gray.astype(np.float32) * (1.0 / 255.0)
        self._stamp_marker(self._obs[0])
        self._inject_meta(self._obs[0])

        self._obs[1] = ch1_map.astype(np.float32, copy=False)
        self._obs[2] = ch2_map.astype(np.float32, copy=False)
        self._obs[3] = self.risk_heatmap

        # debug (minimal)
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
                    f"ch{ch} conf={self.last_conf:.2f} uv=({u},{v}) fill={self.hazard_global_fill:.3f} tau={tau_eff:.2f}",
                    x=10,
                    y=28,
                )
                cv2.imshow(self.win_crop, vis)
            except Exception:
                pass

        if self.show_reimu_debug and self.reimu_dbg_view is not None:
            try:
                self.reimu_dbg_view.render(img_bgr)
            except Exception as e:
                print("[reimu_dbg_view.render ERROR]", repr(e))

        return self._obs.copy()
