# env/obs_builder.py
from __future__ import annotations

import time
from typing import Optional, Tuple

import cv2
import numpy as np

from env.reimu_tracker_cv import ReimuTrackerCV
from env.reimu_tracker_debug_view import ReimuTrackerDebugView, DebugViewConfig


class ObsBuilder:
    """
    4채널 관측 (float32):
      ch0: gray (0..1) + player marker + meta pixels(xy/conf)
      ch1: absdiff(current_gray, prev_gray) (0..1)
      ch2: bullet_candidate_mask (0..1)
      ch3: risk_heatmap (distanceTransform 기반, 0..1)

    ✅ UI 제외 버전(권장):
    - playfield만 crop 한 뒤 -> obs_out_size로 리사이즈
    - tracker는 full-frame 기준으로 step(img_bgr)
    - last_xy_norm: playfield 기준 정규화(가로/세로 모두 playfield crop 기준)
    - ch0에 십자 마커 + meta pixels(x/y/conf)
    - ✅ 초미세 회피용 local risk:
        - risk_local_valid (conf 기반)
        - risk_local_p90 / risk_local_p99 / risk_local_mean
        - (포화 방지) ROI에서 '탄 픽셀(=risk~1)'은 제외하고 분위수 계산
    """

    def __init__(
        self,
        screen,
        obs_out_size: int = 128,
        crop_size: int = 256,  # (호환용) 미사용
        use_fallback_full_preprocess: bool = True,  # (호환용) 미사용
    ):
        self.screen = screen

        self.obs_out_size = int(obs_out_size)
        self.crop_size = int(crop_size)
        self.obs_channels = 4

        img0 = self.screen.capture()
        h0, w0 = img0.shape[:2]
        self.H, self.W = int(h0), int(w0)

        # ===== playfield crop 설정 =====
        # 기본: 우측 UI 패널 제외 (x: 0 ~ playfield_w)
        self._playfield_ratio = float(getattr(self.screen, "PLAYFIELD_RIGHT_RATIO", 0.70))
        self._playfield_x0 = 0
        self._playfield_x1 = max(1, min(self.W, int(self.W * self._playfield_ratio)))

        # 세로 crop: Screen에 설정이 있으면 사용, 없으면 전체(0~H)
        top = float(getattr(self.screen, "PLAYFIELD_TOP_CROP", 0.0))
        bot = float(getattr(self.screen, "PLAYFIELD_BOTTOM_CROP", 1.0))
        self._playfield_y0 = int(np.clip(round(self.H * top), 0, self.H - 1))
        self._playfield_y1 = int(np.clip(round(self.H * bot), self._playfield_y0 + 1, self.H))

        self._pf_w = max(1, self._playfield_x1 - self._playfield_x0)
        self._pf_h = max(1, self._playfield_y1 - self._playfield_y0)

        # tracker (full-frame)
        self.tracker = ReimuTrackerCV()

        # 정책/리워드용 좌표/신뢰도 (playfield 기준 정규화)
        self.last_xy_norm: Tuple[float, float] = (0.5, 0.78)
        self.last_conf: float = 0.0

        # det None일 때 유지 (full-frame 좌표)
        self.player_center: Tuple[int, int] = (w0 // 2, int(h0 * 0.78))

        # meta pixels
        self.meta_patch: int = 4

        # prev gray (obs_out_size 기준으로 저장)
        self._prev_gray_small_u8: Optional[np.ndarray] = None

        # ----- auto inversion -----
        self.auto_invert_gray: bool = True
        self.invert_mean_thr: float = 0.58  # 0..1
        self._last_inverted: bool = False

        # ----- bullet/background separation -----
        self.use_motion_for_bullets: bool = True
        self.diff_bullet_min: int = 10
        self.diff_bullet_k_mad: float = 3.0
        # motion이 화면을 덮어버리면 risk 포화 → HSV fallback
        self.max_bullet_fill_ratio: float = 0.30

        # tracker bbox 캐시 (full 기준)
        self._last_bbox_full: Optional[Tuple[int, int, int, int]] = None  # (x,y,w,h)

        # bullet/risk
        self.enable_bullet_channels: bool = True
        self.bullet_hsv_s_min: int = 40
        self.bullet_hsv_v_min: int = 140
        self.bullet_hsv_v_max: int = 255
        self.bullet_close_morph: int = 0  # 0이면 morph 스킵

        self.risk_tau_px: float = 8.0
        self.risk_clip_max: float = 1.0
        self.risk_use_max_normalize: bool = False

        # ===== player marker on ch0 =====
        self.mark_player_on_ch0: bool = True
        self.marker_half: int = 2
        self.marker_value: float = 1.0
        self.marker_use_conf: bool = True
        self.marker_min_scale: float = 0.35
        self._last_player_uv_small: Tuple[int, int] = (self.obs_out_size // 2, self.obs_out_size // 2)

        # ===== local risk ROI =====
        self.local_risk_enable: bool = True
        self.local_risk_radius: int = 12
        self.local_risk_quantile_p: float = 0.90  # 노출용
        self.local_risk_conf_thr: float = 0.20

        self.local_risk_exclude_saturated: bool = True
        self.local_risk_sat_thr: float = 0.999
        self.local_risk_min_valid_bg_frac: float = 0.30

        self.risk_local_valid: bool = False
        self.risk_local_mean: float = 0.0
        self.risk_local_p90: float = 0.0
        self.risk_local_p99: float = 0.0
        self.risk_local_bg_frac: float = 0.0
        self.risk_local_max: float = 0.0

        # 외부 getattr 안전
        self.bullet_candidate_mask = None
        self.risk_heatmap = None

        # 레이무 디버그 창
        self.show_reimu_debug: bool = True
        dbg_cfg = DebugViewConfig(
            window_name="debug_hell",
            enable_keys=False,
            wait_ms=1,
        )
        self.reimu_dbg_view = ReimuTrackerDebugView(self.tracker, cfg=dbg_cfg)

        # OBS 디버그 (playfield만 보여줌)
        self.show_obs_debug: bool = True
        self.win_crop: str = "OBS_CROP"
        self._obs_win_inited: bool = False
        self.obs_debug_channel: int = 0

        # 디버그 표시 업스케일 (텍스트 선명도)
        self.debug_upscale: int = 4
        self.debug_font_scale: float = 0.70
        self.debug_thickness: int = 2

        # tracker pause (bomb etc.)
        self._track_pause_until: float = 0.0
        self._track_pause_active: bool = False
        self._track_pause_resume_reset_pending: bool = False

        # ===== buffers =====
        s = self.obs_out_size
        self._zeros_small_u8 = np.zeros((s, s), dtype=np.uint8)
        self._zeros_small_f32 = np.zeros((s, s), dtype=np.float32)
        self._obs_buf = np.empty((4, s, s), dtype=np.float32)

        self._bullet_kernel = None
        self._bullet_kernel_k = -1

    # -------------------------
    # lifecycle / hooks
    # -------------------------
    def reset(self):
        try:
            self.tracker.reset()
        except Exception:
            pass

        self.player_center = (self.W // 2, int(self.H * 0.78))
        self.last_xy_norm = (0.5, 0.78)
        self.last_conf = 0.0
        self._prev_gray_small_u8 = None
        self._last_inverted = False
        self._last_bbox_full = None
        self._track_pause_until = 0.0
        self._track_pause_active = False
        self._track_pause_resume_reset_pending = False
        self._last_player_uv_small = (self.obs_out_size // 2, self.obs_out_size // 2)

        self.risk_local_valid = False
        self.risk_local_mean = 0.0
        self.risk_local_p90 = 0.0
        self.risk_local_p99 = 0.0
        self.risk_local_bg_frac = 0.0
        self.risk_local_max = 0.0

    def on_player_death(self):
        try:
            self.tracker.reset()
        except Exception:
            pass

    def on_bomb_used(self, pause_sec: float = 2.0):
        now = time.time()
        self._track_pause_until = float(now + float(pause_sec))
        self._track_pause_active = True
        self._track_pause_resume_reset_pending = True

    def pump_key(self, key: int):
        if key is None or key < 0:
            return
        if self.reimu_dbg_view is not None:
            self.reimu_dbg_view.handle_key(int(key))

    # -------------------------
    # debug window
    # -------------------------
    def _ensure_obs_window(self):
        if self._obs_win_inited:
            return
        try:
            cv2.namedWindow(self.win_crop, cv2.WINDOW_NORMAL)
        except Exception:
            pass
        self._obs_win_inited = True

    # -------------------------
    # playfield helpers
    # -------------------------
    def _crop_playfield(self, img_bgr: np.ndarray) -> np.ndarray:
        # 안전하게 clamp
        x0, x1 = int(self._playfield_x0), int(self._playfield_x1)
        y0, y1 = int(self._playfield_y0), int(self._playfield_y1)
        x0 = int(np.clip(x0, 0, img_bgr.shape[1] - 1))
        x1 = int(np.clip(x1, x0 + 1, img_bgr.shape[1]))
        y0 = int(np.clip(y0, 0, img_bgr.shape[0] - 1))
        y1 = int(np.clip(y1, y0 + 1, img_bgr.shape[0]))
        return img_bgr[y0:y1, x0:x1]

    def _full_xy_to_playfield_norm(self, cx_full: int, cy_full: int) -> Tuple[float, float]:
        # full 좌표 -> playfield 좌표
        x_pf = float(cx_full - self._playfield_x0)
        y_pf = float(cy_full - self._playfield_y0)

        x_n = float(np.clip(x_pf / max(1, self._pf_w - 1), 0.0, 1.0))
        y_n = float(np.clip(y_pf / max(1, self._pf_h - 1), 0.0, 1.0))
        return x_n, y_n

    def _update_player_uv_small_from_full(self) -> None:
        """
        full-frame player_center를 playfield->small 좌표(u,v)로 변환.
        """
        try:
            px, py = map(int, self.player_center)

            x_pf = float(px - self._playfield_x0)
            y_pf = float(py - self._playfield_y0)

            u = int(round(x_pf * (self.obs_out_size - 1) / max(1, (self._pf_w - 1))))
            v = int(round(y_pf * (self.obs_out_size - 1) / max(1, (self._pf_h - 1))))

            u = int(np.clip(u, 0, self.obs_out_size - 1))
            v = int(np.clip(v, 0, self.obs_out_size - 1))
            self._last_player_uv_small = (u, v)
        except Exception:
            self._last_player_uv_small = (self.obs_out_size // 2, self.obs_out_size // 2)

    # -------------------------
    # meta pixels / marker
    # -------------------------
    def _inject_meta_pixels_ch0_only(self, ch0_01: np.ndarray) -> np.ndarray:
        try:
            x_n, y_n = self.last_xy_norm
            c = float(self.last_conf)

            x_n = float(np.clip(x_n, 0.0, 1.0))
            y_n = float(np.clip(y_n, 0.0, 1.0))
            c = float(np.clip(c, 0.0, 1.0))

            p = int(self.meta_patch)
            if p > 0 and ch0_01.shape[0] >= p and ch0_01.shape[1] >= p * 3:
                ch0_01[0:p, 0:p] = x_n
                ch0_01[0:p, p:2 * p] = y_n
                ch0_01[0:p, 2 * p:3 * p] = c
        except Exception:
            pass
        return ch0_01

    def _stamp_player_marker_ch0(self, ch0_01: np.ndarray) -> None:
        if not self.mark_player_on_ch0:
            return
        if ch0_01 is None or ch0_01.size == 0:
            return

        s = int(ch0_01.shape[0])
        if s <= 0:
            return

        cx, cy = self._last_player_uv_small
        cx = int(np.clip(cx, 0, s - 1))
        cy = int(np.clip(cy, 0, s - 1))

        r = int(self.marker_half)
        if r <= 0:
            return

        v = float(self.marker_value)
        if self.marker_use_conf:
            c = float(np.clip(self.last_conf, 0.0, 1.0))
            v *= max(float(self.marker_min_scale), c)

        y1 = max(0, cy - r)
        y2 = min(s, cy + r + 1)
        x1 = max(0, cx - r)
        x2 = min(s, cx + r + 1)

        ch0_01[cy, x1:x2] = v
        ch0_01[y1:y2, cx] = v

    # -------------------------
    # bullets / risk
    # -------------------------
    def _get_bullet_kernel(self):
        k = int(self.bullet_close_morph)
        if k <= 0:
            return None
        if self._bullet_kernel is not None and self._bullet_kernel_k == k:
            return self._bullet_kernel
        ksz = 2 * k + 1
        self._bullet_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
        self._bullet_kernel_k = k
        return self._bullet_kernel

    def _compute_bullet_mask_u8_small(self, bgr_small: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(bgr_small, cv2.COLOR_BGR2HSV)
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]
        mask = (s >= int(self.bullet_hsv_s_min)) & (v >= int(self.bullet_hsv_v_min)) & (
            v <= int(self.bullet_hsv_v_max)
        )
        mask_u8 = (mask.astype(np.uint8) * 255)

        kernel = self._get_bullet_kernel()
        if kernel is not None:
            mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=1)
            mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=1)

        return mask_u8

    @staticmethod
    def _mad_u8(x: np.ndarray):
        med = float(np.median(x))
        mad = float(np.median(np.abs(x.astype(np.float32) - med)))
        return med, mad

    def _compute_bullet_mask_u8_small_robust(self, bgr_small: np.ndarray, diff_small_u8: np.ndarray) -> np.ndarray:
        if not self.enable_bullet_channels:
            return self._zeros_small_u8

        hsv_mask = self._compute_bullet_mask_u8_small(bgr_small)
        if not self.use_motion_for_bullets:
            return hsv_mask

        med, mad = self._mad_u8(diff_small_u8)
        thr = max(float(self.diff_bullet_min), med + float(self.diff_bullet_k_mad) * mad)
        motion = (diff_small_u8.astype(np.float32) >= float(thr)).astype(np.uint8) * 255

        comb = cv2.bitwise_or(hsv_mask, motion)
        fill = float(np.mean(comb > 0))

        # 포화 방지: motion이 화면을 덮으면 HSV로 fallback
        if fill >= float(self.max_bullet_fill_ratio):
            comb = hsv_mask

        kernel = self._get_bullet_kernel()
        if kernel is not None:
            comb = cv2.morphologyEx(comb, cv2.MORPH_OPEN, kernel, iterations=1)
            comb = cv2.morphologyEx(comb, cv2.MORPH_CLOSE, kernel, iterations=1)

        return comb

    def _maybe_invert_gray_small(self, gray_small_u8: np.ndarray) -> np.ndarray:
        if not self.auto_invert_gray:
            self._last_inverted = False
            return gray_small_u8

        m = float(np.mean(gray_small_u8, dtype=np.float32) / 255.0)
        inv = bool(m >= float(self.invert_mean_thr))
        self._last_inverted = inv
        if inv:
            return (255 - gray_small_u8)
        return gray_small_u8

    def _compute_risk_heat_small(self, bullet_mask_u8_small: np.ndarray) -> np.ndarray:
        if bullet_mask_u8_small is None or bullet_mask_u8_small.size == 0:
            return self._zeros_small_f32
        if int(np.count_nonzero(bullet_mask_u8_small)) == 0:
            return self._zeros_small_f32

        inv = cv2.bitwise_not(bullet_mask_u8_small)  # 탄=0, 배경=255
        dist = cv2.distanceTransform(inv, distanceType=cv2.DIST_L2, maskSize=3)

        tau = max(1e-6, float(self.risk_tau_px))
        risk = np.exp(-dist / tau).astype(np.float32)

        # per-frame max normalize는 포화(=항상 1.00)를 쉽게 만든다.
        if self.risk_use_max_normalize:
            m = float(risk.max())
            if m > 1e-6:
                risk *= (1.0 / m)

        if self.risk_clip_max is not None:
            risk = np.clip(risk, 0.0, float(self.risk_clip_max), out=risk)

        return risk

    def _compute_local_risk_stats(self, risk_01: np.ndarray) -> None:
        self.risk_local_valid = False
        self.risk_local_mean = 0.0
        self.risk_local_p90 = 0.0
        self.risk_local_p99 = 0.0
        self.risk_local_bg_frac = 0.0
        self.risk_local_max = 0.0

        if not self.local_risk_enable:
            return
        if risk_01 is None or risk_01.size == 0:
            return

        conf = float(self.last_conf)
        if conf < float(self.local_risk_conf_thr):
            return

        s = int(risk_01.shape[0])
        if s <= 0:
            return

        u, v = self._last_player_uv_small
        u = int(np.clip(u, 0, s - 1))
        v = int(np.clip(v, 0, s - 1))

        r = int(self.local_risk_radius)
        if r <= 0:
            return

        y1 = max(0, v - r)
        y2 = min(s, v + r + 1)
        x1 = max(0, u - r)
        x2 = min(s, u + r + 1)

        roi = risk_01[y1:y2, x1:x2]
        if roi.size == 0:
            return

        rr_all = roi.astype(np.float32, copy=False).reshape(-1)
        self.risk_local_max = float(np.max(rr_all))

        if self.local_risk_exclude_saturated:
            sat_thr = float(self.local_risk_sat_thr)
            bg = rr_all[rr_all < sat_thr]
        else:
            bg = rr_all

        if bg.size == 0:
            return

        self.risk_local_bg_frac = float(bg.size / rr_all.size)
        if self.risk_local_bg_frac < float(self.local_risk_min_valid_bg_frac):
            return

        try:
            self.risk_local_mean = float(np.mean(bg))
            self.risk_local_p90 = float(np.quantile(bg, 0.90))
            self.risk_local_p99 = float(np.quantile(bg, 0.99))
            self.risk_local_valid = True
        except Exception:
            self.risk_local_valid = False

    # -------------------------
    # main
    # -------------------------
    def make_state(self, img_bgr: np.ndarray) -> np.ndarray:
        # 1) tracker step (full-frame)
        now = time.time()
        if self._track_pause_active and now < float(self._track_pause_until):
            bbox, conf = None, 0.0
        else:
            if self._track_pause_active and self._track_pause_resume_reset_pending:
                try:
                    self.tracker.reset()
                except Exception:
                    pass
                self._track_pause_resume_reset_pending = False
                self._track_pause_active = False

            bbox, conf = self.tracker.step(img_bgr)

        if bbox is not None:
            self._last_bbox_full = bbox
            x, y, w, h = map(int, bbox)
            cx = int(round(x + 0.5 * w))
            cy = int(round(y + 0.5 * h))
            self.player_center = (cx, cy)
            self.last_conf = float(np.clip(conf, 0.0, 1.0))

            # ✅ playfield 기준으로 정규화
            self.last_xy_norm = self._full_xy_to_playfield_norm(cx, cy)

            # ✅ marker uv도 playfield 기준으로 변환
            self._update_player_uv_small_from_full()
        else:
            self.last_conf = float(np.clip(conf, 0.0, 1.0))
            # last_xy_norm / player_center는 마지막 유효값 유지

        # 2) UI 제외 playfield crop -> small resize
        pf = self._crop_playfield(img_bgr)
        interp = cv2.INTER_AREA if max(pf.shape[0], pf.shape[1]) >= self.obs_out_size else cv2.INTER_LINEAR
        bgr_small = cv2.resize(pf, (self.obs_out_size, self.obs_out_size), interpolation=interp)

        gray_small_u8 = cv2.cvtColor(bgr_small, cv2.COLOR_BGR2GRAY)

        # 3) auto invert
        gray_small_u8 = self._maybe_invert_gray_small(gray_small_u8)

        if self._prev_gray_small_u8 is None or self._prev_gray_small_u8.shape != gray_small_u8.shape:
            diff_small_u8 = self._zeros_small_u8
        else:
            diff_small_u8 = cv2.absdiff(gray_small_u8, self._prev_gray_small_u8)

        self._prev_gray_small_u8 = gray_small_u8.copy()

        # 4) bullet + risk (playfield 기준)
        if self.enable_bullet_channels:
            bullet_mask_u8 = self._compute_bullet_mask_u8_small_robust(bgr_small, diff_small_u8)
            risk_01 = self._compute_risk_heat_small(bullet_mask_u8)
        else:
            bullet_mask_u8 = self._zeros_small_u8
            risk_01 = self._zeros_small_f32

        self.bullet_candidate_mask = (bullet_mask_u8 > 0).astype(np.uint8)  # 0/1
        self.risk_heatmap = risk_01.astype(np.float32)

        # 5) local risk stats
        self._compute_local_risk_stats(self.risk_heatmap)

        # 6) obs assemble
        self._obs_buf[0, :, :] = gray_small_u8.astype(np.float32) * (1.0 / 255.0)
        self._stamp_player_marker_ch0(self._obs_buf[0])

        if diff_small_u8 is self._zeros_small_u8:
            self._obs_buf[1, :, :] = 0.0
        else:
            self._obs_buf[1, :, :] = diff_small_u8.astype(np.float32) * (1.0 / 255.0)

        if bullet_mask_u8 is self._zeros_small_u8:
            self._obs_buf[2, :, :] = 0.0
        else:
            self._obs_buf[2, :, :] = bullet_mask_u8.astype(np.float32) * (1.0 / 255.0)

        if risk_01 is self._zeros_small_f32:
            self._obs_buf[3, :, :] = 0.0
        else:
            self._obs_buf[3, :, :] = risk_01

        self._inject_meta_pixels_ch0_only(self._obs_buf[0])

        obs4 = self._obs_buf

        # ---- debug windows (playfield만 시각화) ----
        if self.show_obs_debug:
            try:
                self._ensure_obs_window()

                ch = int(np.clip(self.obs_debug_channel, 0, 3))
                vis = (np.clip(obs4[ch], 0.0, 1.0) * 255.0).astype(np.uint8)
                vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

                up = int(max(1, self.debug_upscale))
                if up != 1:
                    vis = cv2.resize(vis, (vis.shape[1] * up, vis.shape[0] * up), interpolation=cv2.INTER_NEAREST)

                px, py = self.player_center
                xn, yn = self.last_xy_norm

                line1 = f"ch{ch} conf={self.last_conf:.2f} px=({int(px)},{int(py)}) n=({xn:.3f},{yn:.3f})"
                if self.risk_local_valid:
                    line2 = f"Lp90={self.risk_local_p90:.2f} Lp99={self.risk_local_p99:.2f} bg={self.risk_local_bg_frac:.2f}"
                else:
                    line2 = f"L=NA bg={self.risk_local_bg_frac:.2f}"

                font_scale = float(self.debug_font_scale)
                thickness = int(self.debug_thickness)

                def put_text_outline(img, text, org):
                    cv2.putText(
                        img, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA
                    )
                    cv2.putText(
                        img, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, (255, 255, 255), thickness, cv2.LINE_AA
                    )

                y0 = 24
                put_text_outline(vis, line1, (10, y0))
                put_text_outline(vis, line2, (10, y0 + 28))

                cv2.imshow(self.win_crop, vis)
            except Exception:
                pass

        if self.show_reimu_debug and (self.reimu_dbg_view is not None):
            try:
                self.reimu_dbg_view.render(img_bgr)
            except Exception as e:
                print("[reimu_dbg_view.render ERROR]", repr(e))

        return obs4.copy()
