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
      ch0: crop_gray (0..1) + player marker + meta pixels(xy/conf)
      ch1: absdiff(current_gray, prev_gray) (0..1)
      ch2: bullet_candidate_mask (0..1)
      ch3: risk_heatmap (distanceTransform 기반, 0..1)
    """

    def __init__(
        self,
        screen,
        obs_out_size: int = 128,
        crop_size: int = 256,
        use_fallback_full_preprocess: bool = True,  # (호환용) 현재 미사용
    ):
        self.screen = screen

        self.obs_out_size = int(obs_out_size)
        self.crop_size = int(crop_size)
        self.obs_channels = 4

        img0 = self.screen.capture()
        h0, w0 = img0.shape[:2]
        self.H, self.W = int(h0), int(w0)

        # playfield width 캐시
        self._playfield_ratio = float(getattr(self.screen, "PLAYFIELD_RIGHT_RATIO", 0.70))
        self._playfield_w = max(1, min(self.W, int(self.W * self._playfield_ratio)))

        # tracker
        self.tracker = ReimuTrackerCV()

        # 정책/리워드용 좌표/신뢰도 (playfield 기준 정규화)
        self.last_xy_norm: Tuple[float, float] = (0.5, 0.78)
        self.last_conf: float = 0.0

        # det None일 때 유지
        self.player_center: Tuple[int, int] = (w0 // 2, int(h0 * 0.78))

        # meta pixels
        self.meta_patch: int = 4

        # prev gray (obs_out_size 기준으로 저장)
        self._prev_gray_small_u8: Optional[np.ndarray] = None

        # ----- auto inversion / illumination robustness -----
        self.auto_invert_gray: bool = True
        self.invert_mean_thr: float = 0.58  # 0..1, 이 이상이면 invert
        self._last_inverted: bool = False

        # ----- bullet/background separation -----
        self.use_motion_for_bullets: bool = True
        self.diff_bullet_min: int = 10
        self.diff_bullet_k_mad: float = 3.0
        self.max_bullet_fill_ratio: float = 0.35

        # ----- Reimu brighten 제거 -----
        # 트래커 bbox는 crop 중심 추정/메타(xy/conf)에만 사용됨.
        self._last_bbox_full: Optional[Tuple[int, int, int, int]] = None  # (x,y,w,h)

        # bullet/risk
        self.enable_bullet_channels: bool = True
        self.bullet_hsv_s_min: int = 40
        self.bullet_hsv_v_min: int = 140
        self.bullet_hsv_v_max: int = 255
        self.bullet_close_morph: int = 0  # 0이면 morph 스킵

        self.risk_tau_px: float = 8.0
        self.risk_clip_max: float = 1.0

        # ===== player marker on ch0 (학습용) =====
        # ✅ ch0에만 마커를 찍어서 다른 채널(diff/bullet/risk)을 오염시키지 않는다.
        # ✅ 마커 위치는 "항상 중앙"이 아니라, 트래커가 잡은 player_center가
        #    crop 내부에서 어디에 위치하는지(u,v)로 변환해 찍는다.
        self.mark_player_on_ch0: bool = True
        self.marker_half: int = 2            # 2면 5x5 십자 정도
        self.marker_value: float = 1.0       # ch0(0..1)에 찍을 값
        self.marker_use_conf: bool = True    # conf로 밝기 스케일
        self.marker_min_scale: float = 0.35  # conf가 낮아도 이 정도는 찍음
        self._last_player_uv_small: Tuple[int, int] = (self.obs_out_size // 2, self.obs_out_size // 2)

        # 레이무 디버그 창
        self.show_reimu_debug: bool = False
        dbg_cfg = DebugViewConfig(
            window_name="debug_hell",
            enable_keys=False,
            wait_ms=1,
        )
        self.reimu_dbg_view = ReimuTrackerDebugView(self.tracker, cfg=dbg_cfg)

        # OBS 디버그
        self.show_obs_debug: bool = False
        self.win_crop: str = "OBS_CROP"
        self._obs_win_inited: bool = False

        # OBS 디버그에 무엇을 보여줄지 (0/1/2/3)
        self.obs_debug_channel: int = 0

        # tracker pause (bomb etc.)
        self._track_pause_until: float = 0.0
        self._track_pause_active: bool = False
        self._track_pause_resume_reset_pending: bool = False

        # =========================
        # ✅ 최적화용 캐시/버퍼
        # =========================
        s = self.obs_out_size
        self._zeros_small_u8 = np.zeros((s, s), dtype=np.uint8)
        self._zeros_small_f32 = np.zeros((s, s), dtype=np.float32)
        self._obs_buf = np.empty((4, s, s), dtype=np.float32)  # 재사용

        # bullet morph kernel cache
        self._bullet_kernel = None
        self._bullet_kernel_k = -1

    def reset(self):
        self.tracker.reset()
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

    def _ensure_obs_window(self):
        if self._obs_win_inited:
            return
        try:
            cv2.namedWindow(self.win_crop, cv2.WINDOW_NORMAL)
        except Exception:
            pass
        self._obs_win_inited = True

    @staticmethod
    def _crop_square_bgr(img_bgr: np.ndarray, cx: int, cy: int, size: int):
        h, w = img_bgr.shape[:2]
        size = int(size)
        half = size // 2

        cx = int(np.clip(cx, half, w - half - 1))
        cy = int(np.clip(cy, half, h - half - 1))

        x1 = int(cx - half)
        y1 = int(cy - half)
        x2 = x1 + size
        y2 = y1 + size
        return img_bgr[y1:y2, x1:x2], (cx, cy), (x1, y1)

    def _inject_meta_pixels_ch0_only(self, ch0_01: np.ndarray) -> np.ndarray:
        try:
            x_n, y_n = self.last_xy_norm
            c = float(self.last_conf)

            x_n = float(np.clip(x_n, 0.0, 1.0))
            y_n = float(np.clip(y_n, 0.0, 1.0))
            c = float(np.clip(c, 0.0, 1.0))

            p = int(self.meta_patch)
            if ch0_01.shape[0] >= p and ch0_01.shape[1] >= p * 3:
                ch0_01[0:p, 0:p] = x_n
                ch0_01[0:p, p:2 * p] = y_n
                ch0_01[0:p, 2 * p:3 * p] = c
        except Exception:
            pass
        return ch0_01

    def _update_player_uv_small(self, crop_xy: Tuple[int, int]) -> None:
        """
        전체화면 좌표 player_center(px,py)를
        crop 좌상단(x1,y1) 기준으로 crop 내부 좌표로 바꾸고,
        obs_out_size 해상도(u,v)로 스케일해서 저장.
        """
        try:
            x1, y1 = map(int, crop_xy)
            px, py = map(int, self.player_center)

            scale = float(self.obs_out_size) / float(max(1, self.crop_size))
            u = int(round((px - x1) * scale))
            v = int(round((py - y1) * scale))

            u = int(np.clip(u, 0, self.obs_out_size - 1))
            v = int(np.clip(v, 0, self.obs_out_size - 1))
            self._last_player_uv_small = (u, v)
        except Exception:
            self._last_player_uv_small = (self.obs_out_size // 2, self.obs_out_size // 2)

    def _stamp_player_marker_ch0(self, ch0_01: np.ndarray) -> None:
        """
        ch0(0..1)에만 플레이어 마커(십자)를 찍는다.
        위치는 self._last_player_uv_small(u,v)를 사용한다.
        """
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

    def _full_xy_to_playfield_norm(self, cx: int, cy: int) -> Tuple[float, float]:
        x_n = float(np.clip(cx / max(1, self._playfield_w - 1), 0.0, 1.0))
        y_n = float(np.clip(cy / max(1, self.H - 1), 0.0, 1.0))
        return x_n, y_n

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
        mask = (s >= int(self.bullet_hsv_s_min)) & (v >= int(self.bullet_hsv_v_min)) & (v <= int(self.bullet_hsv_v_max))
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
        if fill >= float(self.max_bullet_fill_ratio):
            comb = motion

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

        m = float(risk.max())
        if m > 1e-6:
            risk *= (1.0 / m)

        if self.risk_clip_max is not None:
            risk = np.clip(risk, 0.0, float(self.risk_clip_max), out=risk)

        return risk

    def make_state(self, img_bgr: np.ndarray) -> np.ndarray:
        # 1) tracker step (전체 화면 기준)
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
            self.last_xy_norm = self._full_xy_to_playfield_norm(cx, cy)

        # 2) crop (crop 좌상단(x1,y1) 필요)
        cx, cy = self.player_center
        crop_bgr, _, (x1, y1) = self._crop_square_bgr(img_bgr, cx, cy, self.crop_size)

        # ✅ 현재 crop 내부에서 player_center가 어디인지 uv로 갱신
        self._update_player_uv_small((x1, y1))

        # 3) small resize
        interp = cv2.INTER_AREA if self.crop_size >= self.obs_out_size else cv2.INTER_LINEAR
        bgr_small = cv2.resize(crop_bgr, (self.obs_out_size, self.obs_out_size), interpolation=interp)

        gray_small_u8 = cv2.cvtColor(bgr_small, cv2.COLOR_BGR2GRAY)

        # 3.5) auto invert
        gray_small_u8 = self._maybe_invert_gray_small(gray_small_u8)

        if self._prev_gray_small_u8 is None or self._prev_gray_small_u8.shape != gray_small_u8.shape:
            diff_small_u8 = self._zeros_small_u8
        else:
            diff_small_u8 = cv2.absdiff(gray_small_u8, self._prev_gray_small_u8)

        self._prev_gray_small_u8 = gray_small_u8.copy()

        # 4) bullet + risk
        if self.enable_bullet_channels:
            bullet_mask_u8 = self._compute_bullet_mask_u8_small_robust(bgr_small, diff_small_u8)
            risk_01 = self._compute_risk_heat_small(bullet_mask_u8)
        else:
            bullet_mask_u8 = self._zeros_small_u8
            risk_01 = self._zeros_small_f32

        # 5) float32 채널 구성 (0..1) - obs buffer 재사용
        self._obs_buf[0, :, :] = gray_small_u8.astype(np.float32) * (1.0 / 255.0)

        # ✅ 플레이어 마커는 ch0에만 (uv 기반)
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

        # 6) meta pixels (ch0만)
        self._inject_meta_pixels_ch0_only(self._obs_buf[0])

        obs4 = self._obs_buf

        # ---- debug windows ----
        if self.show_obs_debug:
            try:
                self._ensure_obs_window()
                ch = int(np.clip(self.obs_debug_channel, 0, 3))
                vis = (np.clip(obs4[ch], 0.0, 1.0) * 255.0).astype(np.uint8)
                vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

                # 디버그 텍스트(옵션): conf와 uv 표시
                u, v = self._last_player_uv_small
                cv2.putText(
                    vis,
                    f"ch{ch} conf={self.last_conf:.2f} uv=({u},{v}) inv={int(self._last_inverted)}",
                    (5, 14),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

                cv2.imshow(self.win_crop, vis)
            except Exception:
                pass

        if self.show_reimu_debug and (self.reimu_dbg_view is not None):
            try:
                self.reimu_dbg_view.render(img_bgr)
            except Exception as e:
                print("[reimu_dbg_view.render ERROR]", repr(e))

        return obs4.copy()
