# env/obs_builder.py
from __future__ import annotations

import time
from typing import Optional, Tuple

import cv2
import numpy as np


class ObsBuilder:
    """
    4채널 관측 (float32):
      ch0: gray (0..1)
      ch1: absdiff(current_gray, prev_gray) (0..1)
      ch2: bullet_candidate_mask (0..1)
      ch3: risk_heatmap (distanceTransform 기반, 0..1)

    ✅ TRACER/십자가/메타 완전 OFF 모드 + UI 제외(playfield crop):
    - 트래커 미사용
    - 관측은 "플레이필드 영역만" 잘라서 obs_out_size 정사각으로 리사이즈
    - last_xy_norm/last_conf/player_center는 외부 호환용 고정값 유지
    """

    def __init__(
        self,
        screen,
        obs_out_size: int = 128,
        crop_size: int = 256,  # (호환용) 더 이상 사용하지 않음
        use_fallback_full_preprocess: bool = True,  # (호환용) 현재 미사용
    ):
        self.screen = screen

        self.obs_out_size = int(obs_out_size)
        self.crop_size = int(crop_size)  # 호환용으로만 보관
        self.obs_channels = 4

        img0 = self.screen.capture()
        h0, w0 = img0.shape[:2]
        self.H, self.W = int(h0), int(w0)

        # playfield crop params (Screen 값을 그대로 사용)
        self._playfield_ratio = float(getattr(self.screen, "PLAYFIELD_RIGHT_RATIO", 0.67))
        self._pf_left_crop = float(getattr(self.screen, "PLAYFIELD_LEFT_CROP", 0.00))
        self._pf_right_crop = float(getattr(self.screen, "PLAYFIELD_RIGHT_CROP", 1.00))
        self._pf_top_crop = float(getattr(self.screen, "PLAYFIELD_TOP_CROP", 0.00))
        self._pf_bottom_crop = float(getattr(self.screen, "PLAYFIELD_BOTTOM_CROP", 1.00))

        # playfield width 캐시 (기존 유지: RewardEngine/기타 코드 호환)
        self._playfield_w = max(1, min(self.W, int(self.W * self._playfield_ratio)))

        # ===== 외부 호환용 상태 (고정값 유지) =====
        self.last_xy_norm: Tuple[float, float] = (0.5, 0.78)
        self.last_conf: float = 0.0
        self.player_center: Tuple[int, int] = (w0 // 2, int(h0 * 0.78))
        self._last_bbox_full: Optional[Tuple[int, int, int, int]] = None

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

        # bullet/risk
        self.enable_bullet_channels: bool = True
        self.bullet_hsv_s_min: int = 40
        self.bullet_hsv_v_min: int = 140
        self.bullet_hsv_v_max: int = 255
        self.bullet_close_morph: int = 0  # 0이면 morph 스킵

        self.risk_tau_px: float = 8.0
        self.risk_clip_max: float = 1.0

        # ===== TRACER/마커/메타 완전 OFF =====
        self.mark_player_on_ch0: bool = False
        self.meta_patch: int = 0  # (혹시 외부가 접근하더라도 안전)

        # OBS 디버그
        self.show_obs_debug: bool = True
        self.win_crop: str = "OBS_CROP"
        self._obs_win_inited: bool = False
        self.obs_debug_channel: int = 0  # 0/1/2/3

        # tracker pause (bomb etc.) - 인터페이스 호환용으로만 유지 (실제 사용 안 함)
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

        # (호환용) obs에 있던 속성들 접근 대비
        self._last_player_uv_small: Tuple[int, int] = (s // 2, int(s * 0.78))

    # -------------------------
    # lifecycle / hooks
    # -------------------------
    def reset(self):
        self.player_center = (self.W // 2, int(self.H * 0.78))
        self.last_xy_norm = (0.5, 0.78)
        self.last_conf = 0.0
        self._prev_gray_small_u8 = None
        self._last_inverted = False
        self._last_bbox_full = None

        self._track_pause_until = 0.0
        self._track_pause_active = False
        self._track_pause_resume_reset_pending = False

        s = self.obs_out_size
        self._last_player_uv_small = (s // 2, int(s * 0.78))

    def on_player_death(self):
        return

    def on_bomb_used(self, pause_sec: float = 2.0):
        # 트래커가 없으므로 실제 pause 의미 없음(호환용)
        now = time.time()
        self._track_pause_until = float(now + float(pause_sec))
        self._track_pause_active = True
        self._track_pause_resume_reset_pending = True

    def pump_key(self, key: int):
        return

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
    # playfield crop (BGR)
    # -------------------------
    def _crop_playfield_bgr(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Screen의 PLAYFIELD_RIGHT_RATIO + (LEFT/RIGHT/TOP/BOTTOM)_CROP 를 이용해
        UI 패널을 제외한 playfield BGR만 반환한다.
        """
        h, w = img_bgr.shape[:2]
        pf_w = int(round(w * float(self._playfield_ratio)))
        pf_w = int(np.clip(pf_w, 1, w))

        # 1) 먼저 좌측 playfield 영역만 자르기
        play = img_bgr[:, :pf_w]

        ph, pw = play.shape[:2]

        # 2) playfield 내부 crop 비율 적용
        x0 = int(round(pw * float(self._pf_left_crop)))
        x1 = int(round(pw * float(self._pf_right_crop)))
        y0 = int(round(ph * float(self._pf_top_crop)))
        y1 = int(round(ph * float(self._pf_bottom_crop)))

        # 안전 클램프
        x0 = int(np.clip(x0, 0, pw - 1))
        x1 = int(np.clip(x1, x0 + 1, pw))
        y0 = int(np.clip(y0, 0, ph - 1))
        y1 = int(np.clip(y1, y0 + 1, ph))

        return play[y0:y1, x0:x1]

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

    # -------------------------
    # main
    # -------------------------
    def make_state(self, img_bgr: np.ndarray) -> np.ndarray:
        # ✅ 0) UI 제외: playfield만 crop
        play_bgr = self._crop_playfield_bgr(img_bgr)

        # 1) playfield -> small resize (정사각)
        ph, pw = play_bgr.shape[:2]
        interp = cv2.INTER_AREA if max(ph, pw) >= self.obs_out_size else cv2.INTER_LINEAR
        bgr_small = cv2.resize(play_bgr, (self.obs_out_size, self.obs_out_size), interpolation=interp)

        gray_small_u8 = cv2.cvtColor(bgr_small, cv2.COLOR_BGR2GRAY)

        # 2) auto invert
        gray_small_u8 = self._maybe_invert_gray_small(gray_small_u8)

        # 3) diff
        if self._prev_gray_small_u8 is None or self._prev_gray_small_u8.shape != gray_small_u8.shape:
            diff_small_u8 = self._zeros_small_u8
        else:
            diff_small_u8 = cv2.absdiff(gray_small_u8, self._prev_gray_small_u8)

        self._prev_gray_small_u8 = gray_small_u8.copy()

        # 4) bullet + risk (playfield 기반으로만)
        if self.enable_bullet_channels:
            bullet_mask_u8 = self._compute_bullet_mask_u8_small_robust(bgr_small, diff_small_u8)
            risk_01 = self._compute_risk_heat_small(bullet_mask_u8)
        else:
            bullet_mask_u8 = self._zeros_small_u8
            risk_01 = self._zeros_small_f32

        # (추가) 외부(ActionMasker/Reward shaping)가 참조할 수 있게 속성으로 노출
        self.bullet_candidate_mask = (bullet_mask_u8 > 0).astype(np.uint8)   # 0/1
        self.risk_heatmap = risk_01.astype(np.float32)

        # 5) float32 채널 구성 (0..1)
        self._obs_buf[0, :, :] = gray_small_u8.astype(np.float32) * (1.0 / 255.0)

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

        obs4 = self._obs_buf

        # ---- debug window ----
        if self.show_obs_debug:
            try:
                self._ensure_obs_window()
                ch = int(np.clip(self.obs_debug_channel, 0, 3))
                vis = (np.clip(obs4[ch], 0.0, 1.0) * 255.0).astype(np.uint8)
                vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
                cv2.putText(
                    vis,
                    f"ch{ch} PLAYFIELD_ONLY TRACKER_OFF inv={int(self._last_inverted)}",
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

        return obs4.copy()
