# env/obs_builder.py
import cv2
import numpy as np

from env.reimu_tracker_cv import ReimuTrackerCV
from env.reimu_tracker_debug_view import ReimuTrackerDebugView, DebugViewConfig


class ObsBuilder:
    """
    4채널 관측 (float32):
      ch0: crop_gray (0..1) + meta pixels(xy/conf)
      ch1: absdiff(current_gray, prev_gray) (0..1)
      ch2: bullet_candidate_mask (0..1)
      ch3: risk_heatmap (distanceTransform 기반, 0..1)
    """

    def __init__(self, screen, obs_out_size=128, crop_size=256, use_fallback_full_preprocess=True):
        self.screen = screen

        self.obs_out_size = int(obs_out_size)
        self.crop_size = int(crop_size)
        self.obs_channels = 4

        img0 = self.screen.capture()
        h0, w0 = img0.shape[:2]
        self.H, self.W = h0, w0

        # playfield width 캐시
        self._playfield_ratio = float(getattr(self.screen, "PLAYFIELD_RIGHT_RATIO", 0.70))
        self._playfield_w = max(1, min(self.W, int(self.W * self._playfield_ratio)))

        # tracker
        self.tracker = ReimuTrackerCV()

        # 정책/리워드용 좌표/신뢰도 (playfield 기준 정규화)
        self.last_xy_norm = (0.5, 0.78)
        self.last_conf = 0.0

        # det None일 때 유지
        self.player_center = (w0 // 2, int(h0 * 0.78))

        # meta pixels
        self.meta_patch = 4

        # prev gray (obs_out_size 기준으로 저장)
        self._prev_gray_small_u8 = None

        # bullet/risk (작은 해상도에서 처리)
        self.enable_bullet_channels = True
        self.bullet_hsv_s_min = 40
        self.bullet_hsv_v_min = 140
        self.bullet_hsv_v_max = 255
        self.bullet_close_morph = 0

        self.risk_tau_px = 8.0
        self.risk_clip_max = 1.0

        # 디버그 창 (원하면 True)
        self.show_reimu_debug = True
        dbg_cfg = DebugViewConfig(
            window_name="debug_hell",
            enable_keys=False,  # waitKey는 main loop에서만!
            wait_ms=1,
        )
        self.reimu_dbg_view = ReimuTrackerDebugView(self.tracker, cfg=dbg_cfg)

        # OBS crop 디버그는 기본 OFF (렉 원인 될 수 있음)
        self.show_obs_debug = True
        self.win_crop = "OBS_CROP"
        self._obs_win_inited = False

    def reset(self):
        self.tracker.reset()
        self.player_center = (self.W // 2, int(self.H * 0.78))
        self.last_xy_norm = (0.5, 0.78)
        self.last_conf = 0.0
        self._prev_gray_small_u8 = None

    def on_player_death(self):
        # 목숨 깎였을 때 tracker 상태 리셋
        try:
            self.tracker.reset()
        except Exception:
            pass
        # 좌표 캐시는 유지해도 되지만, 원하면 초기화 가능
        # self.last_xy_norm = (0.5, 0.78)
        # self.last_conf = 0.0

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

    def _crop_square_bgr(self, img_bgr, cx, cy, size):
        h, w = img_bgr.shape[:2]
        size = int(size)
        half = size // 2

        cx = int(np.clip(cx, half, w - half - 1))
        cy = int(np.clip(cy, half, h - half - 1))

        x1 = int(cx - half)
        y1 = int(cy - half)
        x2 = x1 + size
        y2 = y1 + size
        return img_bgr[y1:y2, x1:x2], (cx, cy)

    def _inject_meta_pixels_ch0_only(self, ch0_01: np.ndarray) -> np.ndarray:
        # ch0_01: (obs_out_size, obs_out_size) float32
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

    def _full_xy_to_playfield_norm(self, cx: int, cy: int) -> tuple[float, float]:
        x_n = float(np.clip(cx / max(1, self._playfield_w - 1), 0.0, 1.0))
        y_n = float(np.clip(cy / max(1, self.H - 1), 0.0, 1.0))
        return x_n, y_n

    def _compute_bullet_mask_u8_small(self, bgr_small: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(bgr_small, cv2.COLOR_BGR2HSV)
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]
        mask = (s >= int(self.bullet_hsv_s_min)) & (v >= int(self.bullet_hsv_v_min)) & (v <= int(self.bullet_hsv_v_max))
        mask_u8 = (mask.astype(np.uint8) * 255)

        k = int(self.bullet_close_morph)
        if k > 0:
            ksz = 2 * k + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
            mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=1)
            mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=1)

        return mask_u8

    def _compute_risk_heat_small(self, bullet_mask_u8_small: np.ndarray) -> np.ndarray:
        # bullet=255, bg=0 가정이면 invert 필요
        inv = cv2.bitwise_not(bullet_mask_u8_small)  # 탄=0, 배경=255
        dist = cv2.distanceTransform(inv, distanceType=cv2.DIST_L2, maskSize=3)

        tau = max(1e-6, float(self.risk_tau_px))
        risk = np.exp(-dist / tau).astype(np.float32)

        m = float(risk.max())
        if m > 1e-6:
            risk /= m

        if self.risk_clip_max is not None:
            risk = np.clip(risk, 0.0, float(self.risk_clip_max))

        return risk.astype(np.float32, copy=False)

    def make_state(self, img_bgr: np.ndarray):
        # 1) tracker step (전체 화면 기준)
        bbox, conf = self.tracker.step(img_bgr)
        if bbox is not None:
            x, y, w, h = map(int, bbox)
            cx = int(round(x + 0.5 * w))
            cy = int(round(y + 0.5 * h))
            self.player_center = (cx, cy)
            self.last_conf = float(np.clip(conf, 0.0, 1.0))
            self.last_xy_norm = self._full_xy_to_playfield_norm(cx, cy)

        # 2) crop (det None이어도 마지막 center 유지)
        cx, cy = self.player_center
        crop_bgr, _ = self._crop_square_bgr(img_bgr, cx, cy, self.crop_size)

        # 3) ✅ 먼저 obs_out_size로 줄인 뒤, 나머지 연산은 전부 작은 이미지에서!
        interp = cv2.INTER_AREA if self.crop_size >= self.obs_out_size else cv2.INTER_LINEAR
        bgr_small = cv2.resize(crop_bgr, (self.obs_out_size, self.obs_out_size), interpolation=interp)

        gray_small_u8 = cv2.cvtColor(bgr_small, cv2.COLOR_BGR2GRAY)

        if self._prev_gray_small_u8 is None or self._prev_gray_small_u8.shape != gray_small_u8.shape:
            diff_small_u8 = np.zeros_like(gray_small_u8)
        else:
            diff_small_u8 = cv2.absdiff(gray_small_u8, self._prev_gray_small_u8)
        self._prev_gray_small_u8 = gray_small_u8

        # 4) bullet + risk (작은 이미지에서)
        if self.enable_bullet_channels:
            bullet_mask_u8 = self._compute_bullet_mask_u8_small(bgr_small)
            risk_01 = self._compute_risk_heat_small(bullet_mask_u8)
        else:
            bullet_mask_u8 = np.zeros_like(gray_small_u8, dtype=np.uint8)
            risk_01 = np.zeros((self.obs_out_size, self.obs_out_size), dtype=np.float32)

        # 5) float32 채널 구성 (0..1)
        ch0 = (gray_small_u8.astype(np.float32) / 255.0)
        ch1 = (diff_small_u8.astype(np.float32) / 255.0)
        ch2 = (bullet_mask_u8.astype(np.float32) / 255.0)
        ch3 = risk_01.astype(np.float32, copy=False)

        # 6) meta pixels (ch0만)
        ch0 = self._inject_meta_pixels_ch0_only(ch0)

        obs4 = np.stack([ch0, ch1, ch2, ch3], axis=0).astype(np.float32, copy=False)

        # ---- debug windows ----
        if self.show_obs_debug:
            try:
                self._ensure_obs_window()
                vis = (np.clip(ch3, 0.0, 1.0) * 255.0).astype(np.uint8)
                vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
                cv2.imshow(self.win_crop, vis)
            except Exception:
                pass

        if self.show_reimu_debug and (self.reimu_dbg_view is not None):
            try:
                self.reimu_dbg_view.render(img_bgr)
            except Exception as e:
                print("[reimu_dbg_view.render ERROR]", repr(e))

        return obs4
