# env/screen.py
import os
import time
from typing import Optional, Tuple

import win32gui
import cv2
import numpy as np

from env.screen_util import (
    find_touhou_window,
    GrayCache,
    load_ui_config,
    load_score_template,
    create_capture_backend,
    dump_capture_debug,
)

# ✅ 신규: metrics 유틸 (Canny N프레임 캐시 + 다운샘플)
from env.screen_util.metrics import (
    CannyCacheConfig,
    CannyEdgeRatioCache,
    downsample_gray,
    mean_std,
    bright_ratio,
)

Rect = Tuple[int, int, int, int]


class Screen:
    # ===== playfield split =====
    PLAYFIELD_RIGHT_RATIO = 0.67

    PLAYFIELD_TOP_CROP = 0.00
    PLAYFIELD_BOTTOM_CROP = 1.00
    PLAYFIELD_LEFT_CROP = 0.00
    PLAYFIELD_RIGHT_CROP = 1.00

    # ===== UI panel heuristics =====
    UI_EDGE_RATIO_THR = 0.040
    UI_STD_MIN = 15.0
    UI_STD_MAX = 80.0
    UI_MEAN_MIN = 20.0
    UI_MEAN_MAX = 200.0

    def __init__(self, mode="low"):
        self.mode = mode
        self.hwnd = find_touhou_window()
        if not self.hwnd:
            raise Exception("동방홍마향 창을 찾을 수 없음")

        title = win32gui.GetWindowText(self.hwnd)
        win_rect = win32gui.GetWindowRect(self.hwnd)
        print("[DEBUG] 잡은 창 제목:", title)
        print("[DEBUG] 창 좌표 (left, top, right, bottom):", win_rect)
        self.win_rect = win_rect

        # ===== capture rect cache (client rect in screen coords) =====
        self._cap_rect = self._get_client_rect_screen()  # (l,t,r,b)
        self._cap_i = 0
        self._cap_refresh_every = 30

        # ✅ 프레임 인덱스 (Canny N프레임 캐시에 사용)
        self._frame_idx = 0

        # gray cache
        self._gray_cache = GrayCache()

        # zero-frame cache
        self._zero_frame = None
        self._zero_frame_shape = None  # (h,w)

        # score screen template
        self.score_tmpl = None
        self.score_roi = None  # (x,y,w,h) in captured image coords

        # debug dump options
        self.debug_dump_on_start = False
        self.debug_dump_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "debug_caps"))
        self.debug_dump_annotated = True

        # load config/template
        env_dir = os.path.dirname(__file__)
        cfg = load_ui_config(env_dir)
        self.score_roi = cfg["score_roi"]
        self.debug_dump_on_start = cfg["capture_debug_dump_on_start"]
        if cfg["capture_debug_dump_dir"]:
            self.debug_dump_dir = cfg["capture_debug_dump_dir"]
        self.debug_dump_annotated = cfg["capture_debug_dump_annotated"]

        self.score_tmpl = load_score_template(env_dir)

        # capture backend
        self._dx_target_fps = 60
        self._backend = create_capture_backend(self._cap_rect, prefer_dxcam=True, dxcam_target_fps=self._dx_target_fps)

        # ✅ Canny 캐시 (UI / Danger 각각 따로)
        # - UI 패널은 비교적 큼 → max_side 160
        # - Danger ROI는 상대적으로 작음 → max_side 128 정도로 충분
        self._ui_edges = CannyEdgeRatioCache(
            CannyCacheConfig(every_n_frames=3, max_side=160, thr1=80, thr2=160)
        )
        self._danger_edges = CannyEdgeRatioCache(
            CannyCacheConfig(every_n_frames=3, max_side=128, thr1=50, thr2=120)
        )

        if self.debug_dump_on_start:
            self.dump_capture_debug(tag="start")

    # ----------------------------
    # cleanup
    # ----------------------------
    def close(self):
        try:
            if self._backend is not None:
                self._backend.close()
        except Exception:
            pass
        self._backend = None

    # ----------------------------
    # rect helpers
    # ----------------------------
    def _get_client_rect_screen(self) -> Rect:
        """
        윈도우 "클라이언트 영역"을 스크린 좌표로 변환해 반환.
        """
        try:
            l, t, r, b = win32gui.GetClientRect(self.hwnd)
            (sx0, sy0) = win32gui.ClientToScreen(self.hwnd, (l, t))
            (sx1, sy1) = win32gui.ClientToScreen(self.hwnd, (r, b))
            if sx1 > sx0 and sy1 > sy0:
                return (int(sx0), int(sy0), int(sx1), int(sy1))
        except Exception:
            pass

        try:
            L, T, R, B = win32gui.GetWindowRect(self.hwnd)
            return (int(L), int(T), int(R), int(B))
        except Exception:
            return (0, 0, 640, 480)

    def _get_capture_rect(self) -> Rect:
        self._cap_i += 1
        if (self._cap_i % self._cap_refresh_every) == 0:
            try:
                self._cap_rect = self._get_client_rect_screen()
            except Exception:
                pass
        return self._cap_rect

    # ----------------------------
    # gray
    # ----------------------------
    def gray(self, img_bgr: Optional[np.ndarray]) -> Optional[np.ndarray]:
        return self._gray_cache.gray(img_bgr)

    # ----------------------------
    # capture
    # ----------------------------
    def _get_zero_frame(self, h: int, w: int) -> np.ndarray:
        key = (h, w)
        if self._zero_frame is None or self._zero_frame_shape != key:
            self._zero_frame = np.zeros((h, w, 3), dtype=np.uint8)
            self._zero_frame_shape = key
        return self._zero_frame

    def capture(self) -> np.ndarray:
        """
        return: BGR uint8 (H,W,3)
        """
        left, top, right, bottom = self._get_capture_rect()
        w = max(1, int(right - left))
        h = max(1, int(bottom - top))

        # ✅ 프레임 카운터 증가 (Canny 캐시용)
        self._frame_idx += 1

        # new frame => reset gray cache
        self._gray_cache.reset()

        try:
            img = self._backend.capture((left, top, right, bottom))
            if img is None or not isinstance(img, np.ndarray) or img.size == 0:
                return self._get_zero_frame(h, w)
            return img
        except Exception:
            return self._get_zero_frame(h, w)

    # ----------------------------
    # debug dump
    # ----------------------------
    def dump_capture_debug(self, tag: str = "manual"):
        try:
            self._cap_rect = self._get_client_rect_screen()
            img = self.capture()

            dump_capture_debug(
                img_bgr=img,
                cap_rect=self._cap_rect,
                debug_dump_dir=self.debug_dump_dir,
                tag=tag,
                debug_dump_annotated=self.debug_dump_annotated,
                playfield_right_ratio=self.PLAYFIELD_RIGHT_RATIO,
                playfield_crops=(
                    self.PLAYFIELD_LEFT_CROP,
                    self.PLAYFIELD_RIGHT_CROP,
                    self.PLAYFIELD_TOP_CROP,
                    self.PLAYFIELD_BOTTOM_CROP,
                ),
                score_roi=self.score_roi,
            )
        except Exception as e:
            print("[SCREEN][DUMP] failed:", repr(e))

    # ----------------------------
    # Existing APIs
    # ----------------------------
    def get_playfield_gray(self, img_bgr: np.ndarray, gray: Optional[np.ndarray] = None) -> np.ndarray:
        if gray is None:
            gray = self.gray(img_bgr)
        h, w = gray.shape

        x2 = int(w * self.PLAYFIELD_RIGHT_RATIO)
        play = gray[:, :x2]

        ph, pw = play.shape
        x1 = int(pw * self.PLAYFIELD_LEFT_CROP)
        x2 = int(pw * self.PLAYFIELD_RIGHT_CROP)
        y1 = int(ph * self.PLAYFIELD_TOP_CROP)
        y2 = int(ph * self.PLAYFIELD_BOTTOM_CROP)

        return play[y1:y2, x1:x2]

    def preprocess(self, img_bgr: np.ndarray, gray: Optional[np.ndarray] = None) -> np.ndarray:
        play = self.get_playfield_gray(img_bgr, gray=gray)

        if self.mode == "low":
            resized = cv2.resize(play, (84, 84), interpolation=cv2.INTER_AREA)
        else:
            resized = cv2.resize(play, (160, 120), interpolation=cv2.INTER_AREA)

        return (resized.astype(np.float32) / 255.0)

    # ✅ Canny: N프레임 캐시 + 다운샘플 ROI 적용
    def ui_panel_present(self, img_bgr: np.ndarray, gray: Optional[np.ndarray] = None) -> bool:
        if gray is None:
            gray = self.gray(img_bgr)
        h, w = gray.shape

        x1 = int(w * self.PLAYFIELD_RIGHT_RATIO)
        panel = gray[:, x1:]
        if panel.size == 0:
            return False

        # ✅ 통계/엣지 계산 모두 다운샘플 ROI로 수행 (더 싸게)
        panel_small = downsample_gray(panel, max_side=160)
        mean, std = mean_std(panel_small)

        edge_ratio = self._ui_edges.edge_ratio(panel_small, frame_idx=self._frame_idx)

        ok = (
            (edge_ratio >= self.UI_EDGE_RATIO_THR)
            and (self.UI_STD_MIN <= std <= self.UI_STD_MAX)
            and (self.UI_MEAN_MIN <= mean <= self.UI_MEAN_MAX)
        )
        return bool(ok)

    def detect_death(self, img_bgr: np.ndarray, gray: Optional[np.ndarray] = None):
        if gray is None:
            gray = self.gray(img_bgr)
        h, w = gray.shape

        full_brightness = gray.mean() / 255.0
        gameover = full_brightness > 0.82

        x1 = int(w * 0.35)
        x2 = int(w * 0.65)
        y1 = int(h * 0.60)
        y2 = int(h * 0.95)

        roi = gray[y1:y2, x1:x2]
        bright = bright_ratio(roi, thr=225)
        hit = bright > 0.020

        return hit, gameover

    def playfield_motion_score(self, prev_play_gray: np.ndarray, curr_play_gray: np.ndarray) -> float:
        diff = cv2.absdiff(prev_play_gray, curr_play_gray)
        return float(diff.mean()) / 255.0

    # ✅ Canny: N프레임 캐시 + 다운샘플 ROI 적용
    def danger_from_playfield(self, play_gray: np.ndarray, return_parts: bool = False):
        h, w = play_gray.shape

        y1 = int(h * 0.60)
        y2 = int(h * 0.98)
        x1 = int(w * 0.20)
        x2 = int(w * 0.80)

        roi = play_gray[y1:y2, x1:x2]
        if roi.size == 0:
            if return_parts:
                return 0.0, 0.0, 0.0, 0.0
            return 0.0

        # ✅ 위험도 측정은 작은 ROI에서도 충분 → 다운샘플해서 계산
        roi_small = downsample_gray(roi, max_side=128)

        edge_ratio = self._danger_edges.edge_ratio(roi_small, frame_idx=self._frame_idx)
        bright = bright_ratio(roi_small, thr=160)
        std_norm = float(roi_small.std()) / 255.0 if roi_small.size > 0 else 0.0

        danger = 0.0
        danger += 4.0 * edge_ratio
        danger += 2.0 * bright
        danger += 1.2 * std_norm
        danger = max(0.0, min(1.0, danger))

        if return_parts:
            return float(danger), float(edge_ratio), float(bright), float(std_norm)
        return float(danger)

    def _crop_roi(self, gray: np.ndarray, roi):
        if roi is None:
            return gray, 0, 0
        x, y, w, h = roi
        H, W = gray.shape
        x0 = max(0, min(W, int(x)))
        y0 = max(0, min(H, int(y)))
        x1 = max(0, min(W, x0 + int(w)))
        y1 = max(0, min(H, y0 + int(h)))
        return gray[y0:y1, x0:x1], x0, y0

    def is_score_screen(self, img_bgr: np.ndarray, thr: float = 0.75, gray: Optional[np.ndarray] = None) -> bool:
        if self.score_tmpl is None:
            return False
        if gray is None:
            gray = self.gray(img_bgr)

        src, _, _ = self._crop_roi(gray, self.score_roi)

        th, tw = self.score_tmpl.shape[:2]
        if src.shape[0] < th or src.shape[1] < tw:
            return False

        res = cv2.matchTemplate(src, self.score_tmpl, cv2.TM_CCOEFF_NORMED)
        _, maxv, _, _ = cv2.minMaxLoc(res)
        return bool(maxv >= float(thr))
