# env/screen.py
import os
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
    # roi / playfield / death / score
    split_playfield_and_panel,
    get_playfield_gray,
    preprocess_playfield,
    motion_score,
    detect_death,
    ScoreScreenDetector,
    # metrics / ui / danger
    CannyCacheConfig,
    CannyEdgeRatioCache,
    UiPanelHeuristics,
    UiPanelDetector,
    DangerWeights,
    DangerEstimator,
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

        # capture rect cache
        self._cap_rect = self._get_client_rect_screen()
        self._cap_i = 0
        self._cap_refresh_every = 30

        # gray cache
        self._gray_cache = GrayCache()

        # zero-frame cache
        self._zero_frame = None
        self._zero_frame_shape = None  # (h,w)

        # config/template
        self.debug_dump_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "debug_caps"))
        self.debug_dump_on_start = False
        self.debug_dump_annotated = True
        self.score_roi = None

        env_dir = os.path.dirname(__file__)
        cfg = load_ui_config(env_dir)
        self.score_roi = cfg["score_roi"]
        self.debug_dump_on_start = cfg["capture_debug_dump_on_start"]
        if cfg["capture_debug_dump_dir"]:
            self.debug_dump_dir = cfg["capture_debug_dump_dir"]
        self.debug_dump_annotated = cfg["capture_debug_dump_annotated"]

        score_tmpl = load_score_template(env_dir)
        self._score_detector = ScoreScreenDetector(score_tmpl, self.score_roi)

        # capture backend
        self._dx_target_fps = 60
        self._backend = create_capture_backend(self._cap_rect, prefer_dxcam=True, dxcam_target_fps=self._dx_target_fps)

        # frame idx
        self._frame_idx = 0

        # UI panel detector (Canny cached)
        ui_edge_cache = CannyEdgeRatioCache(CannyCacheConfig(every_n_frames=3, max_side=160, thr1=80, thr2=160))
        ui_heur = UiPanelHeuristics(
            edge_ratio_thr=self.UI_EDGE_RATIO_THR,
            std_min=self.UI_STD_MIN,
            std_max=self.UI_STD_MAX,
            mean_min=self.UI_MEAN_MIN,
            mean_max=self.UI_MEAN_MAX,
        )
        self._ui_detector = UiPanelDetector(ui_edge_cache, ui_heur)

        # danger estimator (Canny cached)
        danger_edge_cache = CannyEdgeRatioCache(CannyCacheConfig(every_n_frames=3, max_side=160, thr1=50, thr2=120))
        danger_w = DangerWeights(w_edge=4.0, w_bright=2.0, w_std=1.2, bright_thr=160)
        self._danger = DangerEstimator(danger_edge_cache, danger_w)

        if self.debug_dump_on_start:
            self.dump_capture_debug(tag="start")

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
        left, top, right, bottom = self._get_capture_rect()
        w = max(1, int(right - left))
        h = max(1, int(bottom - top))

        self._gray_cache.reset()
        self._frame_idx += 1

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

    # ----------------------------
    # Existing APIs
    # ----------------------------
    def get_playfield_gray(self, img_bgr: np.ndarray, gray: Optional[np.ndarray] = None) -> np.ndarray:
        if gray is None:
            gray = self.gray(img_bgr)
        crops = (self.PLAYFIELD_LEFT_CROP, self.PLAYFIELD_RIGHT_CROP, self.PLAYFIELD_TOP_CROP, self.PLAYFIELD_BOTTOM_CROP)
        return get_playfield_gray(gray, self.PLAYFIELD_RIGHT_RATIO, crops)

    def preprocess(self, img_bgr: np.ndarray, gray: Optional[np.ndarray] = None) -> np.ndarray:
        play = self.get_playfield_gray(img_bgr, gray=gray)
        return preprocess_playfield(play, self.mode)

    def ui_panel_present(self, img_bgr: np.ndarray, gray: Optional[np.ndarray] = None) -> bool:
        if gray is None:
            gray = self.gray(img_bgr)
        _, panel = split_playfield_and_panel(gray, self.PLAYFIELD_RIGHT_RATIO)
        return self._ui_detector.present(panel, self._frame_idx)

    def detect_death(self, img_bgr: np.ndarray, gray: Optional[np.ndarray] = None):
        if gray is None:
            gray = self.gray(img_bgr)
        return detect_death(gray)

    def playfield_motion_score(self, prev_play_gray: np.ndarray, curr_play_gray: np.ndarray) -> float:
        return motion_score(prev_play_gray, curr_play_gray)

    def danger_from_playfield(self, play_gray: np.ndarray, return_parts: bool = False):
        h, w = play_gray.shape[:2]
        y1 = int(h * 0.60)
        y2 = int(h * 0.98)
        x1 = int(w * 0.20)
        x2 = int(w * 0.80)
        roi = play_gray[y1:y2, x1:x2]
        return self._danger.score(roi, self._frame_idx, return_parts=return_parts)

    def is_score_screen(self, img_bgr: np.ndarray, thr: float = 0.75, gray: Optional[np.ndarray] = None) -> bool:
        if gray is None:
            gray = self.gray(img_bgr)
        return self._score_detector.is_score_screen(gray, thr=thr)
