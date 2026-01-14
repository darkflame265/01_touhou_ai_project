# env/game_env_util/frame_skipper.py
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class FrameSkipperConfig:
    # DUP frame skip
    skip_dup_frames: bool = True
    dup_retry: int = 2
    dup_sleep: float = 0.012
    dup_thr_mean_abs: float = 0.05
    dup_sample_stride: int = 8

    # profiling
    prof_enable: bool = True


class FrameSkipper:
    """
    Screen.capture()를 감싸서:
      - DUP frame 감지(샘플링 기반) + 재시도
      - profiling 합산(캡처 시간만)

    ✅ 변경점:
      - dup로 판정된 프레임에서는 _prev_sample을 업데이트하지 않음
        -> retry가 "마지막 유효(비-dup) 프레임"을 기준으로 계속 비교하게 됨
    """

    def __init__(self, screen, cfg: FrameSkipperConfig):
        self.screen = screen
        self.cfg = cfg

        # profiling
        self.sum_capture = 0.0

        # dup detection
        self._prev_sample: Optional[np.ndarray] = None

    def reset(self):
        self._prev_sample = None
        self.sum_capture = 0.0

    # -------------------------
    # internals
    # -------------------------
    def _sample_frame(self, img: np.ndarray) -> Optional[np.ndarray]:
        if img is None:
            return None
        if img.ndim == 3:
            ch0 = img[:, :, 0]
        else:
            ch0 = img
        s = int(self.cfg.dup_sample_stride)
        s = max(1, s)
        return ch0[::s, ::s].astype(np.uint8, copy=False)

    def _is_dup(self, img: np.ndarray) -> bool:
        if (not self.cfg.prof_enable) and (not self.cfg.skip_dup_frames):
            return False

        sample = self._sample_frame(img)
        if sample is None:
            return False

        if self._prev_sample is None:
            self._prev_sample = sample
            return False

        diff = np.abs(sample.astype(np.int16) - self._prev_sample.astype(np.int16))
        mean_abs = float(diff.mean())
        max_abs = int(diff.max())

        thr = float(self.cfg.dup_thr_mean_abs)
        is_dup = bool((max_abs == 0) or (mean_abs < thr))

        # ✅ 핵심: dup면 prev 갱신하지 않음 (retry의 비교 기준 고정)
        if not is_dup:
            self._prev_sample = sample

        return is_dup

    def _capture_once(self) -> np.ndarray:
        t0 = time.perf_counter()
        img = self.screen.capture()
        if self.cfg.prof_enable:
            self.sum_capture += (time.perf_counter() - t0)
        return img

    # -------------------------
    # public
    # -------------------------
    def capture(self) -> Tuple[np.ndarray, bool]:
        """
        returns: (img, is_dup)
        """
        img = self._capture_once()
        is_dup = self._is_dup(img)

        if (not self.cfg.skip_dup_frames) or (not is_dup):
            return img, bool(is_dup)

        # retry
        for _ in range(int(self.cfg.dup_retry)):
            if self.cfg.dup_sleep > 0:
                time.sleep(float(self.cfg.dup_sleep))

            img2 = self._capture_once()
            is_dup2 = self._is_dup(img2)
            if not is_dup2:
                return img2, False

            img = img2
            is_dup = True

        return img, True
