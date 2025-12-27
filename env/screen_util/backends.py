# env/screen_util/backends.py
import time
from typing import Optional, Tuple

import numpy as np
import cv2

try:
    import dxcam
    _HAS_DXCAM = True
except Exception:
    _HAS_DXCAM = False

try:
    import mss
    _HAS_MSS = True
except Exception:
    _HAS_MSS = False


Rect = Tuple[int, int, int, int]


class CaptureBackend:
    def capture(self, rect: Rect) -> np.ndarray:
        raise NotImplementedError

    def close(self):
        pass


class DxcamBackend(CaptureBackend):
    def __init__(self, rect: Rect, target_fps: int = 60):
        self._dx = dxcam.create(output_color="BGR")
        if self._dx is None:
            raise RuntimeError("dxcam.create() returned None")

        self._region = (int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3]))
        self._target_fps = int(target_fps)
        self._dx.start(region=self._region, target_fps=self._target_fps)

        print("[SCREEN] capture backend = dxcam (BGR) | region=", self._region)

    def _ensure_region(self, rect: Rect):
        rect = (int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3]))
        if rect == self._region:
            return
        try:
            self._dx.stop()
        except Exception:
            pass
        self._region = rect
        self._dx.start(region=self._region, target_fps=self._target_fps)

    def capture(self, rect: Rect) -> np.ndarray:
        self._ensure_region(rect)

        frame = None
        try:
            frame = self._dx.get_latest_frame()
        except Exception:
            frame = None

        if frame is None:
            time.sleep(0.001)
            try:
                frame = self._dx.get_latest_frame()
            except Exception:
                frame = None

        if frame is None:
            # caller가 크기 맞춰서 zero-frame 처리하는 쪽이 더 깔끔하지만,
            # 여기서는 최소한의 안전 프레임 반환
            l, t, r, b = rect
            w = max(1, int(r - l))
            h = max(1, int(b - t))
            return np.zeros((h, w, 3), dtype=np.uint8)

        bgr = frame  # output_color="BGR"
        if not isinstance(bgr, np.ndarray):
            bgr = np.asarray(bgr, dtype=np.uint8)

        if bgr.ndim == 2:
            bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
        elif bgr.ndim == 3 and bgr.shape[2] == 4:
            bgr = bgr[:, :, :3]
        elif bgr.ndim != 3 or bgr.shape[2] != 3:
            l, t, r, b = rect
            w = max(1, int(r - l))
            h = max(1, int(b - t))
            return np.zeros((h, w, 3), dtype=np.uint8)

        if not bgr.flags["C_CONTIGUOUS"]:
            bgr = np.ascontiguousarray(bgr)
        return bgr

    def close(self):
        try:
            self._dx.stop()
        except Exception:
            pass
        self._dx = None


class MssBackend(CaptureBackend):
    def __init__(self):
        self._mss = mss.mss()
        self._monitor = None
        print("[SCREEN] capture backend = mss (fallback)")

    def capture(self, rect: Rect) -> np.ndarray:
        l, t, r, b = rect
        w = max(1, int(r - l))
        h = max(1, int(b - t))

        if self._monitor is None:
            self._monitor = {"left": int(l), "top": int(t), "width": int(w), "height": int(h)}
        else:
            self._monitor["left"] = int(l)
            self._monitor["top"] = int(t)
            self._monitor["width"] = int(w)
            self._monitor["height"] = int(h)

        img = np.asarray(self._mss.grab(self._monitor))  # BGRA
        bgr = img[..., :3]
        if not bgr.flags["C_CONTIGUOUS"]:
            bgr = np.ascontiguousarray(bgr)
        return bgr

    def close(self):
        self._mss = None


def create_capture_backend(rect: Rect, prefer_dxcam: bool = True, dxcam_target_fps: int = 60) -> CaptureBackend:
    if prefer_dxcam and _HAS_DXCAM:
        try:
            return DxcamBackend(rect=rect, target_fps=dxcam_target_fps)
        except Exception as e:
            print("[SCREEN] dxcam init failed -> fallback:", repr(e))

    if _HAS_MSS:
        try:
            return MssBackend()
        except Exception as e:
            print("[SCREEN] mss init failed:", repr(e))

    raise RuntimeError("capture backend init failed: dxcam/mss 모두 사용 불가")
