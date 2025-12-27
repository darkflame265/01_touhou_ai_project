# env/screen_util/cache.py
import cv2
import numpy as np
from typing import Optional


class GrayCache:
    """
    같은 img_bgr 객체에 대해 cvtColor를 1번만 수행하도록 캐시.
    Screen.capture()가 새 프레임을 만들 때 reset() 해주면 됨.
    """
    def __init__(self):
        self._src_id = None
        self._gray = None

    def reset(self):
        self._src_id = None
        self._gray = None

    def gray(self, img_bgr: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if img_bgr is None:
            return None
        src_id = id(img_bgr)
        if self._src_id == src_id and self._gray is not None:
            return self._gray
        g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        self._src_id = src_id
        self._gray = g
        return g
