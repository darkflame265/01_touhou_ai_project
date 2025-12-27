# env/screen_util/score_screen.py
from __future__ import annotations
from typing import Optional, Tuple

import cv2
import numpy as np

from .roi import crop_roi

Roi = Tuple[int, int, int, int]


class ScoreScreenDetector:
    def __init__(self, template_gray: Optional[np.ndarray], score_roi: Optional[Roi]):
        self.tmpl = template_gray
        self.roi = score_roi

    def is_score_screen(self, gray: np.ndarray, thr: float = 0.75) -> bool:
        if self.tmpl is None:
            return False

        src, _, _ = crop_roi(gray, self.roi)

        th, tw = self.tmpl.shape[:2]
        if src.shape[0] < th or src.shape[1] < tw:
            return False

        res = cv2.matchTemplate(src, self.tmpl, cv2.TM_CCOEFF_NORMED)
        _, maxv, _, _ = cv2.minMaxLoc(res)
        return bool(float(maxv) >= float(thr))
