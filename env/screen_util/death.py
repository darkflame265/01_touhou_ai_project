# env/screen_util/death.py
from __future__ import annotations
from typing import Tuple

import numpy as np


def detect_death(gray: np.ndarray) -> Tuple[bool, bool]:
    """
    return: (hit, gameover)
    """
    h, w = gray.shape[:2]

    full_brightness = float(gray.mean()) / 255.0
    gameover = full_brightness > 0.82

    x1 = int(w * 0.35)
    x2 = int(w * 0.65)
    y1 = int(h * 0.60)
    y2 = int(h * 0.95)

    roi = gray[y1:y2, x1:x2]
    bright_ratio = float((roi > 225).mean()) if roi.size else 0.0
    hit = bright_ratio > 0.020

    return bool(hit), bool(gameover)
