# env/screen_util/roi.py
from __future__ import annotations
from typing import Optional, Tuple

import numpy as np

Rect = Tuple[int, int, int, int]
Roi = Tuple[int, int, int, int]


def crop_roi(gray: np.ndarray, roi: Optional[Roi]):
    """
    roi=(x,y,w,h)로 gray를 안전하게 crop
    return: (cropped, x0, y0)
    """
    if roi is None:
        return gray, 0, 0
    x, y, w, h = roi
    H, W = gray.shape[:2]
    x0 = max(0, min(W, int(x)))
    y0 = max(0, min(H, int(y)))
    x1 = max(0, min(W, x0 + int(w)))
    y1 = max(0, min(H, y0 + int(h)))
    return gray[y0:y1, x0:x1], x0, y0


def split_playfield_and_panel(gray: np.ndarray, playfield_right_ratio: float):
    """
    gray를 (playfield, panel)로 분리.
    """
    h, w = gray.shape[:2]
    x2 = int(w * float(playfield_right_ratio))
    play = gray[:, :x2]
    panel = gray[:, x2:]
    return play, panel


def crop_playfield(play_gray: np.ndarray, crops: Tuple[float, float, float, float]) -> np.ndarray:
    """
    crops=(left, right, top, bottom) 비율로 playfield를 crop.
    """
    left, right, top, bottom = crops
    ph, pw = play_gray.shape[:2]
    x1 = int(pw * float(left))
    x2 = int(pw * float(right))
    y1 = int(ph * float(top))
    y2 = int(ph * float(bottom))
    return play_gray[y1:y2, x1:x2]
