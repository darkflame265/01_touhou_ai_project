# env/screen_util/playfield.py
from __future__ import annotations
from typing import Tuple

import cv2
import numpy as np

from .roi import split_playfield_and_panel, crop_playfield


def get_playfield_gray(gray: np.ndarray, playfield_right_ratio: float, crops: Tuple[float, float, float, float]) -> np.ndarray:
    play, _ = split_playfield_and_panel(gray, playfield_right_ratio)
    return crop_playfield(play, crops)


def preprocess_playfield(play_gray: np.ndarray, mode: str) -> np.ndarray:
    if mode == "low":
        resized = cv2.resize(play_gray, (84, 84), interpolation=cv2.INTER_AREA)
    else:
        resized = cv2.resize(play_gray, (160, 120), interpolation=cv2.INTER_AREA)
    return (resized.astype(np.float32) / 255.0)


def motion_score(prev_play_gray: np.ndarray, curr_play_gray: np.ndarray) -> float:
    diff = cv2.absdiff(prev_play_gray, curr_play_gray)
    return float(diff.mean()) / 255.0
