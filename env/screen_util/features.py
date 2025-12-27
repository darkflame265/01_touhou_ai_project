# env/screen_util/features.py
from __future__ import annotations
from typing import Optional, Tuple

import cv2
import numpy as np

# metrics 쪽(너가 이미 만든 것들)
from .metrics import (
    CannyEdgeRatioCache,
    UiPanelHeuristics,
    DangerWeights,
    ui_panel_present_cached,
    danger_from_playfield_cached,
)

Rect = Tuple[int, int, int, int]


def get_playfield_gray(
    gray: np.ndarray,
    playfield_right_ratio: float,
    crops: Tuple[float, float, float, float],
) -> np.ndarray:
    """
    gray에서 playfield 부분만 crop해서 반환.
    crops = (left_crop, right_crop, top_crop, bottom_crop) 0~1
    """
    h, w = gray.shape
    x2 = int(w * float(playfield_right_ratio))
    play = gray[:, :x2]

    ph, pw = play.shape
    left_crop, right_crop, top_crop, bottom_crop = crops

    x1 = int(pw * float(left_crop))
    x2 = int(pw * float(right_crop))
    y1 = int(ph * float(top_crop))
    y2 = int(ph * float(bottom_crop))

    return play[y1:y2, x1:x2]


def preprocess_playfield(play_gray: np.ndarray, mode: str = "low") -> np.ndarray:
    if mode == "low":
        resized = cv2.resize(play_gray, (84, 84), interpolation=cv2.INTER_AREA)
    else:
        resized = cv2.resize(play_gray, (160, 120), interpolation=cv2.INTER_AREA)
    return (resized.astype(np.float32) / 255.0)


def detect_death_from_gray(gray: np.ndarray):
    """
    기존 Screen.detect_death 로직 그대로.
    """
    h, w = gray.shape

    full_brightness = gray.mean() / 255.0
    gameover = full_brightness > 0.82

    x1 = int(w * 0.35)
    x2 = int(w * 0.65)
    y1 = int(h * 0.60)
    y2 = int(h * 0.95)

    roi = gray[y1:y2, x1:x2]
    bright_ratio = float((roi > 225).mean())
    hit = bright_ratio > 0.020

    return hit, gameover


def playfield_motion_score(prev_play_gray: np.ndarray, curr_play_gray: np.ndarray) -> float:
    diff = cv2.absdiff(prev_play_gray, curr_play_gray)
    return float(diff.mean()) / 255.0


def ui_panel_present(
    gray: np.ndarray,
    playfield_right_ratio: float,
    frame_idx: int,
    edge_cache: CannyEdgeRatioCache,
    heur: UiPanelHeuristics,
) -> bool:
    """
    UI 패널(gray[:, x1:])을 잘라서 cached canny + heuristics로 판정.
    """
    h, w = gray.shape
    x1 = int(w * float(playfield_right_ratio))
    panel = gray[:, x1:]
    return ui_panel_present_cached(panel, frame_idx=frame_idx, edge_cache=edge_cache, heur=heur)


def danger_from_playfield(
    play_gray: np.ndarray,
    frame_idx: int,
    edge_cache: CannyEdgeRatioCache,
    weights: DangerWeights,
    return_parts: bool = False,
):
    """
    playfield의 하단 ROI를 잘라서 cached canny + brightness/std로 danger 계산.
    """
    h, w = play_gray.shape
    y1 = int(h * 0.60)
    y2 = int(h * 0.98)
    x1 = int(w * 0.20)
    x2 = int(w * 0.80)

    roi = play_gray[y1:y2, x1:x2]
    return danger_from_playfield_cached(
        roi,
        frame_idx=frame_idx,
        edge_cache=edge_cache,
        weights=weights,
        return_parts=return_parts,
    )


def crop_roi(gray: np.ndarray, roi):
    """
    기존 _crop_roi 동일. roi=(x,y,w,h) or None
    """
    if roi is None:
        return gray, 0, 0
    x, y, w, h = roi
    H, W = gray.shape
    x0 = max(0, min(W, int(x)))
    y0 = max(0, min(H, int(y)))
    x1 = max(0, min(W, x0 + int(w)))
    y1 = max(0, min(H, y0 + int(h)))
    return gray[y0:y1, x0:x1], x0, y0


def is_score_screen_gray(gray: np.ndarray, score_tmpl: np.ndarray, score_roi, thr: float = 0.75) -> bool:
    """
    기존 is_score_screen 로직을 'gray' 기준으로 분리.
    """
    if score_tmpl is None:
        return False

    src, _, _ = crop_roi(gray, score_roi)

    th, tw = score_tmpl.shape[:2]
    if src.shape[0] < th or src.shape[1] < tw:
        return False

    res = cv2.matchTemplate(src, score_tmpl, cv2.TM_CCOEFF_NORMED)
    _, maxv, _, _ = cv2.minMaxLoc(res)
    return bool(maxv >= float(thr))
