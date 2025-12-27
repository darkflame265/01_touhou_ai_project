# env/screen_util/danger.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Union

import numpy as np

from .metrics import CannyEdgeRatioCache


@dataclass
class DangerWeights:
    w_edge: float = 4.0
    w_bright: float = 2.0
    w_std: float = 1.2
    bright_thr: int = 160


class DangerEstimator:
    def __init__(self, edge_cache: CannyEdgeRatioCache, weights: DangerWeights):
        self.edge_cache = edge_cache
        self.w = weights

    def score(self, roi_gray: np.ndarray, frame_idx: int, return_parts: bool = False) -> Union[float, Tuple[float, float, float, float]]:
        if roi_gray is None or roi_gray.size == 0:
            return (0.0, 0.0, 0.0, 0.0) if return_parts else 0.0

        edge_ratio = float(self.edge_cache.edge_ratio(roi_gray, frame_idx))
        bright_ratio = float((roi_gray > int(self.w.bright_thr)).mean())
        std_norm = float(roi_gray.std()) / 255.0

        danger = 0.0
        danger += self.w.w_edge * edge_ratio
        danger += self.w.w_bright * bright_ratio
        danger += self.w.w_std * std_norm
        danger = max(0.0, min(1.0, danger))

        if return_parts:
            return float(danger), float(edge_ratio), float(bright_ratio), float(std_norm)
        return float(danger)
