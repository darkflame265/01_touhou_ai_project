# env/screen_util/ui_panel.py
from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from .metrics import CannyEdgeRatioCache


@dataclass
class UiPanelHeuristics:
    edge_ratio_thr: float = 0.040
    std_min: float = 15.0
    std_max: float = 80.0
    mean_min: float = 20.0
    mean_max: float = 200.0


class UiPanelDetector:
    def __init__(self, edge_cache: CannyEdgeRatioCache, heur: UiPanelHeuristics):
        self.edge_cache = edge_cache
        self.heur = heur

    def present(self, panel_gray: np.ndarray, frame_idx: int) -> bool:
        if panel_gray is None or panel_gray.size == 0:
            return False

        mean = float(panel_gray.mean())
        std = float(panel_gray.std())
        edge_ratio = float(self.edge_cache.edge_ratio(panel_gray, frame_idx))

        h = self.heur
        return bool(
            (edge_ratio >= h.edge_ratio_thr)
            and (h.std_min <= std <= h.std_max)
            and (h.mean_min <= mean <= h.mean_max)
        )
