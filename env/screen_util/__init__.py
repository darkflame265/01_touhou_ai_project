# env/screen_util/__init__.py

from .window import find_touhou_window, get_client_rect_screen, ClientRectCache
from .cache import GrayCache
from .assets import load_ui_config, load_score_template
from .backends import create_capture_backend
from .debug_dump import dump_capture_debug

# 추가 export
from .metrics import CannyCacheConfig, CannyEdgeRatioCache
from .roi import crop_roi, split_playfield_and_panel, crop_playfield
from .playfield import get_playfield_gray, preprocess_playfield, motion_score
from .death import detect_death
from .score_screen import ScoreScreenDetector
from .ui_panel import UiPanelHeuristics, UiPanelDetector
from .danger import DangerWeights, DangerEstimator

# metrics (너가 이미 가지고 있는 exports 유지)
from .metrics import (
    CannyCacheConfig,
    CannyEdgeRatioCache,
    UiPanelHeuristics,
    DangerWeights,
    ui_panel_present_cached,
    danger_from_playfield_cached,
)

# NEW: features
from .features import (
    get_playfield_gray,
    preprocess_playfield,
    detect_death_from_gray,
    playfield_motion_score,
    ui_panel_present,
    danger_from_playfield,
    is_score_screen_gray,
)
