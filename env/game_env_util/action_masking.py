# env/action_masking.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from env.actions import ACTIONS
from env.game_env_util.playfield_utils import get_playfield_rect_safe


@dataclass
class MaskingConfig:
    """
    레이무가 화면 끝/상단 제한선 근처에 있을 때, 바깥으로 나가는 방향의 액션을 금지하거나 안전한 쪽으로 바꿔주는 장치
    + BOMB는 "탄막이 충분히 많을 때만" 허용하도록 별도 규칙 추가
    """
    margin_px: int = 200
    use_flip: bool = True

    # 상단 제한(원하면 켜기)
    top_limit_px: Optional[int] = None
    top_limit_fudge_px: int = 10

    # ===== BOMB gating =====
    # obs에 bullet_candidate_mask(bool) / risk_heatmap(float)이 있으면 활용한다.
    # 둘 중 하나라도 임계치를 넘으면 BOMB 허용(OR).
    enable_bomb_gate: bool = True

    # bullet_candidate_mask 기준: True 비율이 이 이상이면 BOMB 허용
    #이 수치가 높을수록, 레이무가 폭탄을 덜 사용함.?
    bomb_fill_ratio_thr: float = 0.06  # 기본값: 6% (너무 빡세면 0.03~0.05로 내리면 됨)

    # risk_heatmap 기준: 평균 위험도가 이 이상이면 BOMB 허용
    bomb_risk_mean_thr: float = 0.12   # 기본값: 0.12 (스케일이 다르면 조절 필요)

    # risk_heatmap을 평균 대신 상위 퍼센타일로 볼지
    bomb_use_risk_percentile: bool = True
    bomb_risk_percentile: float = 90.0
    bomb_risk_pctl_thr: float = 0.25   # p90이 이 이상이면 허용

    # 어떤 관측치를 쓸지
    bomb_use_candidate_mask: bool = True
    bomb_use_risk_heatmap: bool = True

    disable_bomb: bool = True   # 학습 중 폭탄 완전 금지



class ActionMasker:
    def __init__(self, screen, obs, cfg: MaskingConfig):
        self.screen = screen
        self.obs = obs
        self.cfg = cfg

        # BOMB 인덱스 캐시(없을 수도 있으니 안전하게)
        self._bomb_idx = None
        for i, a in enumerate(ACTIONS):
            if getattr(a, "name", "") == "BOMB":
                self._bomb_idx = i
                break

    def _action_dir(self, action_enum) -> Tuple[int, int]:
        keys = set(action_enum.value)
        dx = (-1 if "LEFT" in keys else (1 if "RIGHT" in keys else 0))
        dy = (-1 if "UP" in keys else (1 if "DOWN" in keys else 0))
        return dx, dy

    def _flip_action(self, action_enum, flip_x: bool = False, flip_y: bool = False):
        keys = list(action_enum.value)

        def repl(k: str) -> str:
            if flip_x:
                if k == "LEFT":
                    return "RIGHT"
                if k == "RIGHT":
                    return "LEFT"
            if flip_y:
                if k == "UP":
                    return "DOWN"
                if k == "DOWN":
                    return "UP"
            return k

        new_keys = [repl(k) for k in keys]

        for a in ACTIONS:
            if list(a.value) == new_keys:
                return a
        return action_enum

    def _bomb_should_be_allowed(self) -> bool:
        """
        탄막이 충분히 많을 때만 BOMB 허용.
        obs에 다음 중 하나(또는 둘 다)가 있으면 사용:
          - bullet_candidate_mask: bool/0-1 마스크 (H,W)
          - risk_heatmap: float (H,W), 0~1 스케일 가정(아니면 threshold 조절)
        """
        if not self.cfg.enable_bomb_gate:
            return True

        allow_by_fill = False
        allow_by_risk = False

        if self.cfg.bomb_use_candidate_mask:
            bcm = getattr(self.obs, "bullet_candidate_mask", None)
            if bcm is not None:
                arr = np.asarray(bcm)
                if arr.size > 0:
                    # bool / {0,1} / 확률맵 모두 대응
                    fill = float(np.mean(arr > 0.5))
                    allow_by_fill = (fill >= float(self.cfg.bomb_fill_ratio_thr))

        if self.cfg.bomb_use_risk_heatmap:
            rh = getattr(self.obs, "risk_heatmap", None)
            if rh is not None:
                arr = np.asarray(rh, dtype=np.float32)
                if arr.size > 0:
                    if self.cfg.bomb_use_risk_percentile:
                        p = float(np.percentile(arr, float(self.cfg.bomb_risk_percentile)))
                        allow_by_risk = (p >= float(self.cfg.bomb_risk_pctl_thr))
                    else:
                        m = float(np.mean(arr))
                        allow_by_risk = (m >= float(self.cfg.bomb_risk_mean_thr))

        # 둘 중 하나라도 트리거되면 허용
        return bool(allow_by_fill or allow_by_risk)

    def get_action_mask(self, img_bgr) -> np.ndarray:
        mask = np.ones((len(ACTIONS),), dtype=np.bool_)

        pc = getattr(self.obs, "player_center", None)
        if pc is None:
            # 플레이어 위치가 없으면 이동 마스킹은 못 하되,
            # BOMB는 관측치만으로 가능하면 gate는 적용한다.
            # ===== BOMB disable / gating =====
            if self._bomb_idx is not None:
                if bool(getattr(self.cfg, "disable_bomb", False)):
                    mask[self._bomb_idx] = False
                elif self.cfg.enable_bomb_gate:
                    if not self._bomb_should_be_allowed():
                        mask[self._bomb_idx] = False
            return mask

        px, py = int(pc[0]), int(pc[1])
        l, t, r, b = get_playfield_rect_safe(self.screen, img_bgr)

        margin_px = int(self.cfg.margin_px)

        left_d = px - l
        right_d = r - px
        top_d = py - t
        bot_d = b - py

        near_left = (left_d <= margin_px)
        near_right = (right_d <= margin_px)
        near_top = (top_d <= margin_px)
        near_bot = (bot_d <= margin_px)

        # top-limit: py가 너무 위면 UP 방향 금지
        top_forbid = False
        if self.cfg.top_limit_px is not None:
            top_line = t + int(self.cfg.top_limit_px) - int(self.cfg.top_limit_fudge_px)
            top_forbid = (py <= top_line)

        for i, a in enumerate(ACTIONS):
            # BOMB는 방향 마스킹 대상이 아님(별도 규칙)
            if getattr(a, "name", "") == "BOMB":
                continue

            dx, dy = self._action_dir(a)

            if near_left and dx < 0:
                mask[i] = False
            if near_right and dx > 0:
                mask[i] = False
            if near_top and dy < 0:
                mask[i] = False
            if near_bot and dy > 0:
                mask[i] = False

            if top_forbid and dy < 0:
                mask[i] = False

        # ===== BOMB gating 적용 =====
        if self._bomb_idx is not None and self.cfg.enable_bomb_gate:
            if not self._bomb_should_be_allowed():
                mask[self._bomb_idx] = False

        return mask

    def apply_action_mask(self, action_idx: int, img_bgr):
        mask = self.get_action_mask(img_bgr)

        # 0) 범위 방어
        if not (0 <= int(action_idx) < len(ACTIONS)):
            action_idx = 0

        # 1) 원래 액션이 허용이면 그대로
        if bool(mask[int(action_idx)]):
            return int(action_idx), False, mask

        # 1.5) BOMB가 막힌 경우: 대체/flip 하지 말고 안전 fallback으로 즉시 보낸다.
        # (BOMB를 좌우/상하 flip 해봤자 의미 없고, 더 이상한 액션으로 튀는 걸 막기 위함)
        orig = ACTIONS[int(action_idx)]
        if getattr(orig, "name", "") == "BOMB":
            # SLOW_DOWN 우선
            fallback_name = "SLOW_DOWN"
            for i, a in enumerate(ACTIONS):
                if a.name == fallback_name and bool(mask[i]):
                    return int(i), True, mask
            # 아니면 첫 허용 액션
            for i in range(len(ACTIONS)):
                if bool(mask[i]):
                    return int(i), True, mask
            return 0, True, mask

        # 2) flip 기반 대체 시도
        pc = getattr(self.obs, "player_center", None)

        flip_x = False
        flip_y = False

        if pc is not None:
            px, py = int(pc[0]), int(pc[1])
            l, t, r, b = get_playfield_rect_safe(self.screen, img_bgr)

            margin_px = int(self.cfg.margin_px)

            left_d = px - l
            right_d = r - px
            top_d = py - t
            bot_d = b - py

            dx, dy = self._action_dir(orig)

            if (left_d <= margin_px and dx < 0) or (right_d <= margin_px and dx > 0):
                flip_x = True
            if (top_d <= margin_px and dy < 0) or (bot_d <= margin_px and dy > 0):
                flip_y = True

            if self.cfg.top_limit_px is not None:
                top_forbid = (py <= (t + int(self.cfg.top_limit_px)))
                if top_forbid and dy < 0:
                    flip_y = True

        alt = orig
        if self.cfg.use_flip:
            alt = self._flip_action(orig, flip_x=flip_x, flip_y=flip_y)

        try:
            alt_idx = ACTIONS.index(alt)
        except ValueError:
            alt_idx = 0

        if 0 <= alt_idx < len(ACTIONS) and bool(mask[alt_idx]):
            return int(alt_idx), True, mask

        # 3) 마지막 보정: UP 계열이면 DOWN 계열 강제 시도
        if pc is not None:
            dx, dy = self._action_dir(orig)
            if dy < 0:
                forced = self._flip_action(orig, flip_y=True)
                try:
                    forced_idx = ACTIONS.index(forced)
                    if bool(mask[forced_idx]):
                        return int(forced_idx), True, mask
                except ValueError:
                    pass

        # 4) 그래도 안 되면 "가장 무난한" fallback: SLOW_DOWN(있다면)
        fallback_name = "SLOW_DOWN"
        for i, a in enumerate(ACTIONS):
            if a.name == fallback_name and bool(mask[i]):
                return int(i), True, mask

        # 최후: 첫 번째 허용 액션
        for i in range(len(ACTIONS)):
            if bool(mask[i]):
                return int(i), True, mask

        # 진짜 최후(전부 False는 거의 안 나와야 정상)
        return 0, True, mask
