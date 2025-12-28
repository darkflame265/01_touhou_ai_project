# env/game_env_util/action_executor.py
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple

from env.actions import ACTIONS
from env.controller import press_keys, release_all

# ActionMasker 타입 힌트용(순환 import 방지)
try:
    from env.game_env_util.action_masking import ActionMasker
except Exception:  # pragma: no cover
    ActionMasker = object  # type: ignore


@dataclass
class ActionExecResult:
    masked_idx: int
    was_masked: bool


class ActionExecutor:
    """
    GameEnv에서 '액션 실행' 관련 책임을 분리:
      - apply_action_mask (초기/루프 중)
      - release_all + press_keys 실행
      - state 기록(exec_action_idx / exec_was_masked)
      - 마스킹 카운트 누적

    NOTE:
      - 타이밍/프로파일링(perf_counter)은 GameEnv 쪽에서 감싸도 되고,
        여기서 time.perf_counter()로 추가해도 되는데, 우선 기능만 분리.
    """

    def __init__(self, state, masker: ActionMasker):
        self.s = state
        self.masker = masker
        self.masked_count: int = 0

    def reset(self):
        self.masked_count = 0
        self.s.exec_action_idx = 0
        self.s.exec_was_masked = False

    def _clamp_action_idx(self, action_idx: int) -> int:
        try:
            i = int(action_idx)
        except Exception:
            return 0
        if i < 0 or i >= len(ACTIONS):
            return 0
        return i

    def _press_action_idx(self, action_idx: int):
        idx = self._clamp_action_idx(action_idx)
        action = ACTIONS[idx]
        release_all()
        press_keys(action.value)

    def begin(self, action_idx: int, img_bgr) -> ActionExecResult:
        """
        step() 시작 시 1회:
          - pre_img 기준으로 action mask 적용
          - 해당 액션을 즉시 실행
        """
        action_idx = self._clamp_action_idx(action_idx)

        masked_idx, was_masked, _ = self.masker.apply_action_mask(action_idx, img_bgr)

        self.s.exec_action_idx = int(masked_idx)
        self.s.exec_was_masked = bool(was_masked)
        if was_masked:
            self.masked_count += 1

        self._press_action_idx(masked_idx)
        return ActionExecResult(masked_idx=int(masked_idx), was_masked=bool(was_masked))

    def remask_if_needed(self, cur_masked_idx: int, img_bgr) -> ActionExecResult:
        """
        step() 루프 내부에서:
          - 현재 action_idx 기준으로 다시 마스킹 적용
          - 바뀌면 즉시 키 입력 갱신
        """
        cur_masked_idx = self._clamp_action_idx(cur_masked_idx)

        new_idx, was_masked, _ = self.masker.apply_action_mask(cur_masked_idx, img_bgr)

        changed = (int(new_idx) != int(cur_masked_idx))
        if changed:
            # 새 액션 실행
            self._press_action_idx(int(new_idx))
            self.s.exec_action_idx = int(new_idx)
            self.s.exec_was_masked = True
            self.masked_count += 1
            return ActionExecResult(masked_idx=int(new_idx), was_masked=True)

        # 액션은 그대로인데 "마스킹이 필요했다"는 신호가 오면 기록만 반영
        if was_masked:
            self.s.exec_was_masked = True
            self.masked_count += 1

        self.s.exec_action_idx = int(cur_masked_idx)
        return ActionExecResult(masked_idx=int(cur_masked_idx), was_masked=bool(was_masked))
