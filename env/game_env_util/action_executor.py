# env/game_env_util/action_executor.py
from __future__ import annotations

import time
from dataclasses import dataclass

from env.actions import ACTIONS
from env.controller import press_keys, release_all

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
    - action mask 적용
    - 키 입력 갱신
    - ✅ 폭탄 사용 시:
        1) 입력을 N초 동안 완전 정지 (레이무 이동 정지)
        2) ObsBuilder 트래커를 N초 동안 정지시키고, 종료 시 reset으로 재탐색
    """

    # ✅ 네 요구사항 값
    START_BOMB_FORBID_SEC = 3.0   # 게임 시작 후 3초 폭탄 금지
    BOMB_LOCK_SEC = 3.0          # 폭탄 사용 후 3초 입력/트래킹 정지

    def __init__(self, state, masker: ActionMasker):
        self.s = state
        self.masker = masker
        self.masked_count: int = 0

    def reset(self):
        self.masked_count = 0
        self.s.exec_action_idx = 0
        self.s.exec_was_masked = False
        # reset 때마다 락은 풀어둔다 (env.reset에서 다시 세팅)
        try:
            self.s.bomb_lock_until = 0.0
            self.s.last_bomb_time = 0.0
        except Exception:
            pass

    def _clamp_action_idx(self, action_idx: int) -> int:
        try:
            i = int(action_idx)
        except Exception:
            return 0
        if i < 0 or i >= len(ACTIONS):
            return 0
        return i

    def _find_fallback_idx(self) -> int:
        # "SLOW_DOWN" 있으면 그걸 우선, 없으면 0
        for i, a in enumerate(ACTIONS):
            if getattr(a, "name", "") == "SLOW_DOWN":
                return int(i)
        return 0

    def _is_bomb_action(self, action_enum) -> bool:
        """
        폭탄 액션 판별:
        - name에 BOMB 포함
        - value에 'X' 또는 'BOMB' 문자열 포함
        (네 코드가 어떤 형태로 정의됐든 최대한 안전하게 잡는다)
        """
        try:
            name = str(getattr(action_enum, "name", "")).upper()
            if "BOMB" in name:
                return True
        except Exception:
            pass

        try:
            keys = set(action_enum.value)
            keys_u = {str(k).upper() for k in keys}
            if "X" in keys_u or "BOMB" in keys_u:
                return True
        except Exception:
            pass

        return False

    def _apply_inputs_frozen(self) -> None:
        """
        폭탄 락 동안: 이동 입력을 완전 정지.
        - release_all() 후 press_keys([])로 (공격홀드/always_slow 정책은 controller 내부에서 유지)
        """
        try:
            release_all()
        except Exception:
            pass
        try:
            press_keys([])  # 방향키 없음
        except Exception:
            pass

    def begin(self, action_idx: int, img_bgr) -> ActionExecResult:
        action_idx = self._clamp_action_idx(action_idx)

        now = time.time()

        # ✅ 폭탄 락 동안은 어떤 액션이 와도 입력 정지
        if float(getattr(self.s, "bomb_lock_until", 0.0)) > now:
            self._apply_inputs_frozen()
            self.s.exec_action_idx = int(action_idx)
            self.s.exec_was_masked = False
            return ActionExecResult(masked_idx=int(action_idx), was_masked=False)

        # 마스킹 적용
        masked_idx, was_masked, _ = self.masker.apply_action_mask(action_idx, img_bgr)

        self.s.exec_action_idx = int(masked_idx)
        self.s.exec_was_masked = bool(was_masked)
        if was_masked:
            self.masked_count += 1

        action_enum = ACTIONS[int(masked_idx)]

        # ✅ 폭탄: 시작 3초 금지 + 사용 시 락/트래커 정지
        if self._is_bomb_action(action_enum):
            # 시작 3초 폭탄 금지
            forbid_until = float(getattr(self.s, "bomb_forbid_until", 0.0))
            if now < forbid_until:
                fb = self._find_fallback_idx()
                self.s.exec_action_idx = int(fb)
                self.s.exec_was_masked = True
                self.masked_count += 1
                press_keys(ACTIONS[int(fb)].value)
                return ActionExecResult(masked_idx=int(fb), was_masked=True)

            # 폭탄 실행
            press_keys(action_enum.value)

            # 즉시 락 걸기 (입력/트래킹 정지)
            self.s.last_bomb_time = float(now)
            self.s.bomb_lock_until = float(now + float(self.BOMB_LOCK_SEC))

            # ObsBuilder에 "폭탄 사용" 알림 → 트래킹 pause + 종료 후 reset
            obs = getattr(self.masker, "obs", None)
            if obs is not None and hasattr(obs, "on_bomb_used"):
                try:
                    obs.on_bomb_used(pause_sec=float(self.BOMB_LOCK_SEC))
                except Exception:
                    pass

            # 폭탄 직후부터 이동 정지
            self._apply_inputs_frozen()
            return ActionExecResult(masked_idx=int(masked_idx), was_masked=bool(was_masked))

        # 일반 액션
        release_all()
        press_keys(action_enum.value)
        return ActionExecResult(masked_idx=int(masked_idx), was_masked=bool(was_masked))

    def remask_if_needed(self, cur_masked_idx: int, img_bgr) -> ActionExecResult:
        cur_masked_idx = self._clamp_action_idx(cur_masked_idx)
        now = time.time()

        # ✅ 폭탄 락 동안은 계속 입력 정지 유지
        if float(getattr(self.s, "bomb_lock_until", 0.0)) > now:
            self._apply_inputs_frozen()
            self.s.exec_action_idx = int(cur_masked_idx)
            self.s.exec_was_masked = False
            return ActionExecResult(masked_idx=int(cur_masked_idx), was_masked=False)

        new_idx, was_masked, _ = self.masker.apply_action_mask(cur_masked_idx, img_bgr)

        changed = (int(new_idx) != int(cur_masked_idx))
        if changed:
            action_enum = ACTIONS[int(new_idx)]
            release_all()
            press_keys(action_enum.value)
            self.s.exec_action_idx = int(new_idx)
            self.s.exec_was_masked = True
            self.masked_count += 1
            return ActionExecResult(masked_idx=int(new_idx), was_masked=True)

        if was_masked:
            self.s.exec_was_masked = True
            self.masked_count += 1

        self.s.exec_action_idx = int(cur_masked_idx)
        return ActionExecResult(masked_idx=int(cur_masked_idx), was_masked=bool(was_masked))
