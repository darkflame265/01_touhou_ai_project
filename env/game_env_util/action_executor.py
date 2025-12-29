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

    # ✅ 튜닝 값들
    START_BOMB_FORBID_SEC = 15.0    # 게임 시작 후 폭탄 금지 시간
    BOMB_LOCK_SEC = 2.0           # 폭탄 사용 직후 "이동 입력 정지" 시간
    TRACK_PAUSE_SEC = 1.0         # 트래커 pause+재탐색 시간
    BOMB_COOLDOWN_SEC = 5.0       # ✅ 폭탄 연타 방지(락과 분리!)

    def __init__(self, state, masker: ActionMasker):
        self.s = state
        self.masker = masker
        self.masked_count: int = 0

        # ✅ 락 동안 입력정지 스팸 호출 방지용(한 번만 얼리기)
        self._freeze_active: bool = False
        self._freeze_until: float = 0.0

    def reset(self):
        self.masked_count = 0
        self.s.exec_action_idx = 0
        self.s.exec_was_masked = False

        # reset 때마다 락은 풀어둔다 (env.reset에서 다시 세팅)
        self.s.bomb_lock_until = 0.0
        self.s.last_bomb_time = 0.0

        self._freeze_active = False
        self._freeze_until = 0.0

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

    def _apply_inputs_frozen_once(self, until_ts: float) -> None:
        """
        폭탄 락 동안: 이동 입력을 완전 정지.
        ✅ 매 프레임 release_all/press_keys([]) 하지 말고, 락 시작 시 1회만 적용.
        """
        self._freeze_active = True
        self._freeze_until = float(until_ts)
        try:
            release_all()
        except Exception:
            pass
        try:
            press_keys([])  # 방향키 없음
        except Exception:
            pass

    def _maybe_clear_freeze_flag(self, now: float) -> None:
        """
        락이 끝났으면 내부 플래그만 해제.
        (키 입력은 다음 begin()에서 정상 액션으로 갱신됨)
        """
        if self._freeze_active and now >= self._freeze_until:
            self._freeze_active = False
            self._freeze_until = 0.0

    def begin(self, action_idx: int, img_bgr) -> ActionExecResult:
        action_idx = self._clamp_action_idx(action_idx)
        now = time.time()

        # 락 해제 감지
        self._maybe_clear_freeze_flag(now)

        # ✅ 폭탄 락 동안은 어떤 액션이 와도 입력 정지 유지
        lock_until = float(getattr(self.s, "bomb_lock_until", 0.0))
        if lock_until > now:
            # 락 시작 때만 1회 입력정지
            if not self._freeze_active:
                self._apply_inputs_frozen_once(lock_until)

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

        # ✅ 폭탄 처리
        if self._is_bomb_action(action_enum):
            # (1) 시작 폭탄 금지
            forbid_until = float(getattr(self.s, "bomb_forbid_until", 0.0))
            if now < forbid_until:
                fb = self._find_fallback_idx()
                self.s.exec_action_idx = int(fb)
                self.s.exec_was_masked = True
                self.masked_count += 1
                release_all()
                press_keys(ACTIONS[int(fb)].value)
                return ActionExecResult(masked_idx=int(fb), was_masked=True)

            # (2) ✅ 폭탄 연타 방지: 쿨다운(락과 분리!)
            last = float(getattr(self.s, "last_bomb_time", 0.0))
            if (now - last) < float(self.BOMB_COOLDOWN_SEC):
                fb = self._find_fallback_idx()
                self.s.exec_action_idx = int(fb)
                self.s.exec_was_masked = True
                self.masked_count += 1
                release_all()
                press_keys(ACTIONS[int(fb)].value)
                return ActionExecResult(masked_idx=int(fb), was_masked=True)

            # (3) 폭탄 실행
            release_all()
            press_keys(action_enum.value)

            # (4) 락 걸기 (이동 정지)
            self.s.last_bomb_time = float(now)
            self.s.bomb_lock_until = float(now + float(self.BOMB_LOCK_SEC))

            # (5) 트래커 pause + 재탐색
            obs = getattr(self.masker, "obs", None)
            if obs is not None and hasattr(obs, "on_bomb_used"):
                try:
                    obs.on_bomb_used(pause_sec=float(self.TRACK_PAUSE_SEC))
                except Exception:
                    pass

            # (6) 폭탄 직후부터 이동 정지(락 시작 1회만)
            self._apply_inputs_frozen_once(self.s.bomb_lock_until)
            return ActionExecResult(masked_idx=int(masked_idx), was_masked=bool(was_masked))

        # 일반 액션
        release_all()
        press_keys(action_enum.value)
        return ActionExecResult(masked_idx=int(masked_idx), was_masked=bool(was_masked))

    def remask_if_needed(self, cur_masked_idx: int, img_bgr) -> ActionExecResult:
        cur_masked_idx = self._clamp_action_idx(cur_masked_idx)
        now = time.time()

        # 락 해제 감지
        self._maybe_clear_freeze_flag(now)

        # ✅ 폭탄 락 동안은 입력 정지 유지 (여기서도 "1회만" 적용)
        lock_until = float(getattr(self.s, "bomb_lock_until", 0.0))
        if lock_until > now:
            if not self._freeze_active:
                self._apply_inputs_frozen_once(lock_until)
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
