# env/game_env_util/env_state.py
import numpy as np
from collections import deque
from dataclasses import dataclass, field


@dataclass
class EnvState:
    lives: int = 3
    prev_ui_lives: int | None = None
    prev_ui_lives_raw: int | None = None
    last_hit_time: float = 0.0
    hit_cooldown: float = 0.6

    prev_state: np.ndarray | None = None

    frame_stack_size: int = 4
    frame_stack: deque = field(default_factory=deque)

    slow_streak: int = 0
    slow_streak_max: int = 10

    debug_every: int = 20
    step_i: int = 0

    action_repeat: int = 2
    frame_sleep: float = 0.03

    ui_absent_count: int = 0
    ui_absent_needed: int = 2

    episode_terminated: bool = False
    terminate_until: float = 0.0
    terminate_cooldown_sec: float = 0.8

    prev_action_idx: int | None = None
    same_action_count: int = 0

    # =========================
    # ✅ Bomb / Tracker control
    # =========================
    episode_start_time: float = 0.0          # reset()에서 세팅
    bomb_forbid_until: float = 0.0           # 게임 시작 후 N초 폭탄 금지
    bomb_lock_until: float = 0.0             # 폭탄 사용 직후 N초 입력/트래킹 정지
    last_bomb_time: float = 0.0              # 디버깅/후처리용

    # action executor 기록용 (runner에서 사용)
    exec_action_idx: int = 0
    exec_was_masked: bool = False

    # episode log용
    episode_end_reason: str = ""
    episode_end_pen: float = 0.0
    ep_total_reward: float = 0.0

    def __post_init__(self):
        # ✅ frame_stack_size와 deque(maxlen)를 항상 일치시킨다
        try:
            n = int(self.frame_stack_size)
            if n <= 0:
                n = 1
        except Exception:
            n = 4
            self.frame_stack_size = 4

        if isinstance(self.frame_stack, deque):
            if self.frame_stack.maxlen != n:
                old = list(self.frame_stack)
                self.frame_stack = deque(old[-n:], maxlen=n)
        else:
            self.frame_stack = deque(maxlen=n)
