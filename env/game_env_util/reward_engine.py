# env/game_env_util/reward_engine.py
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class RewardConfig:
    # base
    alive_reward: float = 0.1

    # terminal-ish penalties
    hit_pen: float = -5.0
    death_pen: float = -5.0
    abort_pen: float = -5.0

    # position shaping
    use_position_shaping: bool = True

    # y bad zone
    y_floor: float = 0.60
    y_zone_enter_pen: float = 1.5
    y_zone_stay_pen_k: float = 0.08

    # soft edge penalties (top / right + corner)
    top_soft_y: float = 0.20
    right_soft_x: float = 0.80
    top_pen_k: float = 0.020
    right_pen_k: float = 0.010
    corner_bonus_pen: float = 0.015

    # "UI=0 이후 다음 죽음에서 종료"
    death_fx_reset_cooldown: float = 0.25


class RewardEngine:
    """
    GameEnv에서 '보상/패널티/종료 트리거' 관련 로직을 최대한 분리.

    GameEnv는 아래만 하면 됨:
      - reset()에서 engine.reset(ui_lives)
      - step() 루프에서
          reward = engine.alive_reward
          reward += engine.position_penalties(x_n, y_n, conf)
          term = engine.on_death_fx(gameover_fx, now, reset_tracker_cb)
          ui_term_or_hit = engine.on_ui_lives(ui_now, now, reset_tracker_cb)
      - (추가) 기존 danger postprocess는 그대로 engine.postprocess() 호출
    """

    def __init__(self, state, cfg: RewardConfig | None = None):
        self.s = state
        self.cfg = cfg or RewardConfig()

        # 내부 상태들
        self._pending_gameover_after_ui_zero = False
        self._last_ui_lives = None

        self._in_y_bad_zone = False
        self._last_y_pen = 0.0
        self._last_pos_pen = 0.0

        self._last_death_fx_reset_t = 0.0

    # -------------------------
    # properties
    # -------------------------
    @property
    def alive_reward(self) -> float:
        return float(self.cfg.alive_reward)

    @property
    def hit_pen(self) -> float:
        return float(self.cfg.hit_pen)

    @property
    def death_pen(self) -> float:
        return float(self.cfg.death_pen)

    @property
    def abort_pen(self) -> float:
        return float(self.cfg.abort_pen)

    @property
    def last_y_pen(self) -> float:
        return float(self._last_y_pen)

    @property
    def last_pos_pen(self) -> float:
        return float(self._last_pos_pen)

    # -------------------------
    # lifecycle
    # -------------------------
    def reset(self, ui_lives: int | None):
        self._pending_gameover_after_ui_zero = False
        self._last_ui_lives = None

        self._in_y_bad_zone = False
        self._last_y_pen = 0.0
        self._last_pos_pen = 0.0

        self._last_death_fx_reset_t = 0.0

        # EnvState 동기화
        self.s.prev_ui_lives = ui_lives
        if ui_lives is not None:
            self._last_ui_lives = int(ui_lives)
            # 참고용: 실제 목숨이 별+1인 케이스
            self.s.lives = int(ui_lives) + 1
            if int(ui_lives) == 0:
                self._pending_gameover_after_ui_zero = True

    # -------------------------
    # position shaping
    # -------------------------
    def _position_shaping_penalty(self, x_n: float, y_n: float) -> float:
        if not bool(self.cfg.use_position_shaping):
            self._last_pos_pen = 0.0
            return 0.0

        pen = 0.0

        # top soft
        if y_n < float(self.cfg.top_soft_y):
            d = (float(self.cfg.top_soft_y) - y_n) / max(1e-6, float(self.cfg.top_soft_y))
            pen -= float(self.cfg.top_pen_k) * float(d)

        # right soft
        if x_n > float(self.cfg.right_soft_x):
            d = (x_n - float(self.cfg.right_soft_x)) / max(1e-6, (1.0 - float(self.cfg.right_soft_x)))
            pen -= float(self.cfg.right_pen_k) * float(d)

        # corner extra
        if (y_n < float(self.cfg.top_soft_y)) and (x_n > float(self.cfg.right_soft_x)):
            pen -= float(self.cfg.corner_bonus_pen)

        self._last_pos_pen = float(pen)
        return float(pen)

    def _y_zone_penalty(self, y_n: float) -> float:
        self._last_y_pen = 0.0
        bad = (y_n < float(self.cfg.y_floor))

        if bad and (not self._in_y_bad_zone):
            self._in_y_bad_zone = True
            self._last_y_pen -= float(self.cfg.y_zone_enter_pen)

        if bad:
            d = (float(self.cfg.y_floor) - y_n) / max(1e-6, float(self.cfg.y_floor))
            self._last_y_pen -= float(self.cfg.y_zone_stay_pen_k) * float(d)

        if (not bad) and self._in_y_bad_zone:
            self._in_y_bad_zone = False

        return float(self._last_y_pen)

    def position_penalties(self, x_n: float, y_n: float, conf: float = 0.0) -> float:
        # conf는 현재는 미사용(나중에 conf 낮으면 shaping 약화 같은 확장용)
        _ = float(conf)
        return float(self._y_zone_penalty(y_n) + self._position_shaping_penalty(x_n, y_n))

    # -------------------------
    # UI lives handling
    # -------------------------
    def on_ui_lives(self, ui_now: int | None, now_ts: float, reset_tracker_cb=None) -> float | None:
        """
        ui_now < prev 감지하면:
          - reset_tracker_cb() 호출
          - hit_pen 반환 (reward override)
          - ui_now==0이면 pending flag ON (즉시 종료 X)
        """
        if ui_now is None:
            return None

        ui_now = int(ui_now)
        now_ts = float(now_ts)

        hit_cd = float(getattr(self.s, "hit_cooldown", 0.25))
        last_hit = float(getattr(self.s, "last_hit_time", 0.0))
        prev = getattr(self.s, "prev_ui_lives", None)

        self._last_ui_lives = ui_now

        # prev 초기화
        if prev is None:
            self.s.prev_ui_lives = ui_now
            self.s.lives = ui_now + 1
            return None

        # 쿨다운 중: 변화 감지 안 하고 최신값만 반영
        if (now_ts - last_hit) <= hit_cd:
            self.s.prev_ui_lives = ui_now
            self.s.lives = ui_now + 1
            return None

        # 감소 감지 = hit
        if ui_now < int(prev):
            self.s.last_hit_time = now_ts
            self.s.prev_ui_lives = ui_now
            self.s.lives = ui_now + 1

            try:
                print(f"ui목숨 감소 감지. 현재 남은 목숨은 : {ui_now}")
            except Exception:
                pass

            if callable(reset_tracker_cb):
                try:
                    reset_tracker_cb()
                except Exception:
                    pass

            if ui_now == 0:
                self._pending_gameover_after_ui_zero = True

            return float(self.cfg.hit_pen)

        # 감소 아니면 정상 갱신
        self.s.prev_ui_lives = ui_now
        self.s.lives = ui_now + 1
        return None

    # -------------------------
    # death FX handling
    # -------------------------
    def on_death_fx(self, gameover_fx: bool, now_ts: float, reset_tracker_cb=None) -> tuple[bool, str | None, float]:
        """
        DEATH FX(FLASH) 뜨면:
          - 쿨다운 지나면 tracker reset
          - 'UI 0 이후' 플래그가 켜져 있고 last_ui_lives==0 이면 -> 종료 트리거 반환
        return: (should_terminate, reason, terminal_penalty)
        """
        if not bool(gameover_fx):
            return False, None, 0.0

        now_ts = float(now_ts)
        if (now_ts - float(self._last_death_fx_reset_t)) <= float(self.cfg.death_fx_reset_cooldown):
            return False, None, 0.0

        self._last_death_fx_reset_t = now_ts

        if callable(reset_tracker_cb):
            try:
                reset_tracker_cb()
            except Exception:
                pass

        # UI가 0으로 떨어진 이후라면: 다음 죽음에서 종료
        if bool(self._pending_gameover_after_ui_zero):
            if (self._last_ui_lives is not None) and (int(self._last_ui_lives) == 0):
                return True, "DEATH:AFTER_UI_ZERO", float(self.cfg.death_pen)

        return False, None, 0.0

    # -------------------------
    # danger postprocess (기존 그대로)
    # -------------------------
    def postprocess(self, total_reward, avg_danger, is_slow):
        if is_slow:
            self.s.slow_streak += 1
            if avg_danger > 0.12:
                total_reward += 0.20
            elif avg_danger < 0.05:
                total_reward -= 0.10
            if self.s.slow_streak > self.s.slow_streak_max:
                total_reward -= 0.05
        else:
            self.s.slow_streak = 0
            if avg_danger > 0.12:
                total_reward -= 0.07
            elif avg_danger < 0.05:
                total_reward += 0.02
        return total_reward
