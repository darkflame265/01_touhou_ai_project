from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Prefer project actions if available
try:
    from env.actions import ACTIONS as PROJECT_ACTIONS
except Exception:
    PROJECT_ACTIONS = None


@dataclass(frozen=True)
class _StopAction:
    name: str = "STOP"


@dataclass
class SimConfig:
    # obs/world
    obs_out_size: int = 128
    world_size: int = 256

    # player
    player_speed: float = 4.0
    player_radius: float = 3.0  # visual radius

    # bullets
    max_bullets: int = 512
    bullet_radius: float = 2.0

    # termination (hitbox)
    hit_radius: float = 4.0
    max_steps: int = 3000

    # boss/emitter
    boss_y: float = 0.18
    boss_x_jitter: float = 0.12
    pattern_hold_steps: int = 240
    difficulty: int = 0  # 0..3 (자동 커리큘럼이 덮어씀)

    # reward
    alive_reward: float = 0.02
    hit_penalty: float = 1.0

    # STOP 제외한 액션 패널티
    move_penalty: float = 0.001

    # background randomization
    bg_enable: bool = True
    bg_noise_amp: float = 18.0
    bg_flicker_amp: float = 22.0

    # seed
    seed: int = 0

    # render(debug)
    render: bool = True
    render_upscale: int = 2
    render_wait: int = 1
    render_window: str = "SIM"

    # episode-wise speed randomization
    speed_randomize: bool = True
    player_speed_min: float = 2.0
    player_speed_max: float = 3.0
    bullet_speed_scale_min: float = 0.80
    bullet_speed_scale_max: float = 1.25
    speed_resample_every_episodes: int = 1


# ----------------------------
# Auto Curriculum
# ----------------------------
@dataclass
class CurriculumConfig:
    enable: bool = True

    # performance metric: EMA of survived steps
    ema_alpha: float = 0.05

    # thresholds in "steps survived" (EMA 기준)
    # (256x256 월드, max_steps=3000 기준)
    up_thresholds: Tuple[int, int, int] = (220, 520, 950)   # 0->1, 1->2, 2->3
    down_thresholds: Tuple[int, int, int] = (140, 380, 760) # 1->0, 2->1, 3->2

    # apply every N episodes
    update_every_episodes: int = 1

    # episode-level knobs (per difficulty level 0..3)
    # max bullets used in _spawn_bullet cap (not array cap)
    max_bullets_levels: Tuple[int, int, int, int] = (120, 220, 340, 480)

    # pattern hold steps (짧을수록 패턴이 자주 바뀜)
    pattern_hold_levels: Tuple[int, int, int, int] = (320, 260, 220, 180)

    # boss x jitter scale
    boss_x_jitter_levels: Tuple[float, float, float, float] = (0.05, 0.09, 0.12, 0.16)

    # speed randomization ranges (player speed)
    player_speed_min_levels: Tuple[float, float, float, float] = (2.8, 2.6, 2.3, 2.0)
    player_speed_max_levels: Tuple[float, float, float, float] = (3.0, 3.0, 3.1, 3.2)

    # speed randomization ranges (bullet speed scale)
    bullet_scale_min_levels: Tuple[float, float, float, float] = (0.75, 0.80, 0.85, 0.90)
    bullet_scale_max_levels: Tuple[float, float, float, float] = (1.05, 1.15, 1.25, 1.35)


class CurriculumManager:
    """
    EMA(생존 steps)를 보고 difficulty(0..3)을 자동 조절.
    - up/down threshold로 히스테리시스 적용.
    """

    def __init__(self, ccfg: CurriculumConfig):
        self.ccfg = ccfg
        self.level = 0
        self.ema_steps: Optional[float] = None
        self._last_update_episode = 0

    def update(self, episode: int, steps_survived: int) -> None:
        cc = self.ccfg
        if not cc.enable:
            return

        every = int(max(1, cc.update_every_episodes))
        if (episode % every) != 0:
            return

        x = float(max(0, int(steps_survived)))
        if self.ema_steps is None:
            self.ema_steps = x
        else:
            a = float(np.clip(cc.ema_alpha, 1e-6, 1.0))
            self.ema_steps = (1.0 - a) * float(self.ema_steps) + a * x

        # hysteresis
        lvl = int(self.level)
        ema = float(self.ema_steps)

        up0, up1, up2 = cc.up_thresholds
        dn0, dn1, dn2 = cc.down_thresholds

        if lvl == 0 and ema >= up0:
            lvl = 1
        elif lvl == 1:
            if ema >= up1:
                lvl = 2
            elif ema <= dn0:
                lvl = 0
        elif lvl == 2:
            if ema >= up2:
                lvl = 3
            elif ema <= dn1:
                lvl = 1
        elif lvl == 3:
            if ema <= dn2:
                lvl = 2

        self.level = int(np.clip(lvl, 0, 3))

    def get_episode_params(self, cfg: SimConfig) -> Dict[str, Any]:
        """
        difficulty 레벨별로 episode 파라미터 리턴.
        """
        cc = self.ccfg
        lvl = int(self.level)

        max_b = int(cc.max_bullets_levels[lvl])
        hold = int(cc.pattern_hold_levels[lvl])
        jitter = float(cc.boss_x_jitter_levels[lvl])

        pmin = float(cc.player_speed_min_levels[lvl])
        pmax = float(cc.player_speed_max_levels[lvl])
        if pmax < pmin:
            pmin, pmax = pmax, pmin

        bmin = float(cc.bullet_scale_min_levels[lvl])
        bmax = float(cc.bullet_scale_max_levels[lvl])
        if bmax < bmin:
            bmin, bmax = bmax, bmin

        return {
            "difficulty": lvl,
            "max_bullets_ep": int(np.clip(max_b, 16, int(cfg.max_bullets))),
            "pattern_hold_steps_ep": int(np.clip(hold, 60, 2000)),
            "boss_x_jitter_ep": float(np.clip(jitter, 0.0, 0.35)),
            "player_speed_min_ep": pmin,
            "player_speed_max_ep": pmax,
            "bullet_scale_min_ep": bmin,
            "bullet_scale_max_ep": bmax,
            "ema_steps": float(self.ema_steps) if self.ema_steps is not None else None,
        }


class PatternEmitter:
    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.pattern_id = 0
        self.phase = 0.0
        self.hold_left = 0

    def reset(self, hold_steps: int):
        self.pattern_id = int(self.rng.integers(0, 2))  # {0,1}
        self.phase = float(self.rng.uniform(0.0, 2.0 * np.pi))
        self.hold_left = int(hold_steps)

    def maybe_switch(self, hold_steps: int):
        self.hold_left -= 1
        if self.hold_left <= 0:
            self.reset(hold_steps)

    def spawn(self, env: "SimEnv"):
        cfg = env.cfg
        d = int(cfg.difficulty)

        base_speed = 2.6 + 0.6 * d
        base_n = 10 + 4 * d
        base_rate = 6 - min(3, d)

        if (env.t % max(1, base_rate)) != 0:
            return

        bx, by = env.boss_xy
        spd_scale = float(env._bullet_speed_scale_ep)

        if int(self.pattern_id) == 0:
            n = base_n + int(self.rng.integers(0, 6))
            spd = base_speed * float(self.rng.uniform(0.9, 1.15)) * spd_scale
            a0 = self.phase + 0.25 * np.sin(0.03 * env.t)
            for k in range(n):
                ang = a0 + 2.0 * np.pi * (k / n)
                env._spawn_bullet(bx, by, spd * np.cos(ang), spd * np.sin(ang))
            self.phase += 0.08
        else:
            n = base_n + 6
            spd = (1.9 + 0.35 * d) * float(self.rng.uniform(0.95, 1.15)) * spd_scale
            a0 = self.phase + 0.12 * env.t
            for k in range(n):
                ang = a0 + 2.0 * np.pi * (k / n)
                env._spawn_bullet(bx, by, spd * np.cos(ang), spd * np.sin(ang))


class SimEnv:
    """
    OBS (4ch float32):
      ch0: gray + marker + meta(x,y,conf)
      ch1: prev gray
      ch2: absdiff
      ch3: gaussian position hint
    """

    def __init__(self, cfg: Optional[SimConfig] = None):
        self.cfg = cfg or SimConfig()
        self.rng = np.random.default_rng(self.cfg.seed)

        self.s = int(self.cfg.obs_out_size)
        self.W = int(self.cfg.world_size)
        self.H = int(self.cfg.world_size)

        self._actions = self._build_actions()  # 8-dir + STOP

        # -------- curriculum --------
        self.curr_cfg = CurriculumConfig(enable=True)
        self.curr = CurriculumManager(self.curr_cfg)

        # per-episode curriculum-applied knobs (defaults)
        self._max_bullets_ep = int(min(self.cfg.max_bullets, 220))
        self._pattern_hold_steps_ep = int(self.cfg.pattern_hold_steps)
        self._boss_x_jitter_ep = float(self.cfg.boss_x_jitter)
        self._player_speed_min_ep = float(self.cfg.player_speed_min)
        self._player_speed_max_ep = float(self.cfg.player_speed_max)
        self._bullet_scale_min_ep = float(self.cfg.bullet_speed_scale_min)
        self._bullet_scale_max_ep = float(self.cfg.bullet_speed_scale_max)
        self._ema_steps_dbg: Optional[float] = None

        # state
        self.t = 0
        self.episode = 0
        self.player_xy = np.zeros(2, np.float32)
        self.boss_xy = np.zeros(2, np.float32)

        # bullets (pool)
        self._cap = int(max(32, self.cfg.max_bullets))
        self._b_pos = np.zeros((self._cap, 2), np.float32)
        self._b_vel = np.zeros((self._cap, 2), np.float32)
        self._b_alive = np.zeros((self._cap,), np.bool_)
        self._b_n = 0

        # marker/meta
        self.meta_patch = 4
        self.marker_half = 2
        self.marker_value = 1.0
        self.marker_min_scale = 0.35
        self.last_xy_norm: Tuple[float, float] = (0.5, 0.78)
        self.last_conf: float = 1.0
        self._uv = (self.s // 2, self.s // 2)

        # obs buffers
        self._obs = np.empty((4, self.s, self.s), np.float32)
        self._prev_gray_u8: Optional[np.ndarray] = None
        self._z_u8 = np.zeros((self.s, self.s), np.uint8)

        # world(gray)
        self._world_gray = np.zeros((self.H, self.W), np.uint8)
        self._bg0: Optional[np.ndarray] = None
        self._bg_phase = float(self.rng.uniform(0.0, 2.0 * np.pi))

        # hint grid cache
        self._grid_xy: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self.player_hint_sigma = 2.0
        self.player_hint_peak = 1.0

        # emitter
        self.emitter = PatternEmitter(self.rng)

        # episode randomized speeds
        self._player_speed_ep = float(self.cfg.player_speed)
        self._bullet_speed_scale_ep = 1.0

        # previous episode outcome (for curriculum update)
        self._prev_episode_steps = 0

    # ----------------- public -----------------
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.emitter = PatternEmitter(self.rng)

        # curriculum update using previous episode survival steps
        # (episode=0인 첫 reset은 prev=0이라 그냥 시작)
        if self.episode > 0:
            self.curr.update(self.episode, int(self._prev_episode_steps))

        self.episode += 1
        self.t = 0
        self.player_xy[:] = (self.W * 0.5, self.H * 0.80)

        # apply curriculum parameters for THIS episode
        self._apply_curriculum_for_episode()

        self._clear_bullets()
        self._maybe_resample_episode_speeds()

        self._set_boss_xy()
        self.emitter.reset(self._pattern_hold_steps_ep)

        self.last_conf = 1.0
        self._update_xy_norm_and_uv()
        self._prev_gray_u8 = None

        self._init_bg()
        return self._build_obs()

    def step(self, action_idx: Any) -> Tuple[np.ndarray, float, bool, Dict]:
        self.t += 1

        dx, dy, is_stop = self._action_to_delta(action_idx)
        self.player_xy[0] = float(np.clip(self.player_xy[0] + dx, 0, self.W - 1))
        self.player_xy[1] = float(np.clip(self.player_xy[1] + dy, 0, self.H - 1))

        self._set_boss_xy()

        self.emitter.spawn(self)
        self.emitter.maybe_switch(self._pattern_hold_steps_ep)

        self._move_bullets()

        hit, dmin = self._check_hit_and_dmin()
        done = bool(hit or (self.t >= int(self.cfg.max_steps)))

        # -------- reward --------
        reward = float(self.cfg.alive_reward)
        if not is_stop:
            reward -= float(self.cfg.move_penalty)
        if hit:
            reward -= float(self.cfg.hit_penalty)

        # store steps for next reset curriculum update
        if done:
            self._prev_episode_steps = int(self.t)

        obs = self._build_obs()
        info = {
            "t": int(self.t),
            "episode": int(self.episode),
            "hit": bool(hit),
            "dmin": float(dmin) if np.isfinite(dmin) else 1e9,
            "bullet_n": int(self._b_n),
            "pattern": int(self.emitter.pattern_id),
            "player_xy": self.player_xy.copy(),
            "boss_xy": self.boss_xy.copy(),
            "is_stop": bool(is_stop),
            "player_speed_ep": float(self._player_speed_ep),
            "bullet_speed_scale_ep": float(self._bullet_speed_scale_ep),

            # curriculum debug
            "curr_level": int(self.curr.level),
            "curr_ema_steps": float(self._ema_steps_dbg) if self._ema_steps_dbg is not None else None,
            "max_bullets_ep": int(self._max_bullets_ep),
            "pattern_hold_steps_ep": int(self._pattern_hold_steps_ep),
        }
        return obs, reward, done, info

    # ----------------- curriculum apply -----------------
    def _apply_curriculum_for_episode(self) -> None:
        p = self.curr.get_episode_params(self.cfg)
        self.cfg.difficulty = int(p["difficulty"])

        self._max_bullets_ep = int(p["max_bullets_ep"])
        self._pattern_hold_steps_ep = int(p["pattern_hold_steps_ep"])
        self._boss_x_jitter_ep = float(p["boss_x_jitter_ep"])

        self._player_speed_min_ep = float(p["player_speed_min_ep"])
        self._player_speed_max_ep = float(p["player_speed_max_ep"])
        self._bullet_scale_min_ep = float(p["bullet_scale_min_ep"])
        self._bullet_scale_max_ep = float(p["bullet_scale_max_ep"])

        self._ema_steps_dbg = p["ema_steps"]

    # ----------------- speed randomize -----------------
    def _maybe_resample_episode_speeds(self) -> None:
        cfg = self.cfg
        if not bool(cfg.speed_randomize):
            self._player_speed_ep = float(cfg.player_speed)
            self._bullet_speed_scale_ep = 1.0
            return

        every = int(max(1, cfg.speed_resample_every_episodes))
        if (self.episode % every) != 0:
            return

        # curriculum-aware ranges
        pmin, pmax = float(self._player_speed_min_ep), float(self._player_speed_max_ep)
        if pmax < pmin:
            pmin, pmax = pmax, pmin
        self._player_speed_ep = float(self.rng.uniform(pmin, pmax))

        bmin, bmax = float(self._bullet_scale_min_ep), float(self._bullet_scale_max_ep)
        if bmax < bmin:
            bmin, bmax = bmax, bmin
        self._bullet_speed_scale_ep = float(self.rng.uniform(bmin, bmax))

    # ----------------- bullets -----------------
    def _clear_bullets(self) -> None:
        self._b_alive[:] = False
        self._b_n = 0

    def _spawn_bullet(self, x: float, y: float, vx: float, vy: float) -> None:
        # curriculum per-episode cap (keeps early episodes easy)
        if self._b_n >= int(self._max_bullets_ep):
            return
        if self._b_n >= self._cap:
            return
        i = self._b_n
        self._b_pos[i] = (float(x), float(y))
        self._b_vel[i] = (float(vx), float(vy))
        self._b_alive[i] = True
        self._b_n += 1

    def _move_bullets(self) -> None:
        n = int(self._b_n)
        if n <= 0:
            return

        pos = self._b_pos[:n]
        vel = self._b_vel[:n]
        alive = self._b_alive[:n]

        pos[alive] += vel[alive]

        x = pos[:, 0]
        y = pos[:, 1]
        inb = (x >= -12) & (x <= self.W + 12) & (y >= -12) & (y <= self.H + 12)
        alive &= inb
        self._b_alive[:n] = alive

        if (self.t % 30) == 0:
            self._compact_bullets()

    def _compact_bullets(self) -> None:
        n = int(self._b_n)
        if n <= 0:
            self._b_n = 0
            return
        alive = self._b_alive[:n]
        if bool(alive.all()):
            return
        idxs = np.nonzero(alive)[0]
        newn = int(idxs.size)
        if newn <= 0:
            self._clear_bullets()
            return
        self._b_pos[:newn] = self._b_pos[idxs]
        self._b_vel[:newn] = self._b_vel[idxs]
        self._b_alive[:newn] = True
        self._b_alive[newn:n] = False
        self._b_n = newn

    # ----------------- hit check -----------------
    def _check_hit_and_dmin(self) -> Tuple[bool, float]:
        n = int(self._b_n)
        if n <= 0:
            return False, float("inf")
        alive = self._b_alive[:n]
        if not bool(alive.any()):
            return False, float("inf")

        pos = self._b_pos[:n][alive]
        d = pos - self.player_xy[None, :]
        dist = np.sqrt(np.sum(d * d, axis=1))

        dmin = float(dist.min()) if dist.size else float("inf")
        hit = bool(np.any(dist <= float(self.cfg.hit_radius)))
        return hit, dmin

    # ----------------- actions -----------------
    def _build_actions(self) -> List[Any]:
        if PROJECT_ACTIONS is None:
            moves8: List[Any] = list(range(8))
        else:
            want = [
                "SLOW_LEFT", "SLOW_RIGHT", "SLOW_UP", "SLOW_DOWN",
                "SLOW_UP_LEFT", "SLOW_UP_RIGHT", "SLOW_DOWN_LEFT", "SLOW_DOWN_RIGHT",
            ]
            by_name = {getattr(a, "name", ""): a for a in PROJECT_ACTIONS}
            picked = [by_name[n] for n in want if n in by_name]
            if len(picked) == 8:
                moves8 = picked
            else:
                tmp = []
                for a in PROJECT_ACTIONS:
                    n = str(getattr(a, "name", "")).upper()
                    if "BOMB" in n:
                        continue
                    tmp.append(a)
                moves8 = tmp[:8] if len(tmp) >= 8 else list(range(8))

        return list(moves8) + [_StopAction()]

    def _action_to_delta(self, action_idx: Any) -> Tuple[float, float, bool]:
        spd = float(self._player_speed_ep)

        if isinstance(action_idx, (int, np.integer)):
            act = self._actions[int(action_idx) % len(self._actions)]
        else:
            act = action_idx

        name = str(getattr(act, "name", "")).upper()

        if name == "STOP":
            self.last_conf = 1.0
            self._update_xy_norm_and_uv()
            return 0.0, 0.0, True

        dx = (-spd if "LEFT" in name else (spd if "RIGHT" in name else 0.0))
        dy = (-spd if "UP" in name else (spd if "DOWN" in name else 0.0))

        if name == "" and isinstance(act, (int, np.integer)):
            m = {
                0: (-spd, 0.0), 1: (spd, 0.0), 2: (0.0, -spd), 3: (0.0, spd),
                4: (-spd, -spd), 5: (spd, -spd), 6: (-spd, spd), 7: (spd, spd),
            }
            dx, dy = m[int(act) % 8]

        if dx and dy:
            s = 1.0 / np.sqrt(2.0)
            dx *= s
            dy *= s

        self.last_conf = 1.0
        self._update_xy_norm_and_uv()
        return float(dx), float(dy), False

    # ----------------- boss -----------------
    def _set_boss_xy(self) -> None:
        # curriculum-aware jitter
        j = float(self._boss_x_jitter_ep)
        bx0 = 0.5 + j * np.sin(0.01 * self.t + 0.7)
        by0 = float(self.cfg.boss_y)
        self.boss_xy[0] = float(np.clip(bx0, 0.1, 0.9) * (self.W - 1))
        self.boss_xy[1] = float(np.clip(by0, 0.05, 0.35) * (self.H - 1))

    # ----------------- bg/render/obs -----------------
    def _init_bg(self) -> None:
        if not bool(self.cfg.bg_enable):
            self._bg0 = None
            return
        base = self.rng.normal(0.0, 1.0, size=(self.H, self.W)).astype(np.float32)
        base = cv2.GaussianBlur(base, (0, 0), sigmaX=6.0, sigmaY=6.0)
        base -= float(base.min())
        base /= float(base.max() + 1e-6)
        self._bg0 = (base * float(self.cfg.bg_noise_amp)).astype(np.uint8)

    def _render_world_gray_u8(self) -> np.ndarray:
        img = self._world_gray
        img.fill(0)

        if self._bg0 is not None:
            flick = float(self.cfg.bg_flicker_amp) * (0.5 + 0.5 * np.sin(0.03 * self.t + self._bg_phase))
            bg = self._bg0.astype(np.float32) + flick
            np.clip(bg, 0, 255, out=bg)
            img[:] = bg.astype(np.uint8)

        cv2.circle(img, (int(self.boss_xy[0]), int(self.boss_xy[1])), 5, 60, -1, lineType=cv2.LINE_AA)

        n = int(self._b_n)
        if n > 0:
            pos = self._b_pos[:n]
            alive = self._b_alive[:n]
            for x, y in pos[alive]:
                c = int(120 + 60 * float(self.rng.uniform(0.0, 1.0)))
                cv2.circle(img, (int(x), int(y)), int(self.cfg.bullet_radius), c, -1, lineType=cv2.LINE_AA)

        cv2.circle(img, (int(self.player_xy[0]), int(self.player_xy[1])), int(self.cfg.player_radius), 220, -1, lineType=cv2.LINE_AA)
        return img

    def _build_obs(self) -> np.ndarray:
        world = self._render_world_gray_u8()

        if bool(self.cfg.render):
            vis = cv2.cvtColor(world, cv2.COLOR_GRAY2BGR)
            up = int(max(1, self.cfg.render_upscale))
            if up != 1:
                vis = cv2.resize(vis, (vis.shape[1] * up, vis.shape[0] * up), interpolation=cv2.INTER_NEAREST)

            txt = (
                f"ep{self.episode} lvl={self.curr.level} ema={self._ema_steps_dbg if self._ema_steps_dbg is not None else -1:.0f} "
                f"pspd={self._player_speed_ep:.2f} bmul={self._bullet_speed_scale_ep:.2f} "
                f"mb={self._max_bullets_ep} hold={self._pattern_hold_steps_ep} diff={self.cfg.difficulty}"
            )
            cv2.putText(vis, txt, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(vis, txt, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.imshow(self.cfg.render_window, vis)
            cv2.waitKey(int(self.cfg.render_wait))

        gray_u8 = cv2.resize(world, (self.s, self.s), interpolation=cv2.INTER_AREA)

        if self._prev_gray_u8 is None or self._prev_gray_u8.shape != gray_u8.shape:
            prev_u8 = self._z_u8
            diff_u8 = self._z_u8
        else:
            prev_u8 = self._prev_gray_u8
            diff_u8 = cv2.absdiff(gray_u8, prev_u8)

        self._obs[0] = gray_u8.astype(np.float32) / 255.0
        self._stamp_marker_and_meta(self._obs[0])
        self._obs[1] = prev_u8.astype(np.float32) / 255.0
        self._obs[2] = diff_u8.astype(np.float32) / 255.0
        self._obs[3] = self._player_hint_map()

        self._prev_gray_u8 = gray_u8
        return self._obs.copy()

    def _stamp_marker_and_meta(self, ch0: np.ndarray) -> None:
        u, v = self._uv
        r = int(self.marker_half)
        if r > 0:
            val = float(self.marker_value) * max(float(self.marker_min_scale), float(np.clip(self.last_conf, 0.0, 1.0)))
            x1, x2 = max(0, u - r), min(self.s, u + r + 1)
            y1, y2 = max(0, v - r), min(self.s, v + r + 1)
            ch0[v, x1:x2] = val
            ch0[y1:y2, u] = val

        p = int(self.meta_patch)
        if p > 0 and ch0.shape[0] >= p and ch0.shape[1] >= (p * 3):
            x_n, y_n = self.last_xy_norm
            c = float(np.clip(self.last_conf, 0.0, 1.0))
            ch0[0:p, 0:p] = float(np.clip(x_n, 0.0, 1.0))
            ch0[0:p, p:2 * p] = float(np.clip(y_n, 0.0, 1.0))
            ch0[0:p, 2 * p:3 * p] = c

    def _get_grid(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._grid_xy is None:
            xs = np.arange(self.s, dtype=np.float32)
            ys = np.arange(self.s, dtype=np.float32)
            self._grid_xy = np.meshgrid(xs, ys)
        return self._grid_xy

    def _player_hint_map(self) -> np.ndarray:
        sigma = float(max(1e-6, self.player_hint_sigma))
        peak = float(np.clip(self.player_hint_peak, 0.0, 1.0))
        xx, yy = self._get_grid()
        u, v = self._uv
        d2 = (xx - float(u)) ** 2 + (yy - float(v)) ** 2
        return (np.exp(-d2 / (2.0 * sigma * sigma)).astype(np.float32) * peak)

    def _update_xy_norm_and_uv(self) -> None:
        x_n = float(np.clip(self.player_xy[0] / max(1.0, (self.W - 1)), 0.0, 1.0))
        y_n = float(np.clip(self.player_xy[1] / max(1.0, (self.H - 1)), 0.0, 1.0))
        self.last_xy_norm = (x_n, y_n)
        self._uv = (int(round(x_n * (self.s - 1))), int(round(y_n * (self.s - 1))))
        self._uv = (int(np.clip(self._uv[0], 0, self.s - 1)), int(np.clip(self._uv[1], 0, self.s - 1)))
