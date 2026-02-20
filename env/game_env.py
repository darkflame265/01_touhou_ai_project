# env/game_env.py
from __future__ import annotations

import time
import numpy as np

from env.screen import Screen
from env.controller import release_all, set_attack_hold, set_always_slow
from env.episode_guard import EpisodeGuard
from env.ui_guard import UIGuard
from env.obs_builder import ObsBuilder

from env.game_env_util.env_state import EnvState
from env.game_env_util.reward_engine import RewardEngine, RewardConfig
from env.game_env_util.action_masking import ActionMasker, MaskingConfig
from env.game_env_util.action_executor import ActionExecutor
from env.game_env_util.frame_skipper import FrameSkipper, FrameSkipperConfig
from env.game_env_util.obs_pack import ObsPacker, ObsPackConfig


class GameEnv:
    """
    적용사항(요청 반영):
      1) ObsPacker frame_stack = 4 고정
      2) risk_mix: alive_reward 직후에 적용
      3) diff-risk: ObsBuilder의 diff_risk_* 속성을 사용해 shaping으로만 추가
         - risk 다음에 diff-risk를 적용 (위험 자체 + 위험 증가 둘 다 싫어하게)
    """

    def __init__(self, screen_mode: str = "low"):
        self.screen = Screen(mode=screen_mode)

        # state / guards
        self.s = EnvState()
        self.guard = EpisodeGuard(self.s)
        self.ui = UIGuard(self.screen, self.s)

        # obs builder (4ch + diff_risk_* 제공)
        self.obs = ObsBuilder(self.screen, obs_out_size=128, crop_size=256, use_fallback_full_preprocess=True)

        # reward engine
        r_cfg = RewardConfig(
            alive_reward=0.03,
            hit_pen=-1.5,
            death_pen=-1.5,
            abort_pen=-1.5,
            use_position_shaping=True,
            y_floor=0.60,
            y_zone_enter_pen=0.5,
            y_zone_stay_pen_k=0.05,
            top_soft_y=0.20,
            right_soft_x=0.80,
            top_pen_k=0.010,
            right_pen_k=0.005,
            corner_bonus_pen=0.008,
            death_fx_reset_cooldown=0.25,
        )
        self.reward_engine = RewardEngine(self.s, r_cfg)

        # masking + action executor
        m_cfg = MaskingConfig(
            margin_px=90,
            use_flip=True,
            top_limit_px=None,
            top_limit_fudge_px=10,
            disable_bomb=True,
            enable_bomb_gate=True,
        )
        self.masker = ActionMasker(self.screen, self.obs, m_cfg, state=self.s)
        self.act = ActionExecutor(self.s, self.masker)

        # frame skipper
        fs_cfg = FrameSkipperConfig(
            skip_dup_frames=True,
            dup_retry=2,
            dup_sleep=0.012,
            dup_thr_mean_abs=0.05,
            dup_sample_stride=8,
            prof_enable=True,
        )
        self.fs = FrameSkipper(self.screen, fs_cfg)

        self.s.frame_stack_size = 4
        # obs packer: frame_stack=4 강제
        p_cfg = ObsPackConfig()
        if hasattr(p_cfg, "frame_stack"):
            p_cfg.frame_stack = 4
        self.packer = ObsPacker(self.s, p_cfg)

        # timing
        self.s.action_repeat = 1
        self.s.frame_sleep = 0.012

        # DUP skip reward policy
        self.dup_reward_zero = False

        # death segment skip
        self.skip_death_segment = True
        self.death_skip_min_sec = 0.30
        self.death_skip_max_sec = 1.20
        self.death_skip_clear_consecutive = 3
        self.death_skip_sleep = 0.012

        # runner mask용 최근 프레임
        self.s.last_action_mask_img = None

        # ===== risk shaping (기존) =====
        self.use_local_risk = True
        self.local_risk_mix_p90 = 0.7
        self.local_risk_mix_p99 = 0.3
        self.global_risk_quantile = 0.90
        self.risk_mix = 1.0

        # ===== diff-risk shaping (신규) =====
        self.use_diff_risk = True
        self.diff_risk_mix = 0.35
        self.diff_risk_use_local = True
        self.diff_risk_clip = 1.0
        # tracker reset throttle for hit-fx path
        self._hit_fx_reset_cooldown = 0.15
        self._last_hit_fx_reset_t = 0.0

    # -------------------------
    # misc helpers
    # -------------------------
    def close(self):
        try:
            release_all()
        except Exception:
            pass
        try:
            self.screen.close()
        except Exception:
            pass

    def _end_episode(self, pen: float, reason: str):
        self.guard.set_terminated()
        self.s.episode_end_reason = str(reason)
        self.s.episode_end_pen = float(pen)
        self.packer.ep_add(float(pen))
        return self.packer.pack_frames_concat(), float(pen), True

    def _safe_reset_tracker(self):
        try:
            if hasattr(self.obs, "on_player_death"):
                self.obs.on_player_death()
        except Exception as e:
            print(f"[WARN] obs.on_player_death failed: {e}")

    # -------------------------
    # reset
    # -------------------------
    def reset(self):
        release_all()
        time.sleep(0.5)

        self.s.lives = 3
        self.s.prev_ui_lives = None
        self.s.last_hit_time = 0.0
        self.s.slow_streak = 0
        self.s.step_i = 0
        self.s.ui_absent_count = 0
        self.s.episode_terminated = False
        self.s.terminate_until = 0.0
        self.s.prev_action_idx = None
        self.s.same_action_count = 0
        self.s.episode_end_reason = ""
        self.s.episode_end_pen = 0.0
        self.s.ep_total_reward = 0.0
        self._last_hit_fx_reset_t = 0.0

        now = time.time()
        self.s.episode_start_time = float(now)
        self.s.bomb_forbid_until = float(now + 5.0)
        self.s.bomb_lock_until = 0.0
        self.s.last_bomb_time = 0.0

        self.fs.reset()
        self.act.reset()

        img, _ = self.fs.capture()
        self.s.last_action_mask_img = img

        g = self.screen.gray(img)
        state = self.obs.make_state(img, action_idx=getattr(self.s, "exec_action_idx", None))
        self.packer.reset_stack_fill(state)

        ui_ok = self.screen.ui_panel_present(img, gray=g)
        ui_lives = self.ui.ui_lives_safe(img, ui_ok)
        self.reward_engine.reset(ui_lives)

        if hasattr(self.obs, "reset"):
            self.obs.reset()

        release_all()
        set_attack_hold(False)
        set_always_slow(True)
        return self.packer.pack_frames_concat()

    # -------------------------
    # death FX skip
    # -------------------------
    def _consume_death_segment(self):
        t0 = time.time()
        clear_streak = 0

        last_img, last_g, last_ui_ok = None, None, True

        try:
            release_all()
            set_attack_hold(False)
        except Exception:
            pass

        while True:
            if self.death_skip_sleep and self.death_skip_sleep > 0:
                time.sleep(float(self.death_skip_sleep))

            img, _ = self.fs.capture()
            if img is None:
                break

            self.s.last_action_mask_img = img
            g = self.screen.gray(img)

            ui_ok = self.screen.ui_panel_present(img, gray=g)
            self.ui.update_ui_absent(ui_ok)
            if self.s.ui_absent_count >= self.s.ui_absent_needed:
                last_img, last_g, last_ui_ok = img, g, ui_ok
                break

            _, gameover_fx = self.screen.detect_death(img, gray=g)
            elapsed = float(time.time() - t0)

            if elapsed >= float(self.death_skip_min_sec) and (not gameover_fx):
                clear_streak += 1
                if clear_streak >= int(self.death_skip_clear_consecutive):
                    last_img, last_g, last_ui_ok = img, g, ui_ok
                    break
            else:
                clear_streak = 0

            if elapsed >= float(self.death_skip_max_sec):
                last_img, last_g, last_ui_ok = img, g, ui_ok
                break

            last_img, last_g, last_ui_ok = img, g, ui_ok

        try:
            release_all()
        except Exception:
            pass

        return last_img, last_g, last_ui_ok

    # -------------------------
    # scalar extractors
    # -------------------------
    def _mix2(self, a: float, b: float, va: float, vb: float) -> float:
        s = float(a) + float(b)
        if s <= 1e-9:
            return 0.0
        a = float(a) / s
        b = float(b) / s
        return float(a * va + b * vb)

    def _get_risk_scalar(self) -> float:
        """
        우선순위:
          1) local valid -> mix(p90,p99)
          2) global q-quantile
          3) 0
        """
        try:
            if self.use_local_risk and bool(getattr(self.obs, "risk_local_valid", False)):
                p90 = float(getattr(self.obs, "risk_local_p90", 0.0))
                p99 = float(getattr(self.obs, "risk_local_p99", 0.0))
                return self._mix2(self.local_risk_mix_p90, self.local_risk_mix_p99, p90, p99)

            risk = getattr(self.obs, "risk_heatmap", None)
            if risk is None:
                return 0.0
            r = risk.astype(np.float32, copy=False)
            q = float(np.clip(self.global_risk_quantile, 0.0, 1.0))
            return float(np.quantile(r, q))
        except Exception:
            return 0.0

    def _get_diff_risk_scalar(self) -> float:
        """
        ObsBuilder가 제공하는 요약값에서 scalar 추출.
        - valid 아니면 0
        - local 우선(기본): diff_risk_local_p90
        - 아니면 global: diff_risk_global_p90
        """
        try:
            if (not self.use_diff_risk) or (not bool(getattr(self.obs, "diff_risk_valid", False))):
                return 0.0

            if bool(self.diff_risk_use_local):
                v = float(getattr(self.obs, "diff_risk_local_p90", 0.0))
            else:
                v = float(getattr(self.obs, "diff_risk_global_p90", 0.0))

            if self.diff_risk_clip is not None:
                v = float(np.clip(v, 0.0, float(self.diff_risk_clip)))
            return v
        except Exception:
            return 0.0

    def _apply_risk_penalty(self, reward: float, risk_v: float, mix: float) -> float:
        if risk_v <= 0.0 or float(mix) <= 0.0:
            return reward
        try:
            return float(reward + float(mix) * float(self.reward_engine.risk_penalty(float(risk_v))))
        except Exception:
            return reward

    # -------------------------
    # step
    # -------------------------
    def step(self, action_idx: int):
        if self.s.episode_terminated:
            self.guard.terminated_step_return()
            set_attack_hold(False)
            for _ in range(6):
                release_all()
                time.sleep(0.02)
            return self.packer.pack_frames_concat(), 0.0, True

        # pre-capture (abort pre-check)
        pre_img, pre_is_dup = self.fs.capture()
        self.s.last_action_mask_img = pre_img

        if not (pre_is_dup and self.fs.cfg.skip_dup_frames):
            pre_g = self.screen.gray(pre_img)
            ui_ok = self.screen.ui_panel_present(pre_img, gray=pre_g)
            self.ui.update_ui_absent(ui_ok)

            if self.s.ui_absent_count >= self.s.ui_absent_needed:
                try:
                    st = self.obs.make_state(pre_img, action_idx=getattr(self.s, "exec_action_idx", None))
                    self.packer.push_prev_state(self.packer.as_chw(st), is_dup=bool(pre_is_dup))
                except Exception:
                    pass
                return self._end_episode(self.reward_engine.abort_pen, "ABORT:UI_ABSENT(pre)")

            ui_lives = self.ui.ui_lives_safe(pre_img, ui_ok)
            set_attack_hold(ui_lives is not None and 0 <= int(ui_lives) <= 8)
        else:
            set_attack_hold(False)

        # initial action
        r0 = self.act.begin(action_idx, pre_img)
        masked_idx = int(r0.masked_idx)

        total_reward = 0.0
        for _ in range(int(self.s.action_repeat)):
            if self.s.frame_sleep > 0:
                time.sleep(float(self.s.frame_sleep))

            img, is_dup = self.fs.capture()
            self.s.last_action_mask_img = img

            # DUP skip
            if is_dup and self.fs.cfg.skip_dup_frames:
                # Even on DUP frames, check UI-life drop so tracker reset is not delayed.
                try:
                    ui_ok_dup = self.screen.ui_panel_present(img)
                    ui_peek_dup = self.ui.ui_lives_safe(img, ui_ok_dup)
                    prev_ui_dup = getattr(self.s, "prev_ui_lives", None)
                    if (
                        ui_peek_dup is not None
                        and prev_ui_dup is not None
                        and int(ui_peek_dup) < int(prev_ui_dup)
                    ):
                        self._safe_reset_tracker()
                except Exception:
                    pass
                r = float(self.reward_engine.alive_reward) if not self.dup_reward_zero else 0.0
                self.packer.push_prev_state(self.s.prev_state, is_dup=True)
                total_reward += r
                self.packer.ep_add(r)
                continue

            g = self.screen.gray(img)

            # UI abort
            ui_ok = self.screen.ui_panel_present(img, gray=g)
            self.ui.update_ui_absent(ui_ok)
            if self.s.ui_absent_count >= self.s.ui_absent_needed:
                for _ in range(5):
                    release_all()
                    time.sleep(0.02)
                try:
                    st = self.obs.make_state(img, action_idx=getattr(self.s, "exec_action_idx", None))
                    self.packer.push_prev_state(self.packer.as_chw(st))
                except Exception:
                    pass

                _, pen_r, _ = self._end_episode(self.reward_engine.abort_pen, "ABORT:UI_ABSENT(loop)")
                total_reward += pen_r
                return self.packer.pack_frames_concat(), float(total_reward), True

            # Immediate tracker reset on UI-life drop so debug/player lock is cleared
            # before state construction in this frame.
            try:
                ui_peek = self.ui.ui_lives_safe(img, ui_ok)
            except Exception:
                ui_peek = None
            try:
                prev_ui = getattr(self.s, "prev_ui_lives", None)
                if (
                    ui_peek is not None
                    and prev_ui is not None
                    and int(ui_peek) < int(prev_ui)
                ):
                    self._safe_reset_tracker()
            except Exception:
                pass

            # obs
            state = self.obs.make_state(img, action_idx=getattr(self.s, "exec_action_idx", None))
            state_chw = self.packer.as_chw(state)

            # ===== reward compose (순서 고정) =====
            reward = float(self.reward_engine.alive_reward)

            # (1) risk shaping: alive 직후
            reward = self._apply_risk_penalty(reward, self._get_risk_scalar(), self.risk_mix)

            # (2) diff-risk shaping: risk 다음 (위험 증가 싫어하게)
            reward = self._apply_risk_penalty(reward, self._get_diff_risk_scalar(), self.diff_risk_mix)

            # (3) position penalties
            x_n, y_n = getattr(self.obs, "last_xy_norm", (None, None))
            if x_n is not None and y_n is not None:
                conf = float(getattr(self.obs, "last_conf", 0.0))
                try:
                    reward += float(self.reward_engine.position_penalties(float(x_n), float(y_n), conf))
                except Exception:
                    pass

            # remask
            r1 = self.act.remask_if_needed(masked_idx, img)
            masked_idx = int(r1.masked_idx)

            # death FX
            hit_fx, gameover_fx = self.screen.detect_death(img, gray=g)
            # Fast path: reset tracker on hit-flash itself (not only UI-life/drop or gameover).
            # This avoids stale lock lingering when UI-lives signal is delayed/noisy.
            if bool(hit_fx):
                now_hit_fx = time.time()
                if (now_hit_fx - float(self._last_hit_fx_reset_t)) >= float(self._hit_fx_reset_cooldown):
                    self._safe_reset_tracker()
                    self._last_hit_fx_reset_t = now_hit_fx
            now_fx = time.time()
            term, reason, pen = self.reward_engine.on_death_fx(
                gameover_fx, now_fx, reset_tracker_cb=self._safe_reset_tracker
            )

            if term:
                self.packer.push_prev_state(state_chw)
                for _ in range(3):
                    release_all()
                    time.sleep(0.02)
                _, pen_r, _ = self._end_episode(float(pen), str(reason))
                total_reward += (reward + pen_r)
                return self.packer.pack_frames_concat(), float(total_reward), True

            # death segment skip
            if self.skip_death_segment and bool(gameover_fx):
                try:
                    reward += float(pen)
                except Exception:
                    pass

                self._safe_reset_tracker()

                last_img, last_g, last_ui_ok = self._consume_death_segment()
                if last_img is None:
                    last_img, last_g, last_ui_ok = img, g, ui_ok

                self.s.last_action_mask_img = last_img
                state2 = self.obs.make_state(last_img, action_idx=getattr(self.s, "exec_action_idx", None))
                state2_chw = self.packer.as_chw(state2)

                # reflect lives after skip
                try:
                    ui_now = self.ui.ui_lives_safe(last_img, last_ui_ok)
                    now_ui = time.time()
                    ro = self.reward_engine.on_ui_lives(ui_now, now_ui, reset_tracker_cb=self._safe_reset_tracker)
                    if ro is not None:
                        reward = float(ro)
                except Exception:
                    pass

                self.packer.push_prev_state(state2_chw)
                total_reward += reward
                self.packer.ep_add(reward)
                self.s.step_i += 1
                return self.packer.pack_frames_concat(), float(total_reward), False

            # UI lives update
            ui_now = self.ui.ui_lives_safe(img, ui_ok)
            now_ui = time.time()
            ro = self.reward_engine.on_ui_lives(ui_now, now_ui, reset_tracker_cb=self._safe_reset_tracker)
            if ro is not None:
                reward = float(ro)

            # push state
            self.packer.push_prev_state(state_chw)
            total_reward += reward
            self.packer.ep_add(reward)

        self.s.step_i += 1
        return self.packer.pack_frames_concat(), float(total_reward), False
