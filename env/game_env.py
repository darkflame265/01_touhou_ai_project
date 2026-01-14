# env/game_env.py
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
    def __init__(self, screen_mode: str = "low"):
        self.screen = Screen(mode=screen_mode)

        # state / guards
        self.s = EnvState()
        self.guard = EpisodeGuard(self.s)
        self.ui = UIGuard(self.screen, self.s)

        # obs builder
        self.obs = ObsBuilder(
            self.screen,
            obs_out_size=128,
            crop_size=256,
            use_fallback_full_preprocess=True,
        )

        # reward engine
        r_cfg = RewardConfig(
            alive_reward=0.03,

            hit_pen=-1.5,
            death_pen=-1.5,
            abort_pen=-1.5,

            # tracker ON
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
        self.masker = ActionMasker(self.screen, self.obs, m_cfg)
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

        # obs packer
        self.packer = ObsPacker(self.s, ObsPackConfig())

        # timing
        self.s.action_repeat = 1
        self.s.frame_sleep = 0.012

        # dup reward policy
        self.dup_reward_zero = False

        # =========================
        # 죽음 구간 스킵
        # =========================
        self.skip_death_segment = True
        self.death_skip_min_sec = 0.30
        self.death_skip_max_sec = 1.20
        self.death_skip_clear_consecutive = 3
        self.death_skip_sleep = 0.012

        # runner가 마스크 계산에 쓸 최근 프레임
        self.s.last_action_mask_img = None

        # =========================
        # ✅ local risk shaping 설정 (초미세 회피용)
        # =========================
        self.use_local_risk = True
        # local p90/p99 혼합 (추천: 0.7/0.3)
        self.local_risk_mix_p90 = 0.7
        self.local_risk_mix_p99 = 0.3
        # local이 없으면 global p90 fallback
        self.global_risk_quantile = 0.90

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
        """
        - terminated 플래그 + 종료 이유/패널티 기록
        - ep reward 누적
        - pack 반환
        """
        self.guard.set_terminated()
        self.s.episode_end_reason = str(reason)
        self.s.episode_end_pen = float(pen)
        self.packer.ep_add(float(pen))
        return self.packer.pack_frames_concat(), float(pen), True

    def reset(self):
        release_all()
        time.sleep(0.5)

        # state defaults
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

        # bomb timers
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

        state = self.obs.make_state(img)
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

    def _consume_death_segment(self):
        t0 = time.time()
        clear_streak = 0

        last_img = None
        last_g = None
        last_ui_ok = True

        try:
            release_all()
            set_attack_hold(False)
        except Exception:
            pass

        while True:
            if self.death_skip_sleep and self.death_skip_sleep > 0:
                time.sleep(float(self.death_skip_sleep))

            img, is_dup = self.fs.capture()
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

    # =========================
    # ✅ risk scalar 뽑기 (local 우선 + fallback)
    # =========================
    def _get_risk_scalar(self) -> float:
        """
        reward_engine.risk_penalty()에 넣을 scalar risk 값을 만든다.
        우선순위:
          1) local valid면: mix = a*p90 + b*p99
          2) 아니면: global p90 (risk_heatmap 전체)
          3) 실패하면: 0
        """
        try:
            if self.use_local_risk and bool(getattr(self.obs, "risk_local_valid", False)):
                p90 = float(getattr(self.obs, "risk_local_p90", 0.0))
                p99 = float(getattr(self.obs, "risk_local_p99", 0.0))
                a = float(self.local_risk_mix_p90)
                b = float(self.local_risk_mix_p99)
                # 혹시 합이 1이 아니어도 안전하게
                s = a + b
                if s > 1e-9:
                    a /= s
                    b /= s
                return a * p90 + b * p99

            risk = getattr(self.obs, "risk_heatmap", None)
            if risk is None:
                return 0.0

            r = risk.astype(np.float32, copy=False)
            q = float(self.global_risk_quantile)
            q = float(np.clip(q, 0.0, 1.0))
            return float(np.quantile(r, q))
        except Exception:
            return 0.0

    def step(self, action_idx: int):
        # already terminated
        if self.s.episode_terminated:
            self.guard.terminated_step_return()
            set_attack_hold(False)
            for _ in range(6):
                release_all()
                time.sleep(0.02)
            return self.packer.pack_frames_concat(), 0.0, True

        def reset_tracker():
            try:
                if hasattr(self.obs, "on_player_death"):
                    self.obs.on_player_death()
            except Exception as e:
                print(f"[WARN] obs.on_player_death failed: {e}")

        # pre-capture (abort pre-check)
        pre_img, pre_is_dup = self.fs.capture()
        self.s.last_action_mask_img = pre_img

        if not (pre_is_dup and self.fs.cfg.skip_dup_frames):
            pre_g = self.screen.gray(pre_img)
            ui_ok = self.screen.ui_panel_present(pre_img, gray=pre_g)
            self.ui.update_ui_absent(ui_ok)
            if self.s.ui_absent_count >= self.s.ui_absent_needed:
                # 마지막 관측 남기기
                try:
                    st = self.obs.make_state(pre_img)
                    self.packer.push_prev_state(self.packer.as_chw(st))
                except Exception:
                    pass
                return self._end_episode(self.reward_engine.abort_pen, "ABORT:UI_ABSENT(pre)")

            ui_lives = self.ui.ui_lives_safe(pre_img, ui_ok)
            if ui_lives is not None and 0 <= int(ui_lives) <= 8:
                set_attack_hold(True)
            else:
                set_attack_hold(False)
        else:
            set_attack_hold(False)

        # initial mask + key press
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
                r = float(self.reward_engine.alive_reward) if not self.dup_reward_zero else 0.0
                self.s.frame_stack.append(self.s.prev_state)
                total_reward += r
                self.packer.ep_add(r)
                continue

            g = self.screen.gray(img)

            # UI presence / abort
            ui_ok = self.screen.ui_panel_present(img, gray=g)
            self.ui.update_ui_absent(ui_ok)
            if self.s.ui_absent_count >= self.s.ui_absent_needed:
                for _ in range(5):
                    release_all()
                    time.sleep(0.02)

                # abort도 마지막 관측 남기기
                try:
                    st = self.obs.make_state(img)
                    self.packer.push_prev_state(self.packer.as_chw(st))
                except Exception:
                    pass

                _, pen_r, _ = self._end_episode(self.reward_engine.abort_pen, "ABORT:UI_ABSENT(loop)")
                total_reward += pen_r
                return self.packer.pack_frames_concat(), float(total_reward), True

            # obs
            state = self.obs.make_state(img)
            state_chw = self.packer.as_chw(state)

            # base reward
            reward = float(self.reward_engine.alive_reward)

            # ✅ risk shaping: local mix 우선
            risk_v = self._get_risk_scalar()
            if risk_v > 0.0:
                try:
                    reward += float(self.reward_engine.risk_penalty(float(risk_v)))
                except Exception:
                    pass

            # position penalties
            x_n, y_n = getattr(self.obs, "last_xy_norm", (None, None))
            if x_n is not None and y_n is not None:
                conf = float(getattr(self.obs, "last_conf", 0.0))
                try:
                    reward += float(self.reward_engine.position_penalties(float(x_n), float(y_n), conf))
                except Exception:
                    pass

            # remask / key update
            r1 = self.act.remask_if_needed(masked_idx, img)
            masked_idx = int(r1.masked_idx)

            # death FX
            _, gameover_fx = self.screen.detect_death(img, gray=g)
            now_fx = time.time()
            term, reason, pen = self.reward_engine.on_death_fx(gameover_fx, now_fx, reset_tracker_cb=reset_tracker)

            if term:
                # 종료 시 현재 관측을 마지막 프레임으로 넣고 끝내기
                self.packer.push_prev_state(state_chw)

                for _ in range(3):
                    release_all()
                    time.sleep(0.02)

                _, pen_r, _ = self._end_episode(float(pen), str(reason))
                total_reward += (reward + pen_r)
                return self.packer.pack_frames_concat(), float(total_reward), True

            # 죽음 연출 스킵
            if self.skip_death_segment and bool(gameover_fx):
                try:
                    reward += float(pen)
                except Exception:
                    pass

                reset_tracker()

                last_img, last_g, last_ui_ok = self._consume_death_segment()
                if last_img is None:
                    last_img, last_g, last_ui_ok = img, g, ui_ok

                self.s.last_action_mask_img = last_img

                state2 = self.obs.make_state(last_img)
                state2_chw = self.packer.as_chw(state2)

                # 스킵 이후 UI lives 반영
                try:
                    ui_now = self.ui.ui_lives_safe(last_img, last_ui_ok)
                    now_ui = time.time()
                    ro = self.reward_engine.on_ui_lives(ui_now, now_ui, reset_tracker_cb=reset_tracker)
                    if ro is not None:
                        reward = float(ro)
                except Exception:
                    pass

                self.packer.push_prev_state(state2_chw)
                total_reward += reward
                self.packer.ep_add(reward)

                self.s.step_i += 1
                return self.packer.pack_frames_concat(), float(total_reward), False

            # UI lives
            ui_now = self.ui.ui_lives_safe(img, ui_ok)
            now_ui = time.time()
            ro = self.reward_engine.on_ui_lives(ui_now, now_ui, reset_tracker_cb=reset_tracker)
            if ro is not None:
                reward = float(ro)

            # push state
            self.packer.push_prev_state(state_chw)
            total_reward += reward
            self.packer.ep_add(reward)

        self.s.step_i += 1
        return self.packer.pack_frames_concat(), float(total_reward), False
