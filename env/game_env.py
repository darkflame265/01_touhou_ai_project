# env/game_env.py
import time

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

        # reward engine (여기 값만 조절)
        r_cfg = RewardConfig(
            alive_reward=0.1,
            hit_pen=-5.0,
            death_pen=-5.0,
            abort_pen=-5.0,
            use_position_shaping=True,
            y_floor=0.60,
            y_zone_enter_pen=1.5,
            y_zone_stay_pen_k=0.08,
            top_soft_y=0.20,
            right_soft_x=0.80,
            top_pen_k=0.020,
            right_pen_k=0.010,
            corner_bonus_pen=0.015,
            death_fx_reset_cooldown=0.25,
        )
        self.reward_engine = RewardEngine(self.s, r_cfg)

        # masking + action executor
        m_cfg = MaskingConfig(margin_px=90, use_flip=True, top_limit_px=None, top_limit_fudge_px=10)
        self.masker = ActionMasker(self.screen, self.obs, m_cfg)
        self.act = ActionExecutor(self.s, self.masker)

        # frame skipper (dup skip + profiling)
        fs_cfg = FrameSkipperConfig(
            skip_dup_frames=True,
            dup_retry=2,
            dup_sleep=0.012,
            dup_thr_mean_abs=0.05,
            dup_sample_stride=8,
            prof_enable=True,
        )
        self.fs = FrameSkipper(self.screen, fs_cfg)

        # obs packer (CHW + frame_stack concat + ep reward sum)
        self.packer = ObsPacker(self.s, ObsPackConfig())

        # timing
        self.s.action_repeat = 1
        self.s.frame_sleep = 0.012

        # dup reward policy
        self.dup_reward_zero = True

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
        self.packer.ep_add(pen)
        self.s.frame_stack.append(self.s.prev_state)  # 마지막 프레임 유지
        return self.packer.pack_frames_concat(), float(pen), True

    def reset(self):
        release_all()
        time.sleep(0.5)

        # state defaults (필요한 것만)
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

        # ✅ 시작 타이밍 / 폭탄 금지 타이머
        now = time.time()
        self.s.episode_start_time = float(now)
        self.s.bomb_forbid_until = float(now + 3.0)   # 게임 시작 후 3초 폭탄 금지
        self.s.bomb_lock_until = 0.0
        self.s.last_bomb_time = 0.0

        self.fs.reset()
        self.act.reset()

        img, _ = self.fs.capture()
        g = self.screen.gray(img)

        state = self.obs.make_state(img)
        self.packer.reset_stack_fill(state)

        ui_ok = self.screen.ui_panel_present(img, gray=g)
        ui_lives = self.ui.ui_lives_safe(img, ui_ok)
        self.reward_engine.reset(ui_lives)

        if hasattr(self.obs, "reset"):
            self.obs.reset()

        release_all()
        set_attack_hold(True)
        set_always_slow(True)
        return self.packer.pack_frames_concat()

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
        if not (pre_is_dup and self.fs.cfg.skip_dup_frames):
            pre_g = self.screen.gray(pre_img)
            ui_ok = self.screen.ui_panel_present(pre_img, gray=pre_g)
            self.ui.update_ui_absent(ui_ok)
            if self.s.ui_absent_count >= self.s.ui_absent_needed:
                return self._end_episode(self.reward_engine.abort_pen, "ABORT:UI_ABSENT(pre)")

        # initial mask + key press (여기서 bomb 락/트래킹 pause 트리거됨)
        r0 = self.act.begin(action_idx, pre_img)
        masked_idx = int(r0.masked_idx)

        total_reward = 0.0
        for _ in range(int(self.s.action_repeat)):
            if self.s.frame_sleep > 0:
                time.sleep(float(self.s.frame_sleep))

            img, is_dup = self.fs.capture()

            # DUP skip
            if is_dup and self.fs.cfg.skip_dup_frames:
                r = 0.0 if self.dup_reward_zero else float(self.reward_engine.alive_reward)
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
                _, pen_r, _ = self._end_episode(self.reward_engine.abort_pen, "ABORT:UI_ABSENT(loop)")
                total_reward += pen_r
                return self.packer.pack_frames_concat(), float(total_reward), True

            # obs (ObsBuilder가 bomb pause 처리)
            state = self.obs.make_state(img)
            state_chw = self.packer.as_chw(state)

            # remask / key update (bomb 락 중이면 executor가 입력정지 유지)
            r1 = self.act.remask_if_needed(masked_idx, img)
            masked_idx = int(r1.masked_idx)

            # base reward
            reward = float(self.reward_engine.alive_reward)
            now = time.time()

            # position penalties (RewardEngine)
            x_n, y_n = getattr(self.obs, "last_xy_norm", (None, None))
            if x_n is not None and y_n is not None:
                conf = float(getattr(self.obs, "last_conf", 0.0))
                reward += float(self.reward_engine.position_penalties(float(x_n), float(y_n), conf))

            # death FX (RewardEngine)
            _, gameover_fx = self.screen.detect_death(img, gray=g)
            term, reason, pen = self.reward_engine.on_death_fx(gameover_fx, now, reset_tracker_cb=reset_tracker)
            if term:
                for _ in range(3):
                    release_all()
                    time.sleep(0.02)
                _, pen_r, _ = self._end_episode(float(pen), str(reason))
                total_reward += (reward + pen_r)
                self.packer.push_prev_state(state_chw)
                return self.packer.pack_frames_concat(), float(total_reward), True

            # UI lives (RewardEngine)
            ui_now = self.ui.ui_lives_safe(img, ui_ok)
            ro = self.reward_engine.on_ui_lives(ui_now, now, reset_tracker_cb=reset_tracker)
            if ro is not None:
                reward = float(ro)

            # push state
            self.packer.push_prev_state(state_chw)
            total_reward += reward
            self.packer.ep_add(reward)

        self.s.step_i += 1
        return self.packer.pack_frames_concat(), float(total_reward), False
