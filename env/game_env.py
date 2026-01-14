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

        # reward engine (트래커 OFF 실험용)
        r_cfg = RewardConfig(
            # base
            alive_reward=0.03,

            # terminal-ish penalties
            hit_pen=-1.5,
            death_pen=-1.5,
            abort_pen=-1.5,

            # ✅ tracker OFF면 x/y를 못 믿으니 position shaping은 끈다
            use_position_shaping=True,

            # (아래는 use_position_shaping=False면 실질적으로 사용되지 않지만,
            #  나중에 tracker ON으로 되돌릴 때를 위해 값은 유지해둬도 됨)
            y_floor=0.60,
            y_zone_enter_pen=0.5,
            y_zone_stay_pen_k=0.05,

            top_soft_y=0.20,
            right_soft_x=0.80,
            top_pen_k=0.010,
            right_pen_k=0.005,
            corner_bonus_pen=0.008,

            # death fx reset
            death_fx_reset_cooldown=0.25,
        )
        self.reward_engine = RewardEngine(self.s, r_cfg)

        # masking + action executor
        m_cfg = MaskingConfig(
            margin_px=90,
            use_flip=True,
            top_limit_px=None,
            top_limit_fudge_px=10,
            disable_bomb=True,          # ✅ 학습 중 폭탄 완전 금지
            enable_bomb_gate=True,      # 나중에 disable_bomb=False로 바꾸면 게이트 방식으로 바로 복구 가능
        )
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
        # ✅ 중복 프레임도 "살아있음"으로 동일 처리해서 advantage 노이즈를 줄임
        self.dup_reward_zero = False

        # =========================
        # ✅ 죽음 구간 스킵 설정
        # =========================
        self.skip_death_segment = True
        self.death_skip_min_sec = 0.30
        self.death_skip_max_sec = 1.20
        self.death_skip_clear_consecutive = 3
        self.death_skip_sleep = 0.012

        # ✅ runner가 추가 캡처 없이 마스크 계산할 수 있게 "최근 프레임"을 여기에 유지
        # (EnvState에 없더라도 동적 속성으로 붙여도 됨)
        self.s.last_action_mask_img = None

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
        self.s.bomb_forbid_until = float(now + 5.0)
        self.s.bomb_lock_until = 0.0
        self.s.last_bomb_time = 0.0

        self.fs.reset()
        self.act.reset()

        img, _ = self.fs.capture()
        self.s.last_action_mask_img = img  # ✅ runner용 최신 프레임
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

    # =========================
    # ✅ 죽음 구간 스킵: 내부 프레임 소비
    # =========================
    def _consume_death_segment(self):
        """
        죽음/부활 연출 중인 프레임들을 '학습 transition'으로 쌓지 않기 위해,
        내부에서 조용히 캡처를 반복해서 안정화될 때까지 기다린다.
        return: (last_img, last_gray, ui_ok_last)
        """
        t0 = time.time()
        clear_streak = 0

        last_img = None
        last_g = None
        last_ui_ok = True

        # 죽음 연출 동안 입력은 끊는게 안전
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

            # ✅ runner용 최신 프레임
            self.s.last_action_mask_img = img

            if is_dup and self.fs.cfg.skip_dup_frames:
                pass

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
        self.s.last_action_mask_img = pre_img  # ✅ runner용 최신 프레임

        if not (pre_is_dup and self.fs.cfg.skip_dup_frames):
            pre_g = self.screen.gray(pre_img)
            ui_ok = self.screen.ui_panel_present(pre_img, gray=pre_g)
            self.ui.update_ui_absent(ui_ok)
            if self.s.ui_absent_count >= self.s.ui_absent_needed:
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
            self.s.last_action_mask_img = img  # ✅ runner용 최신 프레임

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
                _, pen_r, _ = self._end_episode(self.reward_engine.abort_pen, "ABORT:UI_ABSENT(loop)")
                total_reward += pen_r
                return self.packer.pack_frames_concat(), float(total_reward), True

            # obs
            state = self.obs.make_state(img)
            state_chw = self.packer.as_chw(state)

            # base reward (먼저 정의!)
            reward = float(self.reward_engine.alive_reward)
            now = time.time()

            # ✅ risk_heatmap shaping: 위험할수록 reward 조금 깎기
            #    mean -> p90 (상위 10% 분위수)
            risk = getattr(self.obs, "risk_heatmap", None)
            if risk is not None:
                try:
                    r = risk.astype(np.float32, copy=False)
                    risk_p90 = float(np.quantile(r, 0.90))
                    reward += float(self.reward_engine.risk_penalty(risk_p90))
                except Exception:
                    pass

            # position penalties
            x_n, y_n = getattr(self.obs, "last_xy_norm", (None, None))
            if x_n is not None and y_n is not None:
                conf = float(getattr(self.obs, "last_conf", 0.0))
                reward += float(self.reward_engine.position_penalties(float(x_n), float(y_n), conf))

            # remask / key update  (이건 reward와 무관하니 위/아래 어디든 OK)
            r1 = self.act.remask_if_needed(masked_idx, img)
            masked_idx = int(r1.masked_idx)

            # death FX
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

                # ✅ runner용 최신 프레임(스킵 후 안정화 프레임)
                self.s.last_action_mask_img = last_img

                state2 = self.obs.make_state(last_img)
                state2_chw = self.packer.as_chw(state2)

                try:
                    ui_now = self.ui.ui_lives_safe(last_img, last_ui_ok)
                    ro = self.reward_engine.on_ui_lives(ui_now, time.time(), reset_tracker_cb=reset_tracker)
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
            ro = self.reward_engine.on_ui_lives(ui_now, now, reset_tracker_cb=reset_tracker)
            if ro is not None:
                reward = float(ro)

            # push state
            self.packer.push_prev_state(state_chw)
            total_reward += reward
            self.packer.ep_add(reward)

        self.s.step_i += 1
        return self.packer.pack_frames_concat(), float(total_reward), False
