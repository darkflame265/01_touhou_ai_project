# env/game_env.py
import time
import numpy as np

from env.screen import Screen
from env.controller import press_keys, set_attack_hold, release_all, set_always_slow
from env.actions import ACTIONS

from env.env_state import EnvState
from env.episode_guard import EpisodeGuard
from env.ui_guard import UIGuard
from env.reward_engine import RewardEngine
from env.obs_builder import ObsBuilder

from env.action_masking import ActionMasker, MaskingConfig


class GameEnv:
    """
    루나틱 회피 학습용
    - DUP FRAME SKIP 포함
    - frame_stack을 채널 concat으로 반환 (C_total,H,W)
    """

    def __init__(self, screen_mode="low"):
        self.screen = Screen(mode=screen_mode)

        self.s = EnvState()
        self.guard = EpisodeGuard(self.s)
        self.ui = UIGuard(self.screen, self.s)
        self.reward_engine = RewardEngine(self.s)

        # 반응속도
        self.s.action_repeat = 1
        self.s.frame_sleep = 0.012

        # 관측
        self.obs = ObsBuilder(
            self.screen,
            obs_out_size=128,
            crop_size=256,
            use_fallback_full_preprocess=True,
        )

        # Reward
        self.alive_reward = 0.1
        self.hit_pen = -5.0
        self.death_pen = -5.0
        self.abort_pen = -5.0

        # 위치 shaping
        self.use_position_shaping = True
        self.y_floor = 0.60
        self.y_zone_enter_pen = 1.5
        self.y_zone_stay_pen_k = 0.08

        self.top_soft_y = 0.20
        self.right_soft_x = 0.80
        self.top_pen_k = 0.020
        self.right_pen_k = 0.010
        self.corner_bonus_pen = 0.015

        self._in_y_bad_zone = False
        self._last_y_pen = 0.0
        self._last_pos_pen = 0.0

        # Action Masking
        self.mask_cfg = MaskingConfig(
            margin_px=90,
            use_flip=True,
            top_limit_px=None,
            top_limit_fudge_px=10,
        )
        self.masker = ActionMasker(self.screen, self.obs, self.mask_cfg)

        # 기록
        self.s.exec_action_idx = 0
        self.s.exec_was_masked = False
        self._masked_count = 0
        self._step_count = 0

        self.s.ep_total_reward = 0.0
        self.s.episode_end_reason = ""
        self.s.episode_end_pen = 0.0

        # =========================
        # DUP FRAME SKIP
        # =========================
        self.skip_dup_frames = True
        self.dup_retry = 2
        self.dup_sleep = 0.012
        self.dup_reward_zero = True
        self.dup_thr_mean_abs = 0.05
        self.dup_sample_stride = 8

        # PROFILING
        self._prof_enable = True
        self._prof_t0 = time.perf_counter()
        self._prof_sum_capture = 0.0
        self._prof_sum_ui = 0.0
        self._prof_sum_obs = 0.0
        self._prof_sum_mask = 0.0
        self._prof_sum_ctrl = 0.0
        self._prof_prev_sample = None

        # =========================
        # ✅ “UI=0 이후 다음 죽음에서 종료”를 위한 상태
        # =========================
        self._pending_gameover_after_ui_zero = False   # UI가 0으로 떨어진 뒤, 다음 죽음이 진짜 게임오버
        self._last_ui_lives = None                     # 마지막으로 성공적으로 읽은 UI 값(별 개수)
        self._death_fx_reset_cooldown = 0.25
        self._last_death_fx_reset_t = 0.0

    # -------------------------
    # small helpers
    # -------------------------
    def _as_chw(self, obs: np.ndarray) -> np.ndarray:
        if obs is None:
            return None
        obs = np.asarray(obs)
        if obs.ndim == 2:
            return obs[None, :, :]
        if obs.ndim == 3:
            return obs
        raise ValueError(f"Unexpected obs shape: {obs.shape}")

    def _pack_frames_concat(self) -> np.ndarray:
        if len(self.s.frame_stack) == 0:
            return self._as_chw(self.s.prev_state)
        frames = [self._as_chw(x) for x in list(self.s.frame_stack)]
        return np.concatenate(frames, axis=0)

    def _ep_add(self, x: float):
        try:
            self.s.ep_total_reward += float(x)
        except Exception:
            pass

    def _end_episode(self, pen: float, reason: str):
        self.guard.set_terminated()
        self.s.episode_end_reason = str(reason)
        self.s.episode_end_pen = float(pen)
        self._ep_add(pen)

        self.s.frame_stack.append(self.s.prev_state)
        packed = self._pack_frames_concat()
        return packed, float(pen), True

    def _get_playfield_xy_norm_for_shaping(self):
        x_n, y_n = getattr(self.obs, "last_xy_norm", (None, None))
        conf = float(getattr(self.obs, "last_conf", 0.0))
        if x_n is None or y_n is None:
            return None
        return float(x_n), float(y_n), conf

    def _position_shaping_penalty(self, x_n: float, y_n: float) -> float:
        if not self.use_position_shaping:
            self._last_pos_pen = 0.0
            return 0.0

        pen = 0.0
        if y_n < self.top_soft_y:
            d = (self.top_soft_y - y_n) / max(1e-6, self.top_soft_y)
            pen -= self.top_pen_k * float(d)

        if x_n > self.right_soft_x:
            d = (x_n - self.right_soft_x) / max(1e-6, (1.0 - self.right_soft_x))
            pen -= self.right_pen_k * float(d)

        if (y_n < self.top_soft_y) and (x_n > self.right_soft_x):
            pen -= float(self.corner_bonus_pen)

        self._last_pos_pen = float(pen)
        return float(pen)

    def _y_zone_penalty(self, y_n: float, conf: float) -> float:
        self._last_y_pen = 0.0
        bad = (y_n < self.y_floor)

        if bad and (not self._in_y_bad_zone):
            self._in_y_bad_zone = True
            self._last_y_pen -= float(self.y_zone_enter_pen)

        if bad:
            d = (self.y_floor - y_n) / max(1e-6, self.y_floor)
            self._last_y_pen -= float(self.y_zone_stay_pen_k) * float(d)

        if (not bad) and self._in_y_bad_zone:
            self._in_y_bad_zone = False

        return float(self._last_y_pen)

    # -------------------------
    # PROF / DUP
    # -------------------------
    def _prof_sample_frame(self, img: np.ndarray) -> np.ndarray:
        if img is None:
            return None
        if img.ndim == 3:
            ch0 = img[:, :, 0]
        else:
            ch0 = img
        s = int(self.dup_sample_stride)
        return ch0[::s, ::s].astype(np.uint8, copy=False)

    def _prof_update_frame_dup(self, img: np.ndarray) -> bool:
        if not self._prof_enable:
            return False

        sample = self._prof_sample_frame(img)
        if sample is None:
            return False

        if self._prof_prev_sample is None:
            self._prof_prev_sample = sample
            return False

        diff = np.abs(sample.astype(np.int16) - self._prof_prev_sample.astype(np.int16))
        mean_abs = float(diff.mean())
        max_abs = int(diff.max())
        self._prof_prev_sample = sample

        thr = float(self.dup_thr_mean_abs)
        is_dup = (max_abs == 0) or (mean_abs < thr)
        return bool(is_dup)

    def _capture_with_dup_retry(self):
        t0 = time.perf_counter()
        img = self.screen.capture()
        self._prof_sum_capture += (time.perf_counter() - t0)

        is_dup = self._prof_update_frame_dup(img)
        if (not self.skip_dup_frames) or (not is_dup):
            return img, bool(is_dup)

        for _ in range(int(self.dup_retry)):
            if self.dup_sleep > 0:
                time.sleep(float(self.dup_sleep))

            t1 = time.perf_counter()
            img2 = self.screen.capture()
            self._prof_sum_capture += (time.perf_counter() - t1)

            is_dup2 = self._prof_update_frame_dup(img2)
            if not is_dup2:
                return img2, False

            img = img2
            is_dup = True

        return img, True

    # =========================
    # ✅ tracker reset helper
    # =========================
    def _reset_tracker_on_death(self):
        try:
            if hasattr(self.obs, "on_player_death"):
                self.obs.on_player_death()
        except Exception as e:
            print(f"[WARN] obs.on_player_death failed: {e}")

    # =========================
    # ✅ UI 목숨(별) 처리 핵심
    # - 별 감소가 감지되면: 트래커 리셋 + hit_pen
    # - 별이 0이 되는 순간엔 종료하지 말고, "다음 죽음이 진짜 게임오버" 플래그만 켠다
    # =========================
    def _handle_ui_lives(self, ui_now: int, now_ts: float):
        """
        return: (reward_override_or_none)
        """
        hit_cd = float(getattr(self.s, "hit_cooldown", 0.25))
        last_hit = float(getattr(self.s, "last_hit_time", 0.0))
        prev = getattr(self.s, "prev_ui_lives", None)

        # 기록(마지막 UI)
        self._last_ui_lives = int(ui_now)

        # prev 초기화
        if prev is None:
            self.s.prev_ui_lives = int(ui_now)
            # 참고용 동기화(실제 목숨=별+1 가정)
            self.s.lives = int(ui_now) + 1
            return None

        # 쿨다운 중이면 변화 감지 안 하고 최신값만 반영
        if (now_ts - last_hit) <= hit_cd:
            self.s.prev_ui_lives = int(ui_now)
            self.s.lives = int(ui_now) + 1
            return None

        # ✅ 별 감소 감지 = “목숨 깎임”
        if int(ui_now) < int(prev):
            self.s.last_hit_time = float(now_ts)
            self.s.prev_ui_lives = int(ui_now)
            self.s.lives = int(ui_now) + 1  # 참고용

            print(f"ui목숨 감소 감지. 현재 남은 목숨은 : {int(ui_now)}")


            # ✅ 여기! “목숨 깎인 순간” 트래커 초기화
            self._reset_tracker_on_death()

            # ✅ 별이 0이 됐다면: 아직 1 목숨 남아있으니 종료 X
            if int(ui_now) == 0:
                self._pending_gameover_after_ui_zero = True

            return float(self.hit_pen)

        # 감소 아니면 정상 갱신
        self.s.prev_ui_lives = int(ui_now)
        self.s.lives = int(ui_now) + 1
        return None

    # -------------------------
    # Gym API
    # -------------------------
    def reset(self):
        release_all()
        time.sleep(0.5)

        # 기본값(혹시 UI 못 읽을 때 대비)
        self.s.lives = 3
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

        self._in_y_bad_zone = False
        self._last_y_pen = 0.0
        self._last_pos_pen = 0.0

        # ✅ 상태 초기화
        self._pending_gameover_after_ui_zero = False
        self._last_ui_lives = None
        self._last_death_fx_reset_t = 0.0
        self._prof_prev_sample = None

        img, _ = self._capture_with_dup_retry()
        g = self.screen.gray(img)

        t1 = time.perf_counter()
        state = self.obs.make_state(img)
        self._prof_sum_obs += (time.perf_counter() - t1)
        self.s.prev_state = self._as_chw(state)

        t2 = time.perf_counter()
        ui_ok = self.screen.ui_panel_present(img, gray=g)
        ui_lives = self.ui.ui_lives_safe(img, ui_ok)
        self._prof_sum_ui += (time.perf_counter() - t2)

        self.s.prev_ui_lives = ui_lives
        if ui_lives is not None:
            self._last_ui_lives = int(ui_lives)
            # 참고용: 실제 목숨이 별+1인 케이스 반영
            self.s.lives = int(ui_lives) + 1
            # 시작부터 별이 0이면 “다음 죽음이 진짜 게임오버” 모드
            if int(ui_lives) == 0:
                self._pending_gameover_after_ui_zero = True

        self.s.frame_stack.clear()
        for _ in range(self.s.frame_stack_size):
            self.s.frame_stack.append(self.s.prev_state)

        self.s.exec_action_idx = 0
        self.s.exec_was_masked = False
        self._masked_count = 0
        self._step_count = 0

        if hasattr(self.obs, "reset"):
            self.obs.reset()

        release_all()
        set_attack_hold(True)
        set_always_slow(True)

        return self._pack_frames_concat()

    def step(self, action_idx):
        if self.s.episode_terminated:
            self.guard.terminated_step_return()
            set_attack_hold(False)
            for _ in range(6):
                release_all()
                time.sleep(0.02)
            return self._pack_frames_concat(), 0.0, True

        # pre capture (abort check)
        pre_img, pre_is_dup = self._capture_with_dup_retry()

        if self.skip_dup_frames and pre_is_dup:
            ui_ok = True
        else:
            pre_g = self.screen.gray(pre_img)
            t_ui = time.perf_counter()
            ui_ok = self.screen.ui_panel_present(pre_img, gray=pre_g)
            self.ui.update_ui_absent(ui_ok)
            self._prof_sum_ui += (time.perf_counter() - t_ui)

            if self.s.ui_absent_count >= self.s.ui_absent_needed:
                return self._end_episode(self.abort_pen, "ABORT:UI_ABSENT(pre)")

        # initial masking
        t2 = time.perf_counter()
        masked_idx, was_masked, _ = self.masker.apply_action_mask(action_idx, pre_img)
        self._prof_sum_mask += (time.perf_counter() - t2)

        action = ACTIONS[masked_idx]
        self.s.exec_action_idx = int(masked_idx)
        self.s.exec_was_masked = bool(was_masked)
        if was_masked:
            self._masked_count += 1

        t3 = time.perf_counter()
        release_all()
        press_keys(action.value)
        self._prof_sum_ctrl += (time.perf_counter() - t3)

        total_reward = 0.0

        for _ in range(self.s.action_repeat):
            if self.s.frame_sleep > 0:
                time.sleep(self.s.frame_sleep)

            img, is_dup = self._capture_with_dup_retry()

            # DUP frame skip heavy parts
            if self.skip_dup_frames and is_dup:
                reward = 0.0 if self.dup_reward_zero else float(self.alive_reward)
                self.s.frame_stack.append(self.s.prev_state)
                total_reward += float(reward)
                self._ep_add(float(reward))
                self._step_count += 1
                continue

            g = self.screen.gray(img)

            # UI check
            t5 = time.perf_counter()
            ui_ok = self.screen.ui_panel_present(img, gray=g)
            self.ui.update_ui_absent(ui_ok)
            self._prof_sum_ui += (time.perf_counter() - t5)

            if self.s.ui_absent_count >= self.s.ui_absent_needed:
                for _ in range(5):
                    release_all()
                    time.sleep(0.02)
                _, pen_reward, _ = self._end_episode(self.abort_pen, "ABORT:UI_ABSENT(loop)")
                total_reward += pen_reward
                return self._pack_frames_concat(), float(total_reward), True

            # obs
            t6 = time.perf_counter()
            state = self.obs.make_state(img)
            self._prof_sum_obs += (time.perf_counter() - t6)
            state_chw = self._as_chw(state)

            # re-mask
            t7 = time.perf_counter()
            cur_idx, cur_was_masked, _ = self.masker.apply_action_mask(masked_idx, img)
            self._prof_sum_mask += (time.perf_counter() - t7)

            if cur_idx != masked_idx:
                masked_idx = cur_idx
                action = ACTIONS[masked_idx]
                self.s.exec_action_idx = int(masked_idx)
                self.s.exec_was_masked = True
                self._masked_count += 1

                t8 = time.perf_counter()
                release_all()
                press_keys(action.value)
                self._prof_sum_ctrl += (time.perf_counter() - t8)
            elif cur_was_masked:
                self.s.exec_was_masked = True
                self._masked_count += 1

            # base reward
            reward = float(self.alive_reward)
            now = time.time()

            pos = self._get_playfield_xy_norm_for_shaping()
            if pos is not None:
                x_n, y_n, conf = pos
                reward += self._y_zone_penalty(y_n, conf)
                reward += self._position_shaping_penalty(x_n, y_n)

            # =========================
            # ✅ DEATH FX (FLASH)
            # - 모든 “죽음 순간”에 뜸 (마지막 포함)
            # - UI가 0으로 떨어진 뒤라면: 다음 죽음(=FLASH)에서 에피소드 종료
            # =========================
            _, gameover_fx = self.screen.detect_death(img, gray=g)
            if gameover_fx:
                if (now - self._last_death_fx_reset_t) > self._death_fx_reset_cooldown:
                    self._last_death_fx_reset_t = now
                    self._reset_tracker_on_death()

                    # ✅ 여기! "UI 0 이후" 또 죽었으면 = 진짜 게임오버
                    if self._pending_gameover_after_ui_zero:
                        # 현재 프레임에서 ui를 못 읽어도, 마지막으로 읽은 값이 0이면 신뢰
                        if (self._last_ui_lives is not None) and (int(self._last_ui_lives) == 0):
                            for _ in range(3):
                                release_all()
                                time.sleep(0.02)

                            _, pen_reward, _ = self._end_episode(self.death_pen, "DEATH:AFTER_UI_ZERO")
                            total_reward += (reward + pen_reward)

                            self.s.prev_state = state_chw
                            self.s.frame_stack.append(self.s.prev_state)
                            return self._pack_frames_concat(), float(total_reward), True

            # =========================
            # ✅ UI 별(목숨) 감소 감지 (핵심)
            # - ui_now < prev: 트래커 리셋 + hit_pen
            # - ui_now==0이 되더라도 종료하지 않고 "다음 죽음에서 종료" 플래그만 켠다
            # =========================
            t_ui_lives = time.perf_counter()
            ui_now = self.ui.ui_lives_safe(img, ui_ok)
            self._prof_sum_ui += (time.perf_counter() - t_ui_lives)

            if ui_now is not None:
                ro = self._handle_ui_lives(int(ui_now), now)
                if ro is not None:
                    reward = float(ro)

            # state update
            self.s.prev_state = state_chw
            self.s.frame_stack.append(self.s.prev_state)

            total_reward += float(reward)
            self._ep_add(float(reward))
            self._step_count += 1

        self.s.step_i += 1
        return self._pack_frames_concat(), float(total_reward), False
