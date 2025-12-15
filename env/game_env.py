# env/game_env.py
import time
import numpy as np

from env.screen import Screen
from env.controller import press_keys, release_all
from env.actions import ACTIONS, Action

from env.ui_lives import count_lives_from_img
from collections import deque



class GameEnv:
    def __init__(self, screen_mode="low"):
        self.screen = Screen(mode=screen_mode)


        self.lives = 3
        self.prev_ui_lives = None
        self.last_hit_time = 0.0
        self.hit_cooldown = 1.0  # 중복 hit 방지

        self.prev_state = None

        # 플레이필드 기반 종료 감지용
        self.prev_play_gray = None
        self.no_motion_count = 0

        self.prev_play_mean = None
        self.prev_play_std = None
        self.scene_change_count = 0

        self.frame_stack_size = 4
        self.frame_stack = deque(maxlen=self.frame_stack_size)


    def reset(self):
        release_all()
        time.sleep(0.5)

        self.lives = 3
        self.last_hit_time = 0.0

        img = self.screen.capture()
        state = self.screen.preprocess(img)
        self.prev_state = state

        # UI 잔기 초기화
        self.prev_ui_lives = count_lives_from_img(img)

        # 플레이필드 상태 초기화
        play = self.screen.get_playfield_gray(img)
        self.prev_play_gray = play
        self.no_motion_count = 0

        self.prev_play_mean = float(play.mean())
        self.prev_play_std = float(play.std())
        self.scene_change_count = 0

        # 첫 state는 diff가 없으므로 0으로
        # frame stack 초기화
        self.frame_stack.clear()
        for _ in range(self.frame_stack_size):
            self.frame_stack.append(state)

        stacked_state = np.stack(self.frame_stack, axis=0)
        return stacked_state


    def step(self, action_idx):
        action = ACTIONS[action_idx]

        if action.name.startswith("SLOW"):
            press_keys(action.value)
            time.sleep(0.01)
            release_all()
        else:
            press_keys(action.value)
            time.sleep(0.03)

        img = self.screen.capture()
        state = self.screen.preprocess(img)
        diff_state = np.abs(state - self.prev_state)

        reward = 0.1
        done = False

        now = time.time()
        ui_now = count_lives_from_img(img)

        # --- 움직임 유도 (정지 패널티) ---
        motion_energy = diff_state.mean()

        if motion_energy < 0.002:
            reward -= 0.05   # 가만히 있으면 손해

        if motion_energy > 0.02:
          reward -= 0.02   # 너무 난폭한 움직임 억제
        
        # --- SLOW 이동 보너스 ---
        if action.name.startswith("SLOW"):
            reward += 0.02
            print("[DEBUG] slow 사용함!")
        else:
            # NONE이 아니면(= 실제 이동이면) FAST 패널티
            if action != Action.NONE:
                reward -= 0.005

        # ----------------------------
        # (1) HIT 감지 (보상만 처리)
        # ----------------------------
        if (now - self.last_hit_time) > self.hit_cooldown:
            if ui_now < self.prev_ui_lives:
                self.lives -= 1
                reward = -10
                self.last_hit_time = now
                print(f"[DEBUG] HIT! internal lives={self.lives} (ui {self.prev_ui_lives}->{ui_now})")

        self.prev_ui_lives = ui_now

        # ----------------------------
        # (2) 진짜 게임오버
        # ----------------------------
        if self.lives <= 0:
            reward = -100
            done = True
            print("[DEBUG] GAME OVER! internal lives=0")
            release_all()

        # ----------------------------
        # (3) 컨티뉴 / 로비 감지
        # ----------------------------
        curr_play = self.screen.get_playfield_gray(img)

        motion = self.screen.playfield_motion_score(self.prev_play_gray, curr_play)
        self.prev_play_gray = curr_play

        if motion < 0.004:
            self.no_motion_count += 1
        else:
            self.no_motion_count = 0

        mean_now = float(curr_play.mean())
        std_now = float(curr_play.std())

        d_mean = abs(mean_now - self.prev_play_mean)
        d_std = abs(std_now - self.prev_play_std)

        if d_mean > 6.0 or d_std > 8.0:
            self.scene_change_count += 1
        else:
            self.scene_change_count = 0

        self.prev_play_mean = mean_now
        self.prev_play_std = std_now

        # 컨티뉴 화면
        if self.no_motion_count >= 30:
            print("[DEBUG] EPISODE END: frozen/continue detected")
            reward = -100
            done = True
            release_all()

        # 로비 이동
        if self.scene_change_count >= 3:
            print("[DEBUG] EPISODE END: lobby/scene change detected")
            reward = -100
            done = True
            release_all()

        self.prev_state = state

        self.frame_stack.append(state)
        stacked_state = np.stack(self.frame_stack, axis=0)

        return stacked_state, reward, done

